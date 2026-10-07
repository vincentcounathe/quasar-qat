"""FSDP2 quantization-aware training: healing (KD, frozen non-quantized modules) and adaptation (CE).

Launch with torchrun (``python -m quasar.train --help`` for the options)::

    torchrun --nproc_per_node 8 -m quasar.train --recipe configs/<recipe>.yaml --method quasar \\
        --bits 2 --train_data TRAIN.jsonl --eval_data EVAL.jsonl --out_dir OUT

Writes ``OUT/train_args.json``, ``OUT/log.jsonl`` (also to wandb when
``WANDB_PROJECT`` is set) and the materialized model in ``OUT/materialized/``,
whose ``receipt.json`` holds the final held-out metrics.
"""

from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from quasar.export import save_materialized
from quasar.quant import apply_bitdistiller_clip, enable_shard_local, quant_linears, quantize_model, snapshot_saliency
from quasar.train.config import TrainConfig
from quasar.train.data import calibration_ids, forever, load_rows, make_loader
from quasar.train.objectives import heldout_metrics, heldout_sums, kd_loss


def init_distributed() -> tuple[int, int, torch.device]:
    """Process group from the torchrun environment (NCCL on GPUs, gloo on CPU)."""
    if "RANK" not in os.environ:
        raise RuntimeError("launch with torchrun, e.g. torchrun --nproc_per_node 8 -m quasar.train ...")
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    # Long timeout: rank 0 writes the checkpoint while the other ranks wait.
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2))
    return dist.get_rank(), dist.get_world_size(), device


def load_causal_lm(path: str, cfg: TrainConfig) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation
    )
    model.config.use_cache = False
    return model


def shard(model: nn.Module, mesh) -> None:
    """FSDP2 over each decoder layer and the root: bf16 compute, fp32 gradient reduction.

    LSQ's fp32 step and offset get their own uncast group, since one FSDP group
    needs a single parameter dtype.
    """
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16)
    for _, m in quant_linears(model):
        if any(True for _ in m.quantizer.parameters()):
            fully_shard(m.quantizer, mesh=mesh, mp_policy=MixedPrecisionPolicy())
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)


def freeze_to_quantized(model: nn.Module) -> None:
    """Healing: train only the quantized projections' weights (and LSQ's step/offset)."""
    model.requires_grad_(False)
    for _, m in quant_linears(model):
        m.weight.requires_grad_(True)
        m.quantizer.requires_grad_(True)


def build_student(cfg: TrainConfig, train_rows: list[dict], device: torch.device, mesh) -> nn.Module:
    """Load, quantize, freeze and shard the student. The order matters:

    BitDistiller clips the full-precision weights before quantizers attach;
    quantizers (and LSQ parameters) exist before freezing, sharding and the
    optimizer; QUASAR's shard-local wrapper goes on just before ``fully_shard``.
    """
    model = load_causal_lm(cfg.model_path, cfg).to(device)
    model.gradient_checkpointing_enable()
    qcfg = cfg.quant_config()
    if cfg.method == "bitdistiller":
        apply_bitdistiller_clip(model, calibration_ids(train_rows), qcfg)
    if qcfg is not None:
        quantize_model(model, qcfg)
    if cfg.freeze:
        freeze_to_quantized(model)
    if cfg.fp32_master:
        model.float()
    if cfg.method == "quasar":
        enable_shard_local(model)
    shard(model, mesh)
    return model.train()


def build_teacher(cfg: TrainConfig, device: torch.device, mesh) -> nn.Module:
    teacher = load_causal_lm(cfg.model_path, cfg)
    teacher.requires_grad_(False)
    teacher.to(device)
    shard(teacher, mesh)
    return teacher.eval()


def make_optimizer(params: list[nn.Parameter], cfg: TrainConfig):
    """AdamW (0.9, 0.999) without weight decay; linear warmup, then cosine decay to zero.

    QUASAR's saliency is this optimizer's ``exp_avg_sq``, so beta2 also sets the
    saliency's averaging horizon.
    """
    from transformers import get_cosine_schedule_with_warmup

    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    warmup = round(cfg.warmup_ratio * cfg.max_steps)
    return optimizer, get_cosine_schedule_with_warmup(optimizer, warmup, cfg.max_steps)


def objective(model: nn.Module, teacher: nn.Module | None, batch: dict) -> torch.Tensor:
    """KD against the teacher's logits, or the model's own causal-LM loss (CE) without one."""
    if teacher is None:
        return model(**batch).loss
    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    with torch.no_grad():
        teacher_logits = teacher(**inputs).logits
    return kd_loss(model(**inputs).logits, teacher_logits, batch["labels"])


@torch.no_grad()
def evaluate(model: nn.Module, teacher: nn.Module | None, loader, device: torch.device) -> dict[str, float]:
    """Held-out CE/PPL over all ranks; with a teacher also KL(teacher || student) and top-1 agreement."""
    was_training = model.training
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
        t_logits = teacher(**inputs).logits if teacher is not None else None
        sums += heldout_sums(model(**inputs).logits, batch["labels"].to(device), t_logits)
    dist.all_reduce(sums)
    model.train(was_training)
    return heldout_metrics(sums, teacher is not None)


class RunLog:
    """Rank-0 JSON lines to stdout and ``log.jsonl``, mirrored to wandb when ``WANDB_PROJECT`` is set.

    wandb reads ``WANDB_ENTITY``/``WANDB_RUN_ID`` itself; the run name defaults to
    the output directory's name (``WANDB_NAME`` overrides).
    """

    def __init__(self, out_dir: Path, rank: int, config: dict):
        self.path = out_dir / "log.jsonl" if rank == 0 else None
        self.wandb = None
        if self.path is not None:
            self.path.write_text("")
        if self.path is not None and os.environ.get("WANDB_PROJECT"):
            import wandb

            self.wandb = wandb.init(name=os.environ.get("WANDB_NAME", out_dir.name), config=config, dir=str(out_dir))

    def __call__(self, record: dict) -> None:
        if self.path is None:
            return
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.wandb is not None:
            scalars = {k: v for k, v in record.items() if isinstance(v, (int, float)) and k != "step"}
            self.wandb.log(scalars, step=record["step"])

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def _mean_over_ranks(t: torch.Tensor) -> float:
    dist.all_reduce(t)  # SUM: gloo has no AVG
    return float(t) / dist.get_world_size()


def train(cfg: TrainConfig) -> dict[str, float]:
    """Run one training job; returns the final held-out metrics."""
    from transformers import AutoTokenizer

    rank, world, device = init_distributed()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_args.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    log = RunLog(out, rank, {**cfg.to_dict(), "world_size": world})

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path or cfg.model_path)
    train_rows = load_rows(cfg.train_data, tokenizer, cfg.max_seq_len, seed=cfg.seed)
    loader_args = dict(batch_size=cfg.per_device_batch_size, rank=rank, world_size=world)
    train_loader = make_loader(train_rows, train=True, seed=cfg.seed, **loader_args)
    eval_rows = load_rows(cfg.eval_data, tokenizer, cfg.max_seq_len)
    eval_loader = make_loader(eval_rows, train=False, **loader_args)

    mesh = init_device_mesh(device.type, (world,))
    model = build_student(cfg, train_rows, device, mesh)
    teacher = build_teacher(cfg, device, mesh) if cfg.objective == "kd" else None
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = make_optimizer(params, cfg)
    log(
        {
            "step": 0,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "world_size": world,
            "global_batch": cfg.per_device_batch_size * cfg.grad_accum * world,
            "trainable_params": sum(p.numel() for p in params),
        }
    )

    if cfg.eval_steps:
        log({"step": 0, **evaluate(model, teacher, eval_loader, device)})
    batches, n_logged, t0 = forever(train_loader), 0, time.time()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    for step in range(1, cfg.max_steps + 1):
        for micro in range(cfg.grad_accum):
            batch = {k: v.to(device) for k, v in next(batches).items()}
            # Reduce-scatter gradients only once, on the last micro-batch.
            model.set_requires_gradient_sync(micro == cfg.grad_accum - 1)
            loss = objective(model, teacher, batch) / cfg.grad_accum
            loss.backward()
            loss_sum += loss.detach()
        grad_norm = nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()
        if cfg.method == "quasar":
            snapshot_saliency(model, optimizer)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        n_logged += 1
        if step % cfg.logging_steps == 0 or step == 1:
            log(
                {
                    "step": step,
                    "loss": _mean_over_ranks(loss_sum / n_logged),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm.full_tensor()),
                    "wall_s": round(time.time() - t0, 1),
                }
            )
            loss_sum.zero_()
            n_logged = 0
        if cfg.eval_steps and step % cfg.eval_steps == 0 and step < cfg.max_steps:
            log({"step": step, **evaluate(model, teacher, eval_loader, device)})

    final = evaluate(model, teacher, eval_loader, device)
    log({"step": cfg.max_steps, **final})
    # Collective: dequantized bf16 weights, QUASAR searched with its live saliency.
    save_materialized(
        model,
        out / "materialized",
        model_path=cfg.model_path,
        tokenizer_path=cfg.tokenizer_path,
        receipt={"train_config": cfg.to_dict(), "final_eval": final},
    )
    log.close()
    dist.destroy_process_group()
    return final
