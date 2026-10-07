"""In-run materialization: the trained model as a plain bf16 HF checkpoint.

Each quantized decoder projection is written as the fake-quantized weight it
computes with, so the checkpoint loads in transformers or vLLM (``dtype=bfloat16``,
no quantization config) and reproduces the trained quantized model.

QUASAR can only be materialized from the live model: its saliency (the AdamW
second moment) lives on the quantizer, per rank, and no saved state dict carries
it; without it the search would silently fall back to uniform weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from quasar.quant import QuantLinear, quant_linears

from .files import copy_model_files
from .writer import ShardedSafetensorsWriter

DTYPE = torch.bfloat16  # artifact dtype and the compute dtype training quantized in
REFS_FILE = "nvfp4_tensor_refs.json"


def _full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t


@torch.no_grad()
def deployed_weight(m: QuantLinear, gather: bool = True) -> torch.Tensor:
    """The bf16 weight ``m`` computes with (a collective under FSDP2).

    Shard-local QUASAR repeats the training search on this rank's rows with this
    rank's saliency and then gathers the rows; groups never straddle the row
    sharding, so this equals the search on the full weight. Other methods call
    the quantizer on the gathered weight; under FSDP2 its forward hook gathers
    LSQ's step and offset, as in training.
    """
    qz, w = m.quantizer, m.weight
    if qz.bypass:  # shard-local QUASAR: an FSDP2 DTensor over a ShardLocalWeight
        r = qz(w.to_local()._tensor.to(DTYPE))
        if gather:
            r = DTensor.from_local(r, w.device_mesh, w.placements, shape=w.shape, stride=w.stride()).full_tensor()
        return r
    return qz(_full(w).to(DTYPE))


@torch.no_grad()
def save_materialized(
    model: nn.Module,
    out_dir: str | Path,
    *,
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    receipt: dict | None = None,
) -> dict:
    """Write ``model`` to ``out_dir`` as a bf16 HF checkpoint of its deployed weights.

    Under ``torch.distributed``, call on every rank between optimizer steps, after
    at least one forward through the model (FSDP2 initializes from the root):
    tensors are gathered, quantized and written by rank 0 one at a time, so no host
    ever holds the whole model. A tensor tied to an earlier one (``lm_head`` under
    tied embeddings) is written once under its first name. Config, generation
    config and tokenizer files are copied verbatim from ``model_path`` /
    ``tokenizer_path``; ``receipt`` goes to ``receipt.json``. NVFP4 runs also
    record the per-tensor references in ``nvfp4_tensor_refs.json``, which
    :mod:`quasar.export.nvfp4` needs to pack the weights.
    """
    out = Path(out_dir)
    distributed = dist.is_initialized()
    if distributed:
        dist.barrier()
    was_training = model.training
    model.eval()  # search every weight afresh instead of reusing the training step's search results
    quant = {f"{name}.weight": m for name, m in quant_linears(model)}
    # A fused NVFP4 partition takes the max of its partners' latest references;
    # one pass first makes all of them references of the final weights.
    for m in quant.values():
        if m.quantizer.fused_ref is not None:
            deployed_weight(m, gather=False)

    tied = getattr(model.config, "tie_word_embeddings", False)
    writer = ShardedSafetensorsWriter(out) if not distributed or dist.get_rank() == 0 else None
    refs, seen = {}, set()
    for key, t in model.state_dict(keep_vars=True).items():
        if ".quantizer." in key or id(t) in seen:
            continue
        if key == "lm_head.weight" and tied:
            raise RuntimeError("tie_word_embeddings is set but lm_head is not tied to the embeddings")
        seen.add(id(t))
        if key in quant:
            t = deployed_weight(quant[key])
            if quant[key].quantizer.config.format == "nvfp4":
                refs[key] = float(quant[key].quantizer.last_tensor_ref).hex()
        else:
            t = _full(t).detach()
            t = t.to(DTYPE) if t.is_floating_point() else t
        if writer is not None:
            writer.add(key, t)
        del t

    summary = {**(receipt or {}), "quantized_linears": len(quant)}
    if writer is not None:
        writer.finalize()
        if refs:
            (out / REFS_FILE).write_text(json.dumps(refs, indent=2, sort_keys=True))
        copy_model_files(out, model_path, tokenizer_path)
        (out / "receipt.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    model.train(was_training)
    if distributed:
        dist.barrier()
    return summary
