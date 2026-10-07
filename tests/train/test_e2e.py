"""End-to-end: 2-rank FSDP2 training on CPU (gloo) for every method, then the materialized checkpoint.

The materialized model, scored offline by ``quasar.eval.heldout``, must reproduce the
final in-loop held-out metrics: it holds exactly the weights the trained model computed
with, and on CPU the two forwards agree to rounding.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from quasar.eval import heldout
from quasar.export import nvfp4
from quasar.quant.config import SALIENCY_WARMUP
from quasar.train.data import load_rows

REPO = str(Path(__file__).resolve().parents[2])
ARMS = [  # method, format, bits, objective, freeze, extra flags
    ("quasar", "int", 2, "kd", True, []),
    ("standard", "int", 4, "kd", True, []),
    ("lsq", "int", 3, "kd", True, []),
    ("denoising", "int", 2, "ce", False, []),
    ("bitdistiller", "int", 2, "kd", True, []),
    ("none", "int", 4, "ce", False, []),
    ("quasar", "nvfp4", 4, "kd", True, ["--fp32_master", "true"]),
    ("standard", "nvfp4", 4, "kd", True, []),
]
WORLD, STEPS, SEQ = 2, SALIENCY_WARMUP + 2, 256  # QUASAR trains and materializes with live saliency


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _train(tiny_checkpoint, out, method, fmt, bits, objective, freeze, extra):
    model_dir, train, heldout = tiny_checkpoint
    args = dict(
        model_path=model_dir,
        train_data=train,
        eval_data=heldout,
        out_dir=out,
        method=method,
        format=fmt,
        bits=bits,
        objective=objective,
        freeze=str(freeze).lower(),
        max_steps=STEPS,
        eval_steps=STEPS // 2,
        logging_steps=1,
        per_device_batch_size=2,
        grad_accum=2,
        max_seq_len=SEQ,
        learning_rate=1e-3,
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        f"--nproc_per_node={WORLD}",
        "--master_addr",
        "127.0.0.1",
        f"--master_port={_free_port()}",
        "-m",
        "quasar.train",
    ]
    cmd += [x for k, v in args.items() for x in (f"--{k}", str(v))] + extra
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}
    env.update(PYTHONPATH=REPO, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


def _reference_eval(materialized, base, rows_path, kd):
    """Offline held-out metrics (quasar.eval.heldout) of the materialized checkpoint on the rows the run evaluated."""
    rows = load_rows(rows_path, AutoTokenizer.from_pretrained(materialized), SEQ)
    rows = rows[: len(rows) // WORLD * WORLD]  # the distributed eval drops the ragged tail
    teacher = heldout.load_model(base, "cpu") if kd else None
    return heldout.evaluate(heldout.load_model(materialized, "cpu"), rows, teacher)


@pytest.mark.slow
@pytest.mark.parametrize("method,fmt,bits,objective,freeze,extra", ARMS, ids=[f"{a[0]}-{a[1]}{a[2]}" for a in ARMS])
def test_two_rank_run_and_materialization(tmp_path, tiny_checkpoint, method, fmt, bits, objective, freeze, extra):
    out = tmp_path / "run"
    _train(tiny_checkpoint, out, method, fmt, bits, objective, freeze, extra)

    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if "loss" in r] == list(range(1, STEPS + 1))
    assert [r["step"] for r in log if "eval_ce" in r] == [0, STEPS // 2, STEPS]
    assert json.loads((out / "train_args.json").read_text())["method"] == method

    mat = out / "materialized"
    receipt = json.loads((mat / "receipt.json").read_text())
    final = receipt["final_eval"]
    assert receipt["quantized_linears"] == (0 if method == "none" else 14)
    assert (mat / "nvfp4_tensor_refs.json").exists() == (fmt == "nvfp4")
    if fmt == "nvfp4":  # one recorded tensor scale must describe every rank's rows
        assert nvfp4.export(mat, out / "ct")["quantized_linears"] == 14
    ref = _reference_eval(mat, tiny_checkpoint[0], tiny_checkpoint[2], objective == "kd")
    # Fused NVFP4 partitions share a tensor scale that lags one forward behind the weights
    # (quasar.quant.nvfp4.FusedRef): the first eval batch after an update sees the stale value,
    # while the artifact is written after a refresh.
    ce_rel, kl_rel = (1e-4, 1e-2) if fmt == "nvfp4" else (1e-5, 1e-4)
    assert ref["eval_ce"] == pytest.approx(final["eval_ce"], rel=ce_rel)
    if objective == "kd":
        assert ref["eval_kl"] == pytest.approx(final["eval_kl"], rel=kl_rel, abs=1e-7)
        assert ref["eval_top1"] == pytest.approx(final["eval_top1"], abs=0.01 if fmt == "nvfp4" else 0)
