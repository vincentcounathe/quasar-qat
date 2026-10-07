"""Held-out metrics of a checkpoint: CE / perplexity on assistant tokens and, with a
teacher, KL(teacher || student) and top-1 agreement.

Rows are loaded, masked and scored by the trainer's own code
(:func:`quasar.train.data.load_rows`, :func:`quasar.train.objectives.heldout_sums`),
so at the run's ``max_seq_len`` the numbers match its in-loop evaluation.

    python -m quasar.eval.heldout --model CKPT --data eval.jsonl --max_seq_len 18432 --out_dir OUT
    python -m quasar.eval.heldout --model CKPT --teacher Qwen/Qwen3-4B-Thinking-2507 \\
        --data eval.jsonl --max_seq_len 4096 --out_dir OUT          # writes OUT/heldout.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from quasar.data.chat import collate
from quasar.eval.common import write_json
from quasar.train.data import load_rows
from quasar.train.objectives import heldout_metrics, heldout_sums


def load_model(path: str, device: str, revision: str | None = None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, revision=revision)
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, rows: list[dict], teacher=None) -> dict[str, float]:
    """Token-weighted ``eval_ce`` / ``eval_ppl`` (+ ``eval_kl`` / ``eval_top1`` with a teacher)."""
    sums = torch.zeros(4, dtype=torch.float64)
    for row in rows:  # one row per batch: nothing is padded
        batch = collate([row])
        ids, labels = batch["input_ids"].to(model.device), batch["labels"].to(model.device)
        t_logits = teacher(input_ids=ids).logits if teacher is not None else None
        sums += heldout_sums(model(input_ids=ids).logits, labels, t_logits).cpu()
    return heldout_metrics(sums, teacher is not None)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Checkpoint to score (its tokenizer renders the rows).")
    ap.add_argument("--data", required=True, help='Held-out jsonl, rows {"messages": [...]}.')
    ap.add_argument("--max_seq_len", type=int, required=True)
    ap.add_argument("--teacher", help="BF16 reference for KL / top-1.")
    ap.add_argument("--teacher_revision")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    rows = load_rows(args.data, AutoTokenizer.from_pretrained(args.model), args.max_seq_len)
    teacher = load_model(args.teacher, args.device, args.teacher_revision) if args.teacher else None
    metrics = evaluate(load_model(args.model, args.device), rows, teacher)
    write_json(
        Path(args.out_dir) / "heldout.json", {"task": "heldout", **metrics, "n_rows": len(rows), "settings": vars(args)}
    )
    print("[heldout] " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()), flush=True)


if __name__ == "__main__":
    main()
