"""JSONL and prompt helpers shared by the corpus tools."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from quasar.data.chat import render_text


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    """Write rows atomically (temp file + rename), so a partial file never looks complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def prompt_text(row: dict) -> str:
    """The prompt of a row: ``prompt`` (prompt slices) or the first user turn of ``messages``."""
    if "prompt" in row:
        return row["prompt"]
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def load_prompts(paths: Iterable[str]) -> set[str]:
    """Stripped prompts of every row in ``paths``: the exclusion set of held-out prompts."""
    return {prompt_text(row).strip() for path in paths for row in iter_jsonl(path)}


def token_len(tokenizer: Any, messages: list[dict], *, add_generation_prompt: bool = False) -> int:
    """Token count of the trainer's render of ``messages``, without truncation."""
    text = render_text(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def write_splits(kept: list[list[dict]], counts: Counter, args: argparse.Namespace) -> None:
    """Shuffle the kept conversations with ``random.Random(args.seed)``; the first
    ``args.eval_rows`` go to ``eval.jsonl``, the rest to ``train.jsonl`` and the filter
    counts to ``stats.json``, all under ``args.out_dir``."""
    if len(kept) <= args.eval_rows:
        raise SystemExit(f"only {len(kept)} rows kept; need more than --eval_rows {args.eval_rows}")
    random.Random(args.seed).shuffle(kept)
    out = Path(args.out_dir)
    if args.eval_rows:
        write_jsonl(out / "eval.jsonl", ({"messages": m} for m in kept[: args.eval_rows]))
    write_jsonl(out / "train.jsonl", ({"messages": m} for m in kept[args.eval_rows :]))
    stats = {"counts": dict(counts), "train": len(kept) - args.eval_rows, "eval": args.eval_rows, "args": vars(args)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats["counts"], indent=1))
