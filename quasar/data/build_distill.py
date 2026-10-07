"""Assemble a distillation corpus from ``sample_teacher`` records.

Records are read from ``--src`` in the order given. A response whose reasoning came
back inline in ``content`` (``"[<think>]...</think>answer"``: the server ran no
reasoning parser, or one that does not split this model's output) is first split into
``reasoning_content`` and ``content``, the form the Qwen3 thinking template renders.
A record is kept when

  * ``finish_reason`` is ``stop`` and the final answer is non-empty;
  * its prompt (stripped text) is not in an ``--exclude`` file and has not been kept
    yet: one response per prompt, the first that passes every filter;
  * with ``--max_seq_len``, its full render fits that many tokens (long rows are
    dropped, never truncated);
  * with ``--max_prompt_len``, its rendered prompt, generation prompt included, fits
    that many tokens. Use it when the trainer right-truncates long rows instead, so
    every row keeps a substantial supervised completion.

Kept rows are shuffled with ``random.Random(--seed)``; the first ``--eval_rows`` are
written to ``eval.jsonl`` and the rest to ``train.jsonl`` (``{"messages"}`` only), with
the filter counts in ``stats.json``. Lines that are not valid JSON (a sampling run
killed mid-write) are counted and skipped.

Paper corpora. The trainer shuffles all train.jsonl rows (seeded) and draws
max_steps x global batch of them, about one pass for these corpora:

  Qwen3-4B-Thinking-2507 healing (paper: 136,111 train rows, 512 eval):
    python -m quasar.data.build_distill --src samples.jsonl --tokenizer Qwen/Qwen3-4B-Thinking-2507 \\
        --max_seq_len 4096 --eval_rows 512 --out_dir qwen3_4b_thinking_corpus
  Llama-3.1-8B-Instruct healing (paper: 524,288 train rows, 1,019 eval):
    python -m quasar.data.build_distill --src samples.jsonl --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
        --max_seq_len 4096 --eval_rows 1024 --out_dir llama31_8b_corpus
  Qwen3-8B NVFP4 self-distillation (paper: 99,872 train rows, 128 eval; the trainer
  right-truncates rows to 8192 tokens):
    python -m quasar.data.build_distill --src samples.jsonl --tokenizer Qwen/Qwen3-8B \\
        --max_prompt_len 8064 --eval_rows 128 --out_dir qwen3_8b_corpus
"""

from __future__ import annotations

import argparse
import fileinput
import json
from collections import Counter
from typing import Any

from quasar.data.common import load_prompts, prompt_text, token_len, write_splits


def split_think(message: dict) -> dict:
    """Move inline reasoning (``"[<think>]reasoning</think>answer"``) into ``reasoning_content``."""
    content = message["content"]
    if message["reasoning_content"] or "</think>" not in content:
        return message
    think, _, answer = content.partition("</think>")
    think = think.strip("\n").removeprefix("<think>").strip("\n")
    return {"role": message["role"], "content": answer.lstrip("\n"), "reasoning_content": think}


def filter_records(
    lines,
    tokenizer: Any,
    *,
    exclude: set[str] = frozenset(),
    max_seq_len: int | None = None,
    max_prompt_len: int | None = None,
) -> tuple[list[list[dict]], Counter]:
    """Kept conversations (``messages`` lists), in input order, and the filter counts."""
    counts: Counter = Counter()
    seen = set(exclude)
    kept = []
    for line in lines:
        if not line.strip():
            continue
        counts["read"] += 1
        try:
            record = json.loads(line)
        except ValueError:
            counts["bad_json"] += 1
            continue
        messages = record["messages"][:-1] + [split_think(record["messages"][-1])]
        prompt = prompt_text(record).strip()
        reason = None
        if prompt in seen:
            reason = "excluded" if prompt in exclude else "duplicate"
        elif record["finish_reason"] != "stop":
            reason = "not_stop"
        elif not messages[-1]["content"].strip():
            reason = "empty_answer"
        elif max_seq_len is not None and (n_tokens := token_len(tokenizer, messages)) > max_seq_len:
            reason = "too_long"
        elif max_prompt_len is not None and (
            token_len(tokenizer, messages[:-1], add_generation_prompt=True) > max_prompt_len
        ):
            reason = "prompt_too_long"
        if reason:
            counts[reason] += 1
            continue
        seen.add(prompt)
        kept.append(messages)
        counts["kept"] += 1
        if max_seq_len is not None:
            counts["kept_tokens"] += n_tokens
    return kept, counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, help="sample_teacher JSONL files, read in this order")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer", required=True, help="the student's tokenizer (chat template)")
    ap.add_argument("--max_seq_len", type=int, default=None, help="drop rows whose render is longer")
    ap.add_argument("--max_prompt_len", type=int, default=None, help="drop rows whose prompt render is longer")
    ap.add_argument("--exclude", nargs="+", default=[], help="JSONL files whose prompts are never kept")
    ap.add_argument("--eval_rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    exclude = load_prompts(args.exclude)
    with fileinput.input(args.src, encoding="utf-8") as lines:
        kept, counts = filter_records(
            lines, tokenizer, exclude=exclude, max_seq_len=args.max_seq_len, max_prompt_len=args.max_prompt_len
        )
    write_splits(kept, counts, args)


if __name__ == "__main__":
    main()
