"""Seeded Open-PerfectBlend prompt slices for teacher sampling.

The prompt of an Open-PerfectBlend (OPB) row is its first non-blank human/user turn;
``orig_row`` is the row's index in the dataset at the pinned revision. A slice is a
uniform sample without replacement: the eligible rows are shuffled with
``random.Random(--seed)``, the first ``--n`` are taken and written in dataset order as
``{"prompt", "source", "orig_row"}``. With one seed, a smaller slice is a subset of a
larger one. ``--exclude`` drops every row whose prompt equals (after stripping) a
prompt of the given JSONL files: prompt slices, teacher samples or chat corpora.

With ``--pairs`` a row is instead the original first user -> assistant exchange
(prompt of 16-6000 characters, reply of at least 64), written as a chat row
``{"messages", "source", "orig_row"}``: held-out real conversations for teacher KL.

Paper corpora. Slices of
200,000 prompts (Qwen3-4B-Thinking-2507 healing), 100,000 (Qwen3-8B NVFP4
self-distillation) and 560,000 (Llama-3.1-8B-Instruct healing) leave at least the
paper's training rows after ``build_distill``:

  python -m quasar.data.prompts --n 200000 --out opb_200k.jsonl

The teacher-KL evaluation set of the NVFP4 table, disjoint from that corpus's prompts:

  python -m quasar.data.prompts --pairs --n 512 --exclude opb_100k.jsonl --out eval_opb_512.jsonl
"""

from __future__ import annotations

import argparse
import random

from quasar.data.common import load_prompts, prompt_text, write_jsonl

OPB_REPO = "mlabonne/open-perfectblend"
OPB_REVISION = "af60f3c18201652a83a93f46fcfee1b646ba3df7"
USER_TAGS = ("human", "user")
ASSISTANT_TAGS = ("gpt", "assistant", "chatgpt", "bing")
PAIR_PROMPT_CHARS = (16, 6000)
PAIR_MIN_REPLY_CHARS = 64


def first_exchange(conversations: list[dict] | None) -> tuple[str, str | None] | None:
    """(first non-blank user turn, the assistant reply that follows it or None), or None."""
    prompt = None
    for turn in conversations or []:
        text = turn.get("value")
        if prompt is None:
            if turn.get("from") in USER_TAGS and isinstance(text, str) and text.strip():
                prompt = text
        elif turn.get("from") in ASSISTANT_TAGS:
            return prompt, text
    return None if prompt is None else (prompt, None)


def to_record(orig_row: int, row: dict, pairs: bool = False) -> dict | None:
    """The slice record of an OPB row, or None if the row is not eligible."""
    exchange = first_exchange(row["conversations"])
    if exchange is None:
        return None
    prompt, reply = exchange
    if not pairs:
        return {"prompt": prompt, "source": row["source"], "orig_row": orig_row}
    lo, hi = PAIR_PROMPT_CHARS
    if not (lo <= len(prompt) <= hi and reply and reply.strip() and len(reply) >= PAIR_MIN_REPLY_CHARS):
        return None
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    return {"messages": messages, "source": row["source"], "orig_row": orig_row}


def sample(n: int, seed: int = 1234, *, exclude: set[str] = frozenset(), pairs: bool = False) -> list[dict]:
    """Seeded sample of ``n`` eligible, non-excluded OPB rows, in dataset order."""
    from datasets import load_dataset

    ds = load_dataset(OPB_REPO, split="train", revision=OPB_REVISION)

    def eligible(i: int, row: dict) -> bool:
        record = to_record(i, row, pairs)
        return record is not None and prompt_text(record).strip() not in exclude

    pool = [i for i, row in enumerate(ds) if eligible(i, row)]
    if len(pool) < n:
        raise SystemExit(f"only {len(pool)} eligible rows for --n {n}")
    random.Random(seed).shuffle(pool)
    picked = sorted(pool[:n])
    return [to_record(i, row, pairs) for i, row in zip(picked, ds.select(picked), strict=True)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--exclude", nargs="+", default=[], help="JSONL files whose prompts are never sampled")
    ap.add_argument("--pairs", action="store_true", help="original user/assistant pairs instead of prompts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    exclude = load_prompts(args.exclude)
    rows = sample(args.n, args.seed, exclude=exclude, pairs=args.pairs)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows -> {args.out} ({len(exclude)} prompts excluded)")


if __name__ == "__main__":
    main()
