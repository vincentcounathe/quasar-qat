"""Build the OpenMathReasoning adaptation corpus.

Streams ``nvidia/OpenMathReasoning`` (split ``cot``, pinned revision), shuffled with
``--seed`` (shard order plus a 10k-row buffer), and keeps a row when

  * the problem has no think tags or ChatML markers and matches no problem of the
    adaptation table's benchmarks (MATH-500, GSM8K, AIME'24, AIME'25, HMMT'25): 13-gram
    containment of at least 0.3 against a single problem, which also catches exact copies;
  * the solution holds exactly one ``<think>`` ... ``</think>`` pair, in that order, no
    ChatML markers, and a non-empty answer after it containing ``\\boxed{``;
  * the problem has not been kept yet (one trace per problem);
  * its render fits ``--max_seq_len`` tokens. Long rows are dropped, never truncated,
    and do not use up their problem, so a shorter trace of it can still be kept.

A row is ``{"messages": [user, assistant]}``. The user turn is the problem followed by
``MATH_INSTRUCTION``, the prompt the math evaluation uses; the assistant content keeps
the reasoning inline and opens with exactly ``"<think>\\n"``, the prefix the evaluation
seeds. Render the corpus with a template that emits assistant content verbatim
(Qwen3-4B-Instruct-2507's), never with the Qwen3 thinking template, which strips the
reasoning from the content. Streaming stops after ``--rows`` rows; they are shuffled
with ``random.Random(--seed)``, the first ``--eval_rows`` become ``eval.jsonl`` (the
held-out perplexity set) and the rest ``train.jsonl``.

Paper corpus (32,754 train + 512 eval rows; Qwen3-4B-Base trained with the
Qwen3-4B-Instruct-2507 tokenizer and chat template):

  python -m quasar.data.build_math --tokenizer Qwen/Qwen3-4B-Instruct-2507 \\
      --max_seq_len 18432 --rows 33266 --eval_rows 512 --out_dir omr_corpus
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any

from quasar.data import MATH_INSTRUCTION, THINK_OPEN
from quasar.data.common import token_len, write_splits
from quasar.eval.common import hf_rows
from quasar.eval.mathbench import BENCHES
from quasar.eval.settings import TABLES

OMR_REPO = "nvidia/OpenMathReasoning"
OMR_REVISION = "d3d08664755704f422af97d43a7ff0ded4bd95df"
SHUFFLE_BUFFER = 10_000
CHATML_MARKERS = ("<|im_start|>", "<|im_end|>")

DECONTAMINATE = tuple(TABLES["adaptation"]["qwen"])  # every benchmark the adaptation table scores
NGRAM = 13
MIN_CONTAINMENT = 0.30


def _ngrams(text: str) -> set[str]:
    """Word n-grams of the lowercased alphanumeric text (the whole text if shorter)."""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    if len(words) < NGRAM:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


class Contamination:
    """Flags a text whose n-gram containment ``|shared| / min(|text grams|, |problem grams|)``
    against a single benchmark problem reaches ``MIN_CONTAINMENT``. Requiring containment,
    rather than any shared n-gram, avoids flagging the boilerplate that many competition
    problems share ("where m and n are relatively prime positive integers ...")."""

    def __init__(self, problems: list[str]):
        self.grams = [_ngrams(p) for p in problems]
        self.index: dict[str, list[int]] = {}
        for i, grams in enumerate(self.grams):
            for g in grams:
                self.index.setdefault(g, []).append(i)

    def __call__(self, text: str) -> bool:
        grams = _ngrams(text)
        shared = Counter(i for g in grams for i in self.index.get(g, ()))
        return any(c / min(len(grams), len(self.grams[i])) >= MIN_CONTAINMENT for i, c in shared.items())


def benchmark_problems() -> list[str]:
    return [p for b in DECONTAMINATE for p in hf_rows(b)[BENCHES[b][0]] if p]


def rejection(problem: str, solution: str) -> str | None:
    """Why a (problem, solution) pair is unusable, or None."""
    if any(tag in problem for tag in ("<think>", "</think>") + CHATML_MARKERS):
        return "problem_markers"
    if solution.count("<think>") != 1 or solution.count("</think>") != 1:
        return "think_tags"
    reasoning, _, answer = solution.partition("</think>")
    if "<think>" not in reasoning:
        return "think_tags"
    if not answer.strip():
        return "empty_answer"
    if "\\boxed{" not in answer:
        return "no_boxed"
    if any(m in solution for m in CHATML_MARKERS):
        return "solution_markers"
    return None


def to_messages(problem: str, solution: str) -> list[dict]:
    body = solution.split("<think>", 1)[1].lstrip()
    return [
        {"role": "user", "content": f"{problem.rstrip()}\n\n{MATH_INSTRUCTION}"},
        {"role": "assistant", "content": THINK_OPEN + body},
    ]


def build(stream, tokenizer: Any, contaminated, *, max_seq_len: int, rows: int) -> tuple[list[list[dict]], Counter]:
    """Kept conversations, in stream order, and the filter counts."""
    counts: Counter = Counter()
    kept, seen = [], set()
    for ex in stream:
        counts["read"] += 1
        problem, solution = ex.get("problem") or "", ex.get("generated_solution") or ""
        reason = rejection(problem, solution) if problem and solution else "missing"
        if reason is None and problem.strip() in seen:
            reason = "duplicate"
        if reason is None and contaminated(problem):
            reason = "contaminated"
        if reason is None:
            messages = to_messages(problem, solution)
            n_tokens = token_len(tokenizer, messages)
            reason = "too_long" if n_tokens > max_seq_len else None
        if reason:
            counts[reason] += 1
            continue
        seen.add(problem.strip())
        kept.append(messages)
        counts["kept"] += 1
        counts["kept_tokens"] += n_tokens
        if counts["kept"] % 2000 == 0:
            print(f"read {counts['read']}, kept {counts['kept']}", flush=True)
        if counts["kept"] == rows:
            break
    return kept, counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="tokenizer whose chat template the trainer uses")
    ap.add_argument("--max_seq_len", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True, help="train + eval rows")
    ap.add_argument("--eval_rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args(argv)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stream = load_dataset(OMR_REPO, split="cot", streaming=True, revision=OMR_REVISION)
    stream = stream.shuffle(seed=args.seed, buffer_size=SHUFFLE_BUFFER)
    kept, counts = build(
        stream, tokenizer, Contamination(benchmark_problems()), max_seq_len=args.max_seq_len, rows=args.rows
    )
    write_splits(kept, counts, args)


if __name__ == "__main__":
    main()
