"""Math benchmarks: MATH-500, GSM8K, AIME'24, AIME'25, HMMT'25, HMMT'26.

Every problem gets the boxed-answer instruction the adaptation corpus was built with.
One scorer serves all benches: the answer is read after ``</think>`` and Math-Verify
checks it against the gold (the whole answer span, then its last ``\\boxed{}``). Golds
Math-Verify cannot parse fall back to a normalized exact match with that box.
"""

from __future__ import annotations

import re

from quasar.data import MATH_INSTRUCTION
from quasar.eval.common import hf_rows, strip_thinking


def _gsm8k_gold(row: dict) -> str:
    return str(row["answer"]).split("####")[-1].strip().replace(",", "")


# bench -> (question field, gold accessor); field names follow each dataset card.
BENCHES = {
    "math500": ("problem", lambda r: str(r["answer"])),
    "gsm8k": ("question", _gsm8k_gold),
    "aime24": ("Problem", lambda r: str(r["Answer"])),
    "aime25": ("question", lambda r: str(r["answer"])),
    "hmmt25": ("problem", lambda r: str(r["answer"])),
    "hmmt26": ("problem", lambda r: str(r["answer"])),
}


def load(task: str) -> list[dict]:
    field, gold = BENCHES[task]
    return [
        {"id": i, "gold": gold(r), "messages": [{"role": "user", "content": f"{r[field]}\n\n{MATH_INSTRUCTION}"}]}
        for i, r in enumerate(hf_rows(task))
    ]


# ------------------------------------------------------------------- extraction
def last_boxed(text: str) -> str | None:
    """Content of the LAST top-level, brace-matched ``\\boxed{...}``, or None."""
    last, pos = None, text.find("\\boxed")
    while pos != -1:
        brace = text.find("{", pos)
        if brace == -1:
            break
        depth = 0
        for j in range(brace, len(text)):
            depth += (text[j] == "{") - (text[j] == "}")
            if depth == 0:
                last = text[brace + 1 : j]
                break
        else:  # unclosed box: nothing after it can close either
            break
        pos = text.find("\\boxed", j + 1)
    return last


def extract_answer(sample: str) -> str | None:
    """The last ``\\boxed{}`` of the answer after thinking."""
    boxed = last_boxed(strip_thinking(sample))
    return None if boxed is None else boxed.strip()


def normalize(ans: str) -> str:
    s = ans.strip().strip("$").strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac")):
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s).rstrip(".")
    if s.startswith("{") and last_boxed("\\boxed" + s) == s[1:-1]:
        s = s[1:-1]
    return s.replace(",", "")


# --------------------------------------------------------------------- scoring
def is_correct(gold: str, sample: str) -> bool:
    """Math-Verify equivalence of the gold and the answer span (or its last box)."""
    from math_verify import parse, verify  # `eval` extra; parse/verify never raise (they return [] / False)

    gold_parsed = parse(f"${gold}$") or parse(gold)
    pred = extract_answer(sample)
    if not gold_parsed:
        return pred is not None and normalize(pred) == normalize(gold)
    return any(verify(gold_parsed, parse(c)) for c in (strip_thinking(sample), pred or ""))


def judge(item: dict, sample: str) -> tuple[str | None, bool]:
    return extract_answer(sample), is_correct(item["gold"], sample)
