"""Multiple-choice evals: MMLU-Pro (official 5-shot CoT prompt) and SuperGPQA (official
zero-shot prompt, 4000-item stratified subset), plus the one answer extractor that
every multiple-choice task in this package uses.
"""

from __future__ import annotations

import random
import re

from quasar.eval.common import hf_rows, strip_thinking
from quasar.eval.settings import MMLU_PRO_SHOTS, SEED, SUPERGPQA_ITEMS

LETTERS = "ABCDEFGHIJ"

# ----------------------------------------------------------------- extraction
# Explicit answer statements: "the answer is (X)", "answer should be X", "Answer: **X**",
# "correct option is X", "\boxed{X}", "\boxed{\text{(X)}}". Keywords are case-insensitive,
# the letter must be uppercase ("the answer is a bit tricky" names no option). Wrappers are
# matched by one character class so long whitespace runs cannot cause backtracking blow-ups.
_WRAP = r"[\s*$(\[{'\"`:]*(?:\\(?:boxed|text|textbf|mathbf|mathrm)\s*\{[\s*$(\[{'\"`]*)?"
_STATEMENT = re.compile(
    r"(?:answer\s+(?:is|should\s+be|would\s+be|must\s+be)|answer\s*:|(?:option|choice)\s+is"
    r"|\\boxed\s*\{)" + _WRAP + r"(?-i:([A-J]))(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Fallback near the end of the response: "(X)" or "option X".
_TAIL = re.compile(r"\(([A-J])\)|\boption\s*\(?(?-i:([A-J]))(?![A-Za-z0-9])", re.IGNORECASE)
_PROSE = re.compile(r"\s+[a-z]|'")
TAIL_CHARS = 300


def _is_prose(text: str, m: re.Match) -> bool:
    """An unwrapped "A"/"I" that starts a clause ("Answer: I think ...") is an
    article or pronoun, not a choice."""
    i = m.start(1)
    return m.group(1) in "AI" and text[i - 1] in " \t\n:" and _PROSE.match(text, m.end(1)) is not None


def extract_choice(text: str, n_options: int) -> str | None:
    """Option letter the response commits to, read after ``</think>``.

    The LAST explicit answer statement wins (models restate tentative answers before
    the final one). Without one, the last "(X)" / "option X" in the closing
    ``TAIL_CHARS`` characters; then an article-like "A"/"I" statement as a last resort.
    Letters beyond the item's options never count.
    """
    text = strip_thinking(text)
    valid = LETTERS[:n_options]
    hits = [m for m in _STATEMENT.finditer(text) if m.group(1) in valid]
    strong = [m for m in hits if not _is_prose(text, m)]
    if strong:
        return strong[-1].group(1)
    tail = text[-TAIL_CHARS:]
    for m in reversed(list(_TAIL.finditer(tail))):
        letter = m.group(1) or m.group(2)
        if letter in valid:
            return letter
    return hits[-1].group(1) if hits else None


def judge(item: dict, sample: str) -> tuple[str | None, bool]:
    letter = extract_choice(sample, item["n_options"])
    return letter, letter == item["gold"]


# ------------------------------------------------------------------- MMLU-Pro
MMLU_PRO_HEADER = (
    "The following are multiple choice questions (with answers) about {}. Think step by step and "
    'then finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n\n'
)


def format_cot_example(example: dict, with_answer: bool) -> str:
    """Official MMLU-Pro ``format_cot_example`` ("N/A" options already removed)."""
    prompt = f"Question:\n{example['question']}\nOptions:\n"
    prompt += "".join(f"{LETTERS[i]}. {opt}\n" for i, opt in enumerate(example["options"]))
    if with_answer:
        return (
            prompt
            + example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
            + "\n\n"
        )
    return prompt + "Answer: Let's think step by step."


def mmlu_pro_prompt(item: dict, shots: list[dict]) -> str:
    return (
        MMLU_PRO_HEADER.format(item["category"])
        + "".join(format_cot_example(s, True) for s in shots)
        + format_cot_example(item, False)
    )


def load_mmlu_pro() -> list[dict]:
    """Test items, each prompted with the first five validation items of its category."""

    def rows(split=None):
        return [dict(r, options=[o for o in r["options"] if o != "N/A"]) for r in hf_rows("mmlu_pro", split)]

    val = rows("validation")
    items = []
    for r in rows():
        shots = [v for v in val if v["category"] == r["category"]][:MMLU_PRO_SHOTS]
        items.append(
            {
                "id": str(r["question_id"]),
                "gold": r["answer"].strip().upper(),
                "n_options": len(r["options"]),
                "messages": [{"role": "user", "content": mmlu_pro_prompt(r, shots)}],
            }
        )
    return items


# ------------------------------------------------------------------ SuperGPQA
SUPERGPQA_TEMPLATE = (
    "Answer the following multiple choice question. There is only one correct answer. The last "
    "line of your response should be in the format 'Answer: $LETTER' (without quotes), where "
    "LETTER is one of A, B, C, D, E, F, G, H, I, or J.\n\n{}"
)


def supergpqa_prompt(question: str, options: list[str]) -> str:
    return SUPERGPQA_TEMPLATE.format(question + "\n" + "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options)))


def stratified_subsample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional subsample by discipline (largest remainder, ties by name), seeded
    sampling inside each discipline, dataset order kept."""
    strata: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        strata.setdefault(it["discipline"], []).append(i)
    names = sorted(strata)
    quota = {s: n * len(strata[s]) / len(items) for s in names}
    alloc = {s: int(quota[s]) for s in names}
    for s in sorted(names, key=lambda s: (int(quota[s]) - quota[s], s))[: n - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(seed)
    return [items[i] for i in sorted(i for s in names for i in rng.sample(strata[s], alloc[s]))]


def load_supergpqa() -> list[dict]:
    rows = [dict(r) for r in hf_rows("supergpqa")]
    return [
        {
            "id": r["uuid"],
            "gold": r["answer_letter"].strip().upper(),
            "n_options": len(r["options"]),
            "messages": [{"role": "user", "content": supergpqa_prompt(r["question"], r["options"])}],
        }
        for r in stratified_subsample(rows, SUPERGPQA_ITEMS, SEED)
    ]
