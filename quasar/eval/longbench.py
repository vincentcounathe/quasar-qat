"""LongBench-v2: 503 four-choice questions over contexts of up to ~2M words (official 0-shot prompt).

Documents longer than the prompt budget (the context window minus the generation
budget and the chat wrapper) are cut in the middle at token level like the official
``pred.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from quasar.eval import common as C

PROMPT_MARGIN = 64  # re-encoding drift between the cut text and its rendered form

# THUDM/LongBench prompts/0shot.txt.
TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def build_prompt(row: dict) -> str:
    fields = {
        "$DOC$": "context",
        "$Q$": "question",
        "$C_A$": "choice_A",
        "$C_B$": "choice_B",
        "$C_C$": "choice_C",
        "$C_D$": "choice_D",
    }
    text = TEMPLATE
    for key, field in fields.items():
        text = text.replace(key, str(row[field]).strip())
    return text


def truncate_middle(ids: Sequence[int], max_len: int) -> list[int]:
    """Official ``ids[:max_len//2] + ids[-max_len//2:]`` (an odd budget keeps the longer tail)."""
    ids = list(ids)
    return ids if len(ids) <= max_len else ids[: max_len // 2] + ids[-max_len // 2 :]


def load(tokenizer, thinking: bool, max_prompt_tokens: int) -> list[dict]:
    """Items whose rendered prompt fits ``max_prompt_tokens`` tokens (scored with ``mcq.judge``)."""
    wrapper = len(C.encode(tokenizer, [C.render_prompt(tokenizer, [{"role": "user", "content": ""}], thinking)])[0])
    budget = max_prompt_tokens - wrapper - PROMPT_MARGIN
    items = []
    for row in C.hf_rows("longbench_v2"):
        text = build_prompt(row)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > budget:
            text = tokenizer.decode(truncate_middle(ids, budget), skip_special_tokens=True)
        items.append(
            {
                "id": row["_id"],
                "gold": row["answer"].strip().upper(),
                "n_options": 4,
                "messages": [{"role": "user", "content": text}],
            }
        )
    return items
