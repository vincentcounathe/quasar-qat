"""Chat rendering and completion masks shared by training, evaluation and corpus building.

Every corpus row is ``{"messages": [{"role", "content"[, "reasoning_content"]}, ...]}``
(the Qwen3 thinking template renders the ``<think>`` block from ``reasoning_content``).
Rows are rendered with the model's own chat template and supervised only on assistant
turns, from after the role header through the end-of-turn token. Two template families
are supported: ChatML (Qwen) and Llama-3 headers.
"""

from __future__ import annotations

from typing import Any

import torch


def render_text(tokenizer: Any, messages: list[dict], *, add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _marker_ids(tokenizer: Any, tokens: tuple[str, ...]) -> list[int] | None:
    """Ids of chat markers, or None if the tokenizer lacks one (a missing token maps to ``unk_token_id``)."""
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    return None if tokenizer.unk_token_id in ids else ids


def _chatml_mask(tokenizer: Any, input_ids: list[int], im_start: int, im_end: int) -> list[int]:
    n = len(input_ids)
    mask = [0] * n
    for p in range(n):
        if input_ids[p] != im_start:
            continue
        # Content starts just past the header newline ("<|im_start|>{role}\n").
        cs = next((j + 1 for j in range(p + 1, n) if "\n" in tokenizer.decode([input_ids[j]])), None)
        if cs is None or tokenizer.decode(input_ids[p + 1 : cs]).strip() != "assistant":
            continue
        end = next((j for j in range(cs, n) if input_ids[j] == im_end), n - 1)
        for j in range(cs, min(end + 1, n)):
            mask[j] = 1
    return mask


def _llama3_mask(tokenizer: Any, input_ids: list[int], sh: int, eh: int, eot: int) -> list[int]:
    n = len(input_ids)
    mask = [0] * n
    for p in range(n):
        if input_ids[p] != sh:
            continue
        he = next((j for j in range(p + 1, n) if input_ids[j] == eh), None)
        if he is None or tokenizer.decode(input_ids[p + 1 : he]).strip() != "assistant":
            continue
        end = next((j for j in range(he + 1, n) if input_ids[j] == eot), n - 1)
        for j in range(he + 1, min(end + 1, n)):
            mask[j] = 1
    return mask


def completion_mask(tokenizer: Any, input_ids: list[int]) -> list[int]:
    """1 on every assistant turn from just after its role header through its end-of-turn token, else 0
    (Llama-3 also counts the ``"\\n\\n"`` token that closes the header).

    Scans the final token ids, so it is robust to templates whose partial renders are
    not token prefixes of the full render.
    """
    if ids := _marker_ids(tokenizer, ("<|im_start|>", "<|im_end|>")):
        return _chatml_mask(tokenizer, input_ids, *ids)
    if ids := _marker_ids(tokenizer, ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")):
        return _llama3_mask(tokenizer, input_ids, *ids)
    raise ValueError("unsupported chat template: the tokenizer has neither ChatML nor Llama-3 markers")


def tokenize_chat(tokenizer: Any, messages: list[dict], max_seq_len: int) -> tuple[list[int], list[int]] | None:
    """Render, right-truncate to ``max_seq_len`` and mask one conversation.

    Returns ``(input_ids, completion_mask)``, or None if the row has no supervised
    token inside the window. The template emits BOS itself, hence
    ``add_special_tokens=False``.
    """
    text = render_text(tokenizer, messages)
    input_ids = tokenizer(text, truncation=True, max_length=max_seq_len, add_special_tokens=False)["input_ids"]
    if len(input_ids) < 2:
        return None
    mask = completion_mask(tokenizer, input_ids)
    if sum(mask) == 0:
        return None
    return list(input_ids), mask


def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Right-pad a batch of :func:`quasar.train.data.load_rows` rows. Padding is masked out of
    attention and labels (-100, like unsupervised tokens), so its token id (0) never matters."""
    seqs = [r["input_ids"].long() for r in rows]
    input_ids = torch.zeros((len(seqs), max(s.numel() for s in seqs)), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for i, (seq, r) in enumerate(zip(seqs, rows)):
        n = seq.numel()
        input_ids[i, :n] = seq
        attention_mask[i, :n] = 1
        labels[i, :n][r["completion_mask"]] = seq[r["completion_mask"]]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
