"""Supervision masks on the paper's tokenizers: only assistant turns are trained on."""

import pytest
import torch

from quasar.data.chat import collate, completion_mask, render_text, tokenize_chat

QWEN_THINKING = "Qwen/Qwen3-4B-Thinking-2507"
QWEN_INSTRUCT = "Qwen/Qwen3-4B-Instruct-2507"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
END_OF_TURN = {QWEN_THINKING: "<|im_end|>", QWEN_INSTRUCT: "<|im_end|>", LLAMA: "<|eot_id|>"}

CONV = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "First question?"},
    {"role": "assistant", "content": "First answer."},
    {"role": "user", "content": "Second question?"},
    {"role": "assistant", "content": "Second answer \\boxed{4}.", "reasoning_content": "Think it through."},
]


def _supervised(tok, ids, mask):
    return tok.decode([t for t, m in zip(ids, mask, strict=True) if m])


def test_chatml_mask_offline(tokenizer):  # helpers.byte_tokenizer: a real ChatML template, no download
    ids, mask = tokenize_chat(tokenizer, CONV, 4096)
    assert _supervised(tokenizer, ids, mask) == (
        "First answer.<|im_end|><think>\nThink it through.\n</think>\n\nSecond answer \\boxed{4}.<|im_end|>"
    )


@pytest.mark.parametrize("name", [QWEN_THINKING, QWEN_INSTRUCT, LLAMA])
def test_only_assistant_turns_are_supervised(name, hf_tokenizer):
    tok = hf_tokenizer(name)
    ids, mask = tokenize_chat(tok, CONV, 4096)
    assert len(ids) == len(mask) and mask == completion_mask(tok, ids)
    sup = _supervised(tok, ids, mask)
    assert "Second answer" in sup and sup.endswith(END_OF_TURN[name])
    for prompt in ("You are terse.", "First question?", "Second question?", "assistant", "user"):
        assert prompt not in sup
    if name == QWEN_THINKING:  # the final turn's reasoning is rendered and trained on
        assert "Think it through." in sup and "<think>" in render_text(tok, CONV)
    if name != QWEN_THINKING:  # templates that keep every assistant turn supervise both
        assert "First answer." in sup


@pytest.mark.parametrize("name", [QWEN_INSTRUCT, LLAMA])
def test_truncation_and_rows_without_supervision(name, hf_tokenizer):
    tok = hf_tokenizer(name)
    full, _ = tokenize_chat(tok, CONV, 4096)
    ids, mask = tokenize_chat(tok, CONV, len(full) - 3)
    assert ids == full[: len(full) - 3] and sum(mask) > 0
    prompt_only = tok(render_text(tok, CONV[:2]), add_special_tokens=False)["input_ids"]
    assert tokenize_chat(tok, CONV, len(prompt_only)) is None  # window ends before any answer


def test_unknown_template_is_refused():
    class Plain:  # every chat marker maps to the unknown token
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 0

    with pytest.raises(ValueError, match="unsupported chat template"):
        completion_mask(Plain(), [5, 6, 7])


def test_collate_pads_and_masks_labels():
    rows = [
        {"input_ids": torch.tensor(ids, dtype=torch.int32), "completion_mask": torch.tensor(mask)}
        for ids, mask in [([1, 2, 3], [False, False, True]), ([5, 6], [False, True])]
    ]
    batch = collate(rows)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [5, 6, 0]] and batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert torch.equal(batch["labels"], torch.tensor([[-100, -100, 3], [-100, 6, -100]]))
