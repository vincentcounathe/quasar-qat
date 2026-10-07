"""Shared fixtures (no downloads), built from :mod:`helpers`."""

import pytest
from helpers import byte_tokenizer, chat_rows, save_base, tiny_model, write_jsonl


@pytest.fixture
def tokenizer():
    return byte_tokenizer()


@pytest.fixture(params=["llama", "qwen3"])
def family(request):
    return request.param


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """A saved tiny Qwen3 + tokenizer and train/eval corpora: ``(model_dir, train.jsonl, eval.jsonl)``."""
    root = tmp_path_factory.mktemp("tiny")
    model_dir = str(save_base(tiny_model(), root / "model"))
    train, heldout = write_jsonl(root / "train.jsonl", chat_rows(48)), write_jsonl(root / "eval.jsonl", chat_rows(9, 1))
    return model_dir, train, heldout
