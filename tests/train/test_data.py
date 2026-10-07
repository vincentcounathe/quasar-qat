"""Corpus loading, per-rank loaders and the BitDistiller calibration sample."""

import pytest
import torch
from helpers import chat_rows, write_jsonl

from quasar.data.chat import tokenize_chat
from quasar.train.data import calibration_ids, forever, load_rows, make_loader


def test_load_rows_and_shuffle(tmp_path, tokenizer):
    rows = chat_rows(20)
    path = write_jsonl(tmp_path / "c.jsonl", rows)
    ids, mask = tokenize_chat(tokenizer, rows[0]["messages"], 512)
    first = load_rows(path, tokenizer, 512)[0]
    assert first["input_ids"].dtype == torch.int32 and first["input_ids"].tolist() == ids
    assert first["completion_mask"].dtype == torch.bool and first["completion_mask"].tolist() == [bool(m) for m in mask]

    def order(rows):
        return [r["input_ids"].tolist() for r in rows]

    a, b = order(load_rows(path, tokenizer, 512, seed=3)), order(load_rows(path, tokenizer, 512, seed=3))
    assert a == b and a != order(load_rows(path, tokenizer, 512)) and len(a) == 20


def test_rows_without_supervision_are_skipped(tmp_path, tokenizer):
    rows = chat_rows(3)
    rows[1]["messages"][0]["content"] = "long prompt " * 100  # assistant turn falls outside the window
    path = write_jsonl(tmp_path / "c.jsonl", rows)
    assert len(load_rows(path, tokenizer, 256)) == 2
    write_jsonl(tmp_path / "bad.jsonl", rows[1:2])
    with pytest.raises(RuntimeError, match="no row"):
        load_rows(str(tmp_path / "bad.jsonl"), tokenizer, 256)


def test_loaders_give_every_rank_the_same_batch_count(tmp_path, tokenizer):
    rows = load_rows(write_jsonl(tmp_path / "c.jsonl", chat_rows(11)), tokenizer, 512)
    for train in (True, False):
        loaders = [make_loader(rows, 2, rank=r, world_size=3, train=train) for r in range(3)]
        counts = [len(list(dl)) for dl in loaders]
        assert len(set(counts)) == 1 and counts[0] == (1 if train else 2)
    batch = next(iter(loaders[0]))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert (batch["labels"] != -100).any() and ((batch["labels"] == -100) | (batch["attention_mask"] == 1)).all()


def test_forever_reshuffles_each_epoch(tmp_path, tokenizer):
    rows = load_rows(write_jsonl(tmp_path / "c.jsonl", chat_rows(16)), tokenizer, 512)
    loader = make_loader(rows, 4, rank=0, world_size=1, train=True, seed=0)
    batches = forever(loader)
    first = [next(batches)["input_ids"][0, :40] for _ in range(4)]
    second = [next(batches)["input_ids"][0, :40] for _ in range(4)]
    assert any(not torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_calibration_packs_training_tokens():
    rows = [{"input_ids": torch.arange(i * 10, i * 10 + 10, dtype=torch.int32)} for i in range(5)]
    ids = calibration_ids(rows, n_seq=8, seq_len=12)
    assert ids.shape == (4, 12) and ids.flatten().tolist() == list(range(48))
    assert calibration_ids(rows, n_seq=2, seq_len=12).shape == (2, 12)
    with pytest.raises(RuntimeError):
        calibration_ids(rows, seq_len=100)
