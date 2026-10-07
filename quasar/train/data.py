"""Chat corpora for training: JSONL rows rendered and masked by :mod:`quasar.data.chat`."""

from __future__ import annotations

import itertools
import random

import torch
from torch.utils.data import DataLoader, DistributedSampler

from quasar.data.chat import collate, tokenize_chat
from quasar.data.common import iter_jsonl


def load_rows(path: str, tokenizer, max_seq_len: int, seed: int | None = None) -> list[dict]:
    """Tokenized rows of a ``{"messages": [...]}`` JSONL file: ``input_ids`` (int32) and
    ``completion_mask`` (bool) tensors, compact enough to hold a whole corpus per rank.

    ``seed`` shuffles the rows (``None`` keeps file order). Rows with no supervised
    token inside ``max_seq_len`` are skipped: they would contribute nothing to the loss.
    """
    rows = list(iter_jsonl(path))
    if seed is not None:
        random.Random(seed).shuffle(rows)
    out = []
    for row in rows:
        built = tokenize_chat(tokenizer, row["messages"], max_seq_len)
        if built is not None:
            out.append(
                {
                    "input_ids": torch.tensor(built[0], dtype=torch.int32),
                    "completion_mask": torch.tensor(built[1], dtype=torch.bool),
                }
            )
    if not out:
        raise RuntimeError(f"{path}: no row has a supervised token within {max_seq_len} tokens")
    return out


def make_loader(
    rows: list[dict], batch_size: int, *, rank: int, world_size: int, train: bool, seed: int = 0
) -> DataLoader:
    """Per-rank loader. Every rank gets the same number of batches, as FSDP collectives require.

    Training shuffles per epoch (``loader.sampler.set_epoch``) and drops the ragged
    tail; evaluation keeps file order and drops at most ``world_size - 1`` rows.
    """
    sampler = DistributedSampler(rows, num_replicas=world_size, rank=rank, shuffle=train, seed=seed, drop_last=True)
    return DataLoader(rows, batch_size=batch_size, sampler=sampler, drop_last=train, collate_fn=collate)


def forever(loader: DataLoader):
    """Batches of ``loader`` across epochs, reshuffling each epoch."""
    for epoch in itertools.count():
        loader.sampler.set_epoch(epoch)
        yield from loader


def calibration_ids(rows: list[dict], n_seq: int = 128, seq_len: int = 1024) -> torch.Tensor:
    """BitDistiller calibration: the first training rows' tokens packed into ``[n, seq_len]`` blocks."""
    flat = list(itertools.islice(itertools.chain.from_iterable(r["input_ids"].tolist() for r in rows), n_seq * seq_len))
    n = len(flat) // seq_len
    if n == 0:
        raise RuntimeError(f"need at least {seq_len} training tokens for the BitDistiller calibration")
    return torch.tensor(flat[: n * seq_len], dtype=torch.long).view(n, seq_len)
