"""Reading and writing HF-layout safetensors checkpoints, one tensor at a time."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX = "model.safetensors.index.json"


class ShardedSafetensorsWriter:
    """Incremental HF checkpoint writer that holds at most one shard in memory.

    Shards are written under temporary names and renamed to the HF layout
    (``model.safetensors``, or ``model-XXXXX-of-XXXXX.safetensors`` plus the
    index) only in :meth:`finalize`, so an interrupted save never looks loadable.
    """

    def __init__(self, out_dir: str | Path, max_shard_bytes: int = 4 << 30):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_bytes
        self.shards: list[list[str]] = []
        self.total_bytes = 0
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0

    def _tmp(self, i: int) -> Path:
        return self.out_dir / f"shard-{i:05d}.safetensors.tmp"

    def add(self, name: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().to("cpu").contiguous()
        nbytes = t.numel() * t.element_size()
        if self._buffer and self._buffer_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self._buffer[name] = t
        self._buffer_bytes += nbytes
        self.total_bytes += nbytes

    def _flush(self) -> None:
        save_file(self._buffer, str(self._tmp(len(self.shards))), metadata={"format": "pt"})
        self.shards.append(list(self._buffer))
        self._buffer, self._buffer_bytes = {}, 0

    def finalize(self) -> dict[str, str]:
        """Write the last shard, rename all shards into place; returns the weight map."""
        if self._buffer:
            self._flush()
        n = len(self.shards)
        if n == 0:
            raise RuntimeError("no tensors were written")
        weight_map = {}
        for i, names in enumerate(self.shards):
            fname = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            os.replace(self._tmp(i), self.out_dir / fname)
            weight_map.update(dict.fromkeys(names, fname))
        if n > 1:
            index = {"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map}
            (self.out_dir / INDEX).write_text(json.dumps(index, indent=2, sort_keys=True))
        return weight_map


class SafetensorsReader:
    """Random access by name to the tensors of a (possibly sharded) HF checkpoint."""

    def __init__(self, ckpt_dir: str | Path):
        d = Path(ckpt_dir)
        if (d / INDEX).is_file():
            weight_map = json.loads((d / INDEX).read_text())["weight_map"]
            self.files = {name: d / fname for name, fname in weight_map.items()}
        elif (d / "model.safetensors").is_file():
            with safe_open(str(d / "model.safetensors"), framework="pt") as f:
                self.files = dict.fromkeys(f.keys(), d / "model.safetensors")
        else:
            raise FileNotFoundError(f"no model.safetensors or {INDEX} in {d}")
        self._handles = {}

    def keys(self) -> list[str]:
        return list(self.files)

    def __getitem__(self, name: str) -> torch.Tensor:
        path = self.files[name]
        if path not in self._handles:
            self._handles[path] = safe_open(str(path), framework="pt")
        return self._handles[path].get_tensor(name)
