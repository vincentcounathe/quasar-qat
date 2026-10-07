"""Checkpoint writer/reader and verbatim metadata copies."""

import json

import pytest
import torch

from quasar.export.files import copy_model_files
from quasar.export.writer import INDEX, SafetensorsReader, ShardedSafetensorsWriter


def _tensors():
    return {
        "a": torch.randn(64, 64).to(torch.bfloat16),
        "b": torch.arange(10, dtype=torch.int64),
        "c": torch.randn(32, 8).to(torch.float8_e4m3fn),
        "d": torch.randint(0, 255, (16, 16), dtype=torch.uint8),
    }


def _write(path, tensors, max_shard_bytes):
    w = ShardedSafetensorsWriter(path, max_shard_bytes=max_shard_bytes)
    for k, t in tensors.items():
        w.add(k, t)
    return w, w.finalize()


def _assert_round_trip(path, tensors):
    r = SafetensorsReader(path)
    assert sorted(r.keys()) == sorted(tensors)
    for k, t in tensors.items():
        assert r[k].dtype == t.dtype and torch.equal(r[k].view(torch.uint8), t.view(torch.uint8))
    assert not list(path.glob("*.tmp"))


def test_single_shard(tmp_path):
    tensors = _tensors()
    _, weight_map = _write(tmp_path, tensors, 1 << 30)
    assert set(weight_map.values()) == {"model.safetensors"} and not (tmp_path / INDEX).exists()
    _assert_round_trip(tmp_path, tensors)


def test_sharded_with_index(tmp_path):
    tensors = _tensors()
    w, weight_map = _write(tmp_path, tensors, 4096)
    index = json.loads((tmp_path / INDEX).read_text())
    n = len(w.shards)
    assert n > 1 and index["weight_map"] == weight_map
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in tensors.values())
    assert sorted(set(weight_map.values())) == [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    _assert_round_trip(tmp_path, tensors)


def test_nothing_written_is_an_error(tmp_path):
    with pytest.raises(RuntimeError):
        ShardedSafetensorsWriter(tmp_path).finalize()


def test_copy_model_files_is_verbatim(tmp_path):
    model, tok, out = tmp_path / "model", tmp_path / "tok", tmp_path / "out"
    for d in (model, tok, out):
        d.mkdir()
    (model / "config.json").write_text('{"rope_theta":  1000000.0,\n "x": [1,2]}')  # odd formatting on purpose
    (model / "generation_config.json").write_text('{"eos_token_id": [1, 2]}')
    (model / "tokenizer.json").write_text("model tokenizer")
    (tok / "tokenizer.json").write_text("instruct tokenizer")
    (tok / "chat_template.jinja").write_text("{{ x }}")
    copied = copy_model_files(out, model, tok)
    assert sorted(copied) == ["chat_template.jinja", "config.json", "generation_config.json", "tokenizer.json"]
    for name, src in [("config.json", model), ("generation_config.json", model), ("tokenizer.json", tok)]:
        assert (out / name).read_bytes() == (src / name).read_bytes()
    with pytest.raises(FileNotFoundError):
        copy_model_files(out, tok)
