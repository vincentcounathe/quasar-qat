"""RTN: an untrained Standard QAT materialization of a base checkpoint."""

import torch
from helpers import save_base, tiny_model

from quasar.export import nvfp4, rtn
from quasar.export.writer import SafetensorsReader
from quasar.quant import DECODER_PROJ_RE
from quasar.quant.baselines import standard
from quasar.quant.common import to_blocks


def test_int_rtn(family, tmp_path):
    base = save_base(tiny_model(family), tmp_path / "base")
    out = tmp_path / "rtn"
    rtn.main(["--model", str(base), "--out_dir", str(out), "--bits", "3"])
    before, after = SafetensorsReader(base), SafetensorsReader(out)
    assert sorted(before.keys()) == sorted(after.keys())
    n = 0
    for k in before.keys():
        w = before[k]
        if DECODER_PROJ_RE.fullmatch(k.removesuffix(".weight")):
            expected = standard(to_blocks(w, 128), qmax=7).reshape(w.shape)
            n += 1
        else:
            expected = w
        assert torch.equal(after[k], expected), k
    assert n == 14


def test_nvfp4_rtn_packs(tmp_path):
    base = save_base(tiny_model("qwen3"), tmp_path / "base")
    out = tmp_path / "rtn"
    assert rtn.rtn(str(base), str(out), fmt="nvfp4")["quantized_linears"] == 14
    assert nvfp4.export(out, tmp_path / "ct")["quantized_linears"] == 14
