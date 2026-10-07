"""NVFP4 packing: lossless pack/unpack, the compressed-tensors layout and export."""

import json

import pytest
import torch
from helpers import save_base, tiny_model

from quasar.export import nvfp4, save_materialized
from quasar.export.writer import SafetensorsReader
from quasar.quant import FakeQuantizer, QuantConfig, quant_linears, quantize_model
from quasar.quant.config import SALIENCY_WARMUP
from quasar.quant.nvfp4 import E4M3_MAX, ct_divisor_snap


def _lattice_weight(method, rows=64, cols=256, seed=0):
    """A bf16 weight on the NVFP4 lattice, with its tensor scale ``S``."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 0.02
    w.view(-1)[torch.randint(0, w.numel(), (w.numel() // 100,), generator=g)] *= 20
    w[3] = 0  # all-zero groups
    w[5, :16] = -w[5, :16].abs()  # a one-signed group
    qz = FakeQuantizer(QuantConfig(method, "nvfp4"))
    for _ in range(SALIENCY_WARMUP if method == "quasar" else 0):
        qz.update_saliency(torch.exp(torch.randn(w.shape, generator=g) * 3 - 20))
    with torch.no_grad():
        r = qz(w.to(torch.bfloat16))
    return r, torch.tensor(float(qz.last_tensor_ref) / E4M3_MAX, dtype=torch.float32)


@pytest.mark.parametrize("method", ["standard", "quasar"])
def test_pack_round_trip_is_bitwise(method):
    r, S = _lattice_weight(method)
    for w in (r, -r):  # negated weights are on the lattice too (sign-folded scales)
        packed, scale = nvfp4.pack(w, S, rows=16)
        assert packed.shape == (64, 128) and packed.dtype == torch.uint8
        assert scale.shape == (64, 16) and scale.dtype == torch.float8_e4m3fn
        assert torch.equal(nvfp4.dequantize(packed, scale, nvfp4.global_scale(S)), w)


def test_packed_layout():
    S = torch.tensor(1.0)
    w = torch.zeros(1, 16)
    w[0, :4] = torch.tensor([0.5, -6.0, 0.0, 3.0])
    packed, scale = nvfp4.pack(w.to(torch.bfloat16), S)
    # Codes: 0.5 -> 1, -6 -> 8|7, 0 -> 0, 3 -> 5; low nibble holds the even element.
    assert packed[0, :2].tolist() == [(0xF << 4) | 0x1, (0x5 << 4) | 0x0]
    assert scale.float().item() == 1.0


def test_off_lattice_weight_is_refused():
    r, S = _lattice_weight("standard")
    r[7, 3] = r[7, 3] * 1.01 + 1e-3
    with pytest.raises(ValueError, match="not on the NVFP4 lattice"):
        nvfp4.pack(r, S)


def test_global_scale_round_trip():
    bad = torch.tensor(float.fromhex("0x1.6a1226p+0"))  # the first fp32 S >= 1 with RN(1 / RN(1 / S)) != S
    with pytest.raises(ValueError):
        nvfp4.global_scale(bad)
    ref = bad * E4M3_MAX
    snapped = ct_divisor_snap(ref)  # training moves such a reference up to the next storable one
    assert torch.equal(snapped, torch.nextafter(ref, torch.tensor(float("inf"))))
    nvfp4.global_scale(snapped / E4M3_MAX)
    one = torch.ones((), dtype=torch.float32)
    assert torch.equal(one / nvfp4.global_scale(torch.tensor(0.75)), torch.tensor([0.75]))


def _materialized_nvfp4(tmp_path, method="quasar"):
    model = tiny_model("qwen3")
    base = save_base(model, tmp_path / "base")
    quantize_model(model, QuantConfig(method, "nvfp4"))
    for _, m in quant_linears(model):
        for _ in range(SALIENCY_WARMUP if method == "quasar" else 0):
            m.quantizer.update_saliency(torch.rand(m.weight.shape) * 1e-9)
    out = tmp_path / "materialized"
    save_materialized(model, out, model_path=base)
    return out


@pytest.mark.parametrize("method", ["standard", "quasar"])
def test_export(method, tmp_path):
    src = _materialized_nvfp4(tmp_path, method)
    ct = tmp_path / "ct"
    nvfp4.main(["--materialized", str(src), "--out_dir", str(ct)])
    packed = SafetensorsReader(ct).keys()
    assert sum(k.endswith(".weight_packed") for k in packed) == 14
    assert "model.layers.0.self_attn.q_proj.weight" not in packed and "model.norm.weight" in packed
    qc = json.loads((ct / "config.json").read_text())["quantization_config"]
    assert qc["format"] == "nvfp4-pack-quantized" and qc["config_groups"]["group_0"]["input_activations"] is None
    assert "lm_head" in qc["ignore"] and not any("layers" in i for i in qc["ignore"])
    for name in ("tokenizer.json", "receipt.json"):
        assert (ct / name).read_bytes() == (src / name).read_bytes()
    a, b = SafetensorsReader(src), SafetensorsReader(ct)
    for key in a.keys():  # packed linears decode to the materialized weights; everything else is copied
        m = key.removesuffix(".weight")
        if m + ".weight_packed" in packed:
            w = nvfp4.dequantize(b[m + ".weight_packed"], b[m + ".weight_scale"], b[m + ".weight_global_scale"])
        else:
            w = b[key]
        assert torch.equal(w, a[key]), key


def test_export_refuses_unshared_fused_scales(tmp_path):
    src = _materialized_nvfp4(tmp_path, "standard")
    refs_file = src / "nvfp4_tensor_refs.json"
    refs = json.loads(refs_file.read_text())
    key = "model.layers.1.self_attn.k_proj.weight"
    refs[key] = (float.fromhex(refs[key]) * 2).hex()
    refs_file.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="fused layer"):
        nvfp4.export(src, tmp_path / "ct")
    del refs[key]
    refs_file.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="decoder projections"):
        nvfp4.export(src, tmp_path / "ct")
    assert not (tmp_path / "ct").exists()  # refused before anything is written
