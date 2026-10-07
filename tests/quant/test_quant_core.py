"""CPU tests of the quantizer core: lattices, gradients, QUASAR's objective, model wiring."""

import pytest
import torch
from helpers import tiny_model
from torch import nn

from quasar.quant import (
    DECODER_PROJ_RE,
    FakeQuantizer,
    QuantConfig,
    apply_bitdistiller_clip,
    quant_linears,
    quantize_model,
    snapshot_saliency,
)
from quasar.quant import config as quant_config
from quasar.quant.baselines import bitdistiller, clip_search
from quasar.quant.config import INT_GRID, NVFP4_GRID, SALIENCY_WARMUP
from quasar.quant.nvfp4 import E2M1_LEVELS, E4M3_MAX

INT_METHODS = ["standard", "lsq", "denoising", "bitdistiller", "quasar"]
DTYPES = [torch.float32, torch.bfloat16]


def _weights(rows=16, cols=256, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * 0.02
    w.view(-1)[torch.randint(0, w.numel(), (w.numel() // 100,), generator=g)] *= 20  # outliers
    return w.to(dtype)


def _quantizer(method, fmt="int", bits=4, w=None, saliency_seed=None):
    qz = FakeQuantizer(QuantConfig(method, fmt, bits))
    if method == "lsq":
        qz.init_lsq(w)
    if saliency_seed is not None:
        g = torch.Generator().manual_seed(saliency_seed)
        for _ in range(SALIENCY_WARMUP):
            qz.update_saliency(torch.exp(torch.randn(w.shape, generator=g) * 3 - 20))
    return qz


def _group_error(r, w, h, g):
    return (h.reshape(-1, g) * (r.double() - w.double()).reshape(-1, g) ** 2).sum(-1)


def test_config_defaults_and_validation():
    c = QuantConfig("quasar", bits=2)
    assert (c.group_size, c.grid, c.qmax) == (128, INT_GRID, 3)
    assert len(INT_GRID) == 15 and INT_GRID[0] == 0.30 and INT_GRID[-1] == 1.0
    n = QuantConfig("quasar", "nvfp4")
    assert (n.group_size, n.grid) == (16, NVFP4_GRID)
    for bad in [
        dict(method="gptq"),
        dict(method="lsq", format="nvfp4"),
        dict(method="quasar", bits=5),
        dict(method="standard", format="nvfp4", bits=2),
    ]:
        with pytest.raises(ValueError):
            QuantConfig(**bad)


@pytest.mark.parametrize("method", INT_METHODS)
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("dtype", DTYPES)
def test_int_shapes_dtypes_and_lattice(method, bits, dtype):
    w = _weights(dtype=dtype, seed=bits)
    qz = _quantizer(method, bits=bits, w=w, saliency_seed=1 if method == "quasar" else None)
    with torch.no_grad():
        r = qz(w)
    assert r.shape == w.shape and r.dtype == dtype and torch.isfinite(r).all()
    if dtype == torch.float32:
        # Every group is the affine image of at most 2^bits integer codes.
        for row in r.reshape(-1, 128):
            assert row.unique().numel() <= 2**bits
    assert (r.float() - w.float()).abs().max() < w.float().abs().max()


def _on_nvfp4_lattice(r, ref):
    """Each group of 16 is S * m * e2m1 codes for one E4M3 value m, S = ref / 448 (up to fp32 rounding)."""
    levels = torch.tensor(E2M1_LEVELS, dtype=torch.float64)
    for row in r.double().reshape(-1, 16) / (float(ref) / E4M3_MAX):
        top = row.abs().max()
        fits = top == 0
        for level in levels[1:]:  # whichever level the group's largest magnitude landed on
            m = (top / level).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).double()
            if not fits and (top / level - m).abs() <= 1e-5 * m:
                fits = bool(((row.abs()[:, None] / m - levels).abs().min(dim=1).values < 1e-5).all())
        if not fits:
            return False
    return True


@pytest.mark.parametrize("method", ["standard", "quasar"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_nvfp4_lattice_and_ref(method, dtype):
    w = _weights(8, 128, dtype, seed=3)
    qz = _quantizer(method, "nvfp4", w=w, saliency_seed=2 if method == "quasar" else None)
    with torch.no_grad():
        r = qz(w)
    assert r.shape == w.shape and r.dtype == dtype
    ref = qz.last_tensor_ref
    second = (ref / E4M3_MAX).float()
    assert torch.equal(1 / (1 / second.double()).float(), second)  # compressed-tensors divisor round trip
    if dtype == torch.float32:
        assert _on_nvfp4_lattice(r, ref)


def test_zero_and_constant_weights_stay_finite():
    for method, fmt in [(m, "int") for m in INT_METHODS] + [("standard", "nvfp4"), ("quasar", "nvfp4")]:
        w = torch.zeros(4, 128)
        w[1] = 0.25
        qz = _quantizer(method, fmt, w=w, saliency_seed=0 if method == "quasar" else None)
        with torch.no_grad():
            r = qz(w)
        assert torch.isfinite(r).all(), method
        torch.testing.assert_close(r[0], w[0])
    for method in ["standard", "quasar"]:  # an all-zero NVFP4 tensor (per-tensor scale 0)
        with torch.no_grad():
            z = torch.zeros(4, 32)
            assert torch.equal(_quantizer(method, "nvfp4", w=z, saliency_seed=0)(z), z)


def test_quasar_backward_is_identity():
    for fmt in ["int", "nvfp4"]:
        w = _weights(8, 128, seed=4).requires_grad_(True)
        qz = _quantizer("quasar", fmt, bits=2 if fmt == "int" else 4, w=w.detach(), saliency_seed=5)
        up = torch.randn(w.shape)
        r = qz(w)
        (r * up).sum().backward()
        assert torch.equal(w.grad, up)
        with torch.no_grad():
            assert torch.equal(r.detach(), qz(w))


@pytest.mark.parametrize("method", ["standard", "lsq", "bitdistiller"])
def test_baseline_ste_passes_gradient_inside_the_range(method):
    w = _weights(8, 128, seed=6).requires_grad_(True)
    qz = _quantizer(method, bits=3, w=w.detach())
    up = torch.randn(w.shape)
    (qz(w) * up).sum().backward()
    blocks = w.detach().reshape(8, 1, 128)
    inner = ((blocks > blocks.amin(-1, keepdim=True)) & (blocks < blocks.amax(-1, keepdim=True))).reshape(w.shape)
    passed = (w.grad - up).abs() <= 1e-5 * up.abs() + 1e-6
    if method == "bitdistiller":  # its integer zero-point can push top codes into the clamp (zero gradient)
        assert (passed | (w.grad == 0))[inner].all() and passed[inner].float().mean() > 0.9
    else:
        assert passed[inner].all()
    if method == "lsq":
        assert qz.lsq_scale.grad.abs().sum() > 0 and qz.lsq_beta.grad.abs().sum() > 0


def test_denoising_gradient_flows_through_the_fit():
    w = _weights(8, 128, seed=7).requires_grad_(True)
    qz = _quantizer("denoising", bits=2)
    up = torch.randn(w.shape)
    (qz(w) * up).sum().backward()
    assert torch.isfinite(w.grad).all() and not torch.allclose(w.grad, up)


def test_lsq_starts_on_the_standard_grid():
    w = _weights(seed=8)
    with torch.no_grad():
        lsq, standard = _quantizer("lsq", bits=2, w=w)(w), _quantizer("standard", bits=2)(w)
    torch.testing.assert_close(lsq, standard, rtol=0, atol=1e-6)


def _without_search(qz, w, monkeypatch):
    """``qz(w)`` with the clipping search disabled (grid {1}: min/max codes + the weighted fit)."""
    with monkeypatch.context() as m:
        m.setattr(quant_config, "INT_GRID", (1.0,))
        m.setattr(quant_config, "NVFP4_GRID", (1.0,))
        return qz(w)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("dtype", DTYPES)
def test_quasar_never_worse_than_minmax(bits, dtype, monkeypatch):
    """The grid contains f = 1 (the min/max codes) and the weighted fit is optimal for its codes;
    the search over clipping ranges then improves on f = 1 alone."""
    w = _weights(32, 512, dtype, seed=10 + bits)
    qz = _quantizer("quasar", bits=bits, w=w, saliency_seed=11)
    h = qz.saliency
    with torch.no_grad():
        e_q = _group_error(qz(w), w, h, 128)
        e_1 = _group_error(_without_search(qz, w, monkeypatch), w, h, 128)
        e_s = _group_error(_quantizer("standard", bits=bits)(w), w, h, 128)
    slack = 1e-2 if dtype == torch.bfloat16 else 1e-6  # bf16 storage rounding of the fitted values
    for e_ref in (e_1, e_s):
        assert (e_q <= e_ref * (1 + slack) + 1e-20).all()
    assert e_q.sum() < 0.8 * e_1.sum()


def test_quasar_nvfp4_search_beats_standard_and_no_search(monkeypatch):
    w = _weights(64, 256, seed=12)
    qz = _quantizer("quasar", "nvfp4", w=w, saliency_seed=13)
    with torch.no_grad():
        e_q = _group_error(qz(w), w, qz.saliency, 16).sum()
        e_1 = _group_error(_without_search(qz, w, monkeypatch), w, qz.saliency, 16).sum()
        e_s = _group_error(_quantizer("standard", "nvfp4")(w), w, qz.saliency, 16).sum()
    assert e_q < e_s and e_q < 0.8 * e_1


def test_saliency_warmup():
    w = _weights(seed=14)
    qz = _quantizer("quasar", bits=2)
    v = torch.exp(torch.randn(w.shape, generator=torch.Generator().manual_seed(15)) * 3 - 20)
    with torch.no_grad():
        uniform = qz(w)  # no snapshot yet
        for _ in range(SALIENCY_WARMUP - 1):
            qz.update_saliency(v)
        assert torch.equal(qz(w), uniform)  # still warming up
        qz.update_saliency(v)
        assert not torch.equal(qz(w), uniform)


def test_reused_search_is_redone_after_a_snapshot():
    """The shard-local gather path (training, bypass) reuses a step's winners; the next
    step's saliency snapshot must trigger a fresh search."""
    w = _weights(seed=16)
    qz = _quantizer("quasar", bits=2, w=w, saliency_seed=17)
    qz.bypass = True
    g = torch.Generator().manual_seed(18)
    with torch.no_grad():
        qz(w)
        w = w + 0.005 * torch.randn(w.shape, generator=g)  # optimizer step ...
        qz.update_saliency(torch.exp(torch.randn(w.shape, generator=g) * 3 - 20))  # ... and its snapshot
        after = qz(w)
        assert torch.equal(qz(w), after)  # reused within the step
        qz.eval()  # no reuse: a fresh search
        assert torch.equal(qz(w), after)


def _ids(n=2, seq=8):
    return torch.randint(0, 256, (n, seq))


@pytest.mark.parametrize("method", INT_METHODS)
def test_quantize_model_wiring(method):
    model = tiny_model().float()
    params = {n: p for n, p in model.named_parameters()}
    names = quantize_model(model, QuantConfig(method, bits=2))
    assert len(names) == 14 and all(DECODER_PROJ_RE.fullmatch(n) for n in names)
    assert type(model.lm_head) is nn.Linear and model.lm_head.weight is model.model.embed_tokens.weight
    for n, m in quant_linears(model):
        assert m.weight is params[n + ".weight"]
        x = torch.randn(3, m.in_features)
        torch.testing.assert_close(m(x), torch.nn.functional.linear(x, m.quantizer(m.weight)))
    assert hasattr(model.model.layers[0].mlp.up_proj.quantizer, "lsq_scale") == (method == "lsq")
    with pytest.raises(RuntimeError):
        quantize_model(model, QuantConfig(method, bits=2))


def test_snapshot_saliency_from_adamw():
    model = tiny_model().float()
    quantize_model(model, QuantConfig("quasar", bits=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        model(input_ids=_ids()).logits.logsumexp(-1).mean().backward()
        opt.step()
        opt.zero_grad()
        snapshot_saliency(model, opt)
    down = model.model.layers[1].mlp.down_proj
    assert down.quantizer.saliency_steps == 3
    assert torch.equal(down.quantizer.saliency, opt.state[down.weight]["exp_avg_sq"])
    assert not any("saliency" in k for k in model.state_dict())


@pytest.mark.parametrize("method", ["standard", "quasar"])
def test_nvfp4_fused_groups_share_one_tensor_scale(method):
    model = tiny_model().float()
    quantize_model(model, QuantConfig(method, "nvfp4"))
    for b in model.model.layers:  # vLLM's fused q/k/v and gate/up layers
        a, m = b.self_attn, b.mlp
        assert a.q_proj.quantizer.fused_ref is a.k_proj.quantizer.fused_ref is a.v_proj.quantizer.fused_ref
        assert m.gate_proj.quantizer.fused_ref is m.up_proj.quantizer.fused_ref
        assert a.o_proj.quantizer.fused_ref is None and m.down_proj.quantizer.fused_ref is None
    layer0, layer1 = (b.self_attn.q_proj.quantizer.fused_ref for b in model.model.layers)
    assert layer0 is not layer1
    with torch.no_grad():
        for _ in range(2):  # the second pass sees every partition's current reference
            model(input_ids=_ids())
    for b in model.model.layers:
        refs = {n: float(getattr(b.self_attn, f"{n}_proj").quantizer.last_tensor_ref) for n in "qkvo"}
        assert refs["q"] == refs["k"] == refs["v"]
        assert float(b.mlp.gate_proj.quantizer.last_tensor_ref) == float(b.mlp.up_proj.quantizer.last_tensor_ref)
        own = b.self_attn.o_proj.weight.detach().abs().reshape(-1, 16).amax(-1).max() / 6
        grow = max(NVFP4_GRID) if method == "quasar" else 1.0
        assert own * grow <= refs["o"] <= own * grow * (1 + 1e-5)  # own scale, snapped up by a few ulps at most


def test_bitdistiller_clip():
    model = tiny_model().float()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    n = apply_bitdistiller_clip(model, _ids(4, 16), QuantConfig("bitdistiller", bits=2))
    assert n == 10  # q/k skipped
    for name, p in model.named_parameters():
        if name.endswith(("q_proj.weight", "k_proj.weight")) or "proj" not in name:
            assert torch.equal(p, before[name]), name
        else:
            assert (p.abs() <= before[name].abs() + 1e-7).all()


def test_clip_search_never_worse_than_unclipped():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 2, 32, generator=g)
    w.view(-1)[::37] *= 8
    feat = torch.randn(40, 2, 32, generator=g)
    best_max, best_min = clip_search(w, feat, qmax=3)
    assert (best_max <= w.amax(-1, keepdim=True)).all() and (best_min >= w.amin(-1, keepdim=True)).all()

    def err(lo, hi):
        q = bitdistiller(torch.clamp(w, lo, hi), 3)
        return (((feat.unsqueeze(0) * (q - w).unsqueeze(1)).sum(-1)) ** 2).mean(1)

    unclipped = err(w.amin(-1, keepdim=True), w.amax(-1, keepdim=True))
    assert (err(best_min, best_max) <= unclipped * (1 + 1e-5) + 1e-6).all()
