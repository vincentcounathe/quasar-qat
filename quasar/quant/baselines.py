"""Baseline fake quantizers: Standard QAT, Denoising QAT, LSQ and BitDistiller.

All run in the module forward with autograd, so their gradients are part of the
method: straight-through through the rounding, plus whatever flows through the
range statistics (min/max), the least-squares refit (Denoising) or the learned
step and offset (LSQ).
"""

from __future__ import annotations

import math

import torch

from .common import fit_affine, ste, to_blocks
from .config import DECODER_PROJ_RE, EPS, QuantConfig
from .nvfp4 import E2M1_MAX, project_e2m1, tensor_ref, two_level_scale

# BitDistiller's clip search: calibration tokens per linear, and the range ends shrink in
# steps of 1 / BD_GRID, by less than BD_MAX_SHRINK.
BD_SAMPLE_TOKENS = 512
BD_GRID, BD_MAX_SHRINK = 20, 0.5
BD_BATCH = 8  # calibration sequences per forward


def _minmax_codes(x: torch.Tensor, qmax: int):
    """fp32 weights, group min and max, and straight-through codes on the full min/max range."""
    xf = x.float()
    lo, hi = xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1, keepdim=True)
    v = (xf - lo) / (hi - lo + EPS) * qmax
    return xf, lo, hi, ste(v, v.round().clamp(0, qmax))


def standard(x: torch.Tensor, qmax: int) -> torch.Tensor:
    """Standard QAT (INT): round-to-nearest on the min/max range."""
    _, lo, hi, q = _minmax_codes(x, qmax)
    return (q / qmax * (hi - lo) + lo).to(x.dtype)


def standard_nvfp4(x: torch.Tensor, quantizer) -> torch.Tensor:
    """Standard QAT (NVFP4): absmax group scales on the two-level lattice."""
    xf = x.float()
    s = xf.abs().amax(dim=-1, keepdim=True) / E2M1_MAX
    s = two_level_scale(s, tensor_ref(s, quantizer))
    y = xf / (s + EPS)
    return (s * ste(y, project_e2m1(y))).to(x.dtype)


def denoising(x: torch.Tensor, qmax: int) -> torch.Tensor:
    """Denoising QAT: min/max codes, then an unweighted least-squares refit of scale and offset."""
    xf, _, _, q = _minmax_codes(x, qmax)
    return fit_affine(q, xf).to(x.dtype)


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward identity, gradient multiplied by ``scale`` (Esser et al.)."""
    return (x - x * scale).detach() + x * scale


def lsq(x: torch.Tensor, step: torch.Tensor, offset: torch.Tensor, qmax: int) -> torch.Tensor:
    """LSQ with a learned per-group offset (LSQ+), gradient scale ``1 / sqrt(group_size * qmax)``."""
    g = 1.0 / math.sqrt(x.shape[-1] * qmax)
    s = _grad_scale(step.float(), g) + EPS
    beta = _grad_scale(offset.float(), g)
    v = ((x.float() - beta) / s).clamp(0, qmax)
    return (ste(v, v.round()) * s + beta).to(x.dtype)


@torch.no_grad()
def lsq_init(x: torch.Tensor, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    """RTN initialization: LSQ starts exactly on the Standard QAT code map."""
    xf = x.float()
    lo, hi = xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1, keepdim=True)
    step = (hi - lo + EPS) / qmax
    return (step - EPS).clamp_min(0.0), lo


def bitdistiller(x: torch.Tensor, qmax: int) -> torch.Tensor:
    """BitDistiller's quantizer: min/max scale and an integer zero-point (no gradient through it)."""
    xf = x.float()
    lo, hi = xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1, keepdim=True)
    s = (hi - lo).clamp(min=1e-5) / qmax
    z = (-torch.round(lo / s)).clamp(0, qmax)
    v = xf / s
    return ((torch.clamp(ste(v, v.round()) + z, 0, qmax) - z) * s).to(x.dtype)


@torch.no_grad()
def clip_search(w: torch.Tensor, feat: torch.Tensor, qmax: int):
    """BitDistiller's per-group two-sided clip search.

    ``w``: ``[out, n_groups, g]``; ``feat``: ``[n_tok, n_groups, g]`` calibration
    inputs. Shrinks the max and the min independently and keeps the pair with
    the lowest per-group output MSE.
    Returns ``(best_max, best_min)``, each ``[out, n_groups, 1]``.
    """
    best_max, best_min = w.amax(dim=-1, keepdim=True), w.amin(dim=-1, keepdim=True)
    feat = feat.unsqueeze(0)
    steps = int(BD_MAX_SHRINK * BD_GRID)
    for o in range(0, w.shape[0], 128):  # chunks of output rows bound the memory
        wc = w[o : o + 128].unsqueeze(1)
        w_max, w_min = wc.amax(dim=-1, keepdim=True), wc.amin(dim=-1, keepdim=True)
        b_max, b_min, b_err = w_max, w_min, torch.full_like(w_max, 1e9)
        ref_out = (feat * wc).sum(dim=-1)
        for i in range(steps):
            for j in range(steps):
                c_max, c_min = w_max * (1 - i / BD_GRID), w_min * (1 - j / BD_GRID)
                q = bitdistiller(torch.clamp(wc, c_min, c_max), qmax)
                err = ((feat * q).sum(dim=-1) - ref_out).pow(2).mean(dim=1).view(b_err.shape)
                better = err < b_err
                b_err = torch.where(better, err, b_err)
                b_max, b_min = torch.where(better, c_max, b_max), torch.where(better, c_min, b_min)
        best_max[o : o + 128], best_min[o : o + 128] = b_max[:, 0], b_min[:, 0]
    return best_max, best_min


@torch.no_grad()
def apply_bitdistiller_clip(model: torch.nn.Module, calib_input_ids: torch.Tensor, config: QuantConfig) -> int:
    """BitDistiller's one-shot initialization: clip the full-precision weights in place.

    Run on the unquantized model, on its compute device, before :func:`quantize_model`
    with the same ``config``. Captures the inputs of every decoder projection except
    q/k (BitDistiller skips them: the q.k product makes their output error a poor
    proxy) over ``calib_input_ids`` (``[n_seq, seqlen]``), searches per-group clip
    bounds on about ``BD_SAMPLE_TOKENS`` evenly strided tokens and clamps the
    weights. Returns the number of clipped linears.
    """
    linears = {
        name: m
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and DECODER_PROJ_RE.fullmatch(name)
        and not name.endswith(("q_proj", "k_proj"))
    }
    if not linears:
        raise RuntimeError("bitdistiller clip: no decoder projections found")
    stride = max(1, calib_input_ids.numel() // BD_SAMPLE_TOKENS)  # every linear sees every token once
    feats: dict[str, list[torch.Tensor]] = {name: [] for name in linears}

    def capture(name):
        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            feats[name].append(x[::stride].to(torch.bfloat16).cpu())

        return hook

    hooks = [m.register_forward_pre_hook(capture(name)) for name, m in linears.items()]
    device, was_training = next(model.parameters()).device, model.training
    model.eval()
    try:
        for i in range(0, calib_input_ids.shape[0], BD_BATCH):
            model(input_ids=calib_input_ids[i : i + BD_BATCH].to(device))
    finally:
        for hook in hooks:
            hook.remove()

    for name, m in linears.items():
        feat = torch.cat(feats.pop(name)).to(device=device, dtype=torch.float32)
        if not torch.isfinite(feat).all():
            raise RuntimeError(f"bitdistiller clip: non-finite calibration inputs for {name}")
        w = to_blocks(m.weight.detach().float(), config.group_size)
        best_max, best_min = clip_search(w, to_blocks(feat, config.group_size), config.qmax)
        m.weight.copy_(torch.clamp(w, best_min, best_max).reshape(m.weight.shape))
    model.train(was_training)
    return len(linears)
