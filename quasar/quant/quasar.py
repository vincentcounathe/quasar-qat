"""QUASAR: saliency-weighted clipping-range search with a least-squares dequantizer fit.

For every group QUASAR tries each clipping factor ``f`` of the grid, rounds the
weights onto the clipped grid, fits the dequantizer (scale and offset for INT, a
scale for NVFP4) by saliency-weighted least squares, and keeps the factor with
the lowest saliency-weighted reconstruction error. The saliency is a snapshot of
AdamW's ``exp_avg_sq``, normalized to mean 1 within each group. Nothing here is
differentiated: the caller applies an identity straight-through estimator.
"""

from __future__ import annotations

import torch

from . import kernel
from .common import fit_affine, to_blocks, wmean
from .config import EPS, SALIENCY_FLOOR, SALIENCY_WARMUP
from .nvfp4 import E2M1_MAX, project_e2m1, tensor_ref, two_level_scale, two_level_slope


def saliency_weights(quantizer, w: torch.Tensor) -> torch.Tensor:
    """Per-group saliency with mean 1, shaped like ``w``'s groups (fp32): the quantizer's
    AdamW snapshot once ``SALIENCY_WARMUP`` snapshots exist, uniform before that."""
    g = quantizer.config.group_size
    if quantizer.saliency_steps < SALIENCY_WARMUP:
        return torch.ones(to_blocks(w, g).shape, dtype=torch.float32, device=w.device)
    if quantizer.saliency.shape != w.shape:
        raise ValueError(f"saliency {tuple(quantizer.saliency.shape)} does not match weight {tuple(w.shape)}")
    h = to_blocks(quantizer.saliency.to(device=w.device, dtype=torch.float32), g).clamp_min(SALIENCY_FLOOR)
    return h / h.mean(dim=-1, keepdim=True)


@torch.no_grad()
def quasar(x: torch.Tensor, h: torch.Tensor, quantizer, reuse: bool) -> torch.Tensor:
    """Fake-quantize grouped weights ``x`` with saliency weights ``h`` (same shape).

    ``reuse``: weights and saliency cannot have changed since the last search at
    the same saliency step (micro-batches of one optimizer step, the backward
    re-gather), so the previous winners are reused unless the NVFP4 per-tensor
    reference moved (a fused partition's reference follows its partners).
    """
    cfg = quantizer.config
    xf = x.float()
    ref_key = None
    if cfg.format == "int":
        lo, hi = xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1, keepdim=True)
        center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)

        def recon(f):
            c_lo, c_hi = center - half * f, center + half * f
            q = torch.round((xf - c_lo) / (c_hi - c_lo + EPS) * cfg.qmax).clamp(0, cfg.qmax)
            return fit_affine(q, xf, h)
    else:
        base = xf.abs().amax(dim=-1, keepdim=True) / E2M1_MAX
        # Candidates up to max(grid) must stay representable, so the grid max sets the tensor scale.
        ref = tensor_ref(base, quantizer, max(cfg.grid))
        ref_key = float(ref)

        def recon(f):
            q = project_e2m1(xf / (two_level_scale(base * f, ref) + EPS))
            return two_level_slope(wmean(q * xf, h) / (wmean(q * q, h) + EPS), ref) * q

    # Before the first snapshot the step count does not identify the weights: no reuse.
    key, cached = (quantizer.saliency_steps, ref_key), quantizer._search_cache
    if reuse and key[0] > 0 and cached is not None and cached[0] == key:
        best = cached[1]
    elif cfg.format == "int" and kernel.TRITON_AVAILABLE and x.is_cuda:
        best = kernel.affine_sweep(xf, h, cfg.grid, cfg.qmax, x.dtype == torch.bfloat16, EPS)
    else:
        best = _sweep(recon, xf, h, cfg.grid, x.dtype)
    if reuse:
        quantizer._search_cache = (key, best)
    return recon(best).to(x.dtype)


def _sweep(recon, xf: torch.Tensor, h: torch.Tensor, grid, dtype: torch.dtype) -> torch.Tensor:
    """Per-group arg-min over ``grid`` of the weighted error of ``recon(f)`` stored in ``dtype``."""
    best = best_err = None
    for f in grid:
        err = wmean((recon(f).to(dtype).float() - xf).square(), h)
        if best is None:
            best, best_err = torch.full_like(err, f), err
        else:
            better = err < best_err  # strict: ties keep the earlier factor
            best, best_err = best.masked_fill(better, f), torch.where(better, err, best_err)
    return best
