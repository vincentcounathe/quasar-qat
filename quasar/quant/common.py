"""Grouping, straight-through rounding and the (weighted) least-squares affine fit.

Statistics are computed in fp32; callers cast reconstructions back to the weight dtype.
"""

from __future__ import annotations

import torch

from .config import EPS


def to_blocks(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """View a ``[out, in]`` weight as ``[out, in / group_size, group_size]`` groups."""
    if w.dim() != 2 or w.shape[1] % group_size:
        raise ValueError(f"weight {tuple(w.shape)} cannot be split into groups of {group_size}")
    return w.reshape(w.shape[0], w.shape[1] // group_size, group_size)


def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Value of ``q``, gradient of ``x`` (straight-through estimator)."""
    return x + (q - x).detach()


def wmean(t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Per-group mean of ``t`` weighted by ``h``."""
    return (h * t).sum(dim=-1, keepdim=True) / h.sum(dim=-1, keepdim=True).clamp_min(EPS)


def fit_affine(q: torch.Tensor, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
    """Least-squares dequantizer ``r = s * (q - mean_q) + mean_x`` per group.

    The scale and offset are refit to the codes ``q`` instead of being read off
    the clipping range, weighted by ``h`` (``None``: unweighted).
    """
    mean = (lambda t: t.mean(dim=-1, keepdim=True)) if h is None else (lambda t: wmean(t, h))
    mean_q, mean_x = mean(q), mean(x)
    s = (mean(q * x) - mean_q * mean_x) / (mean(q * q) - mean_q * mean_q + EPS)
    return s * (q - mean_q) + mean_x
