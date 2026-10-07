"""Fused Triton sweep for QUASAR's INT clipping-range search.

The sweep is the training hot loop (every grid factor for every weight, every
optimizer step). One program per group loads the group's weights and saliency
once and scores every factor in registers: codes on the clipped range, the
saliency-weighted least-squares fit, and the weighted error of the fitted
reconstruction as stored in the weight dtype. Only the arg-min factor is
returned; the caller recomputes the winner in PyTorch. Codes round half up and
reductions run in a different order than PyTorch, so a group whose two best
factors tie up to rounding may pick either one.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # CPU-only installs use the PyTorch sweep
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _sweep_kernel(
        x_ptr,
        h_ptr,
        grid_ptr,
        out_ptr,
        qmax,
        eps,
        GROUP: tl.constexpr,
        N_GRID: tl.constexpr,
        ROUND_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        mask = lane < GROUP
        x = tl.load(x_ptr + pid * GROUP + lane, mask=mask, other=0.0).to(tl.float32)
        h = tl.load(h_ptr + pid * GROUP + lane, mask=mask, other=0.0).to(tl.float32)
        sum_h = tl.maximum(tl.sum(h, axis=0), eps)
        x_min = tl.min(tl.where(mask, x, 1.0e38), axis=0)
        x_max = tl.max(tl.where(mask, x, -1.0e38), axis=0)
        center = 0.5 * (x_min + x_max)
        half = 0.5 * (x_max - x_min)
        mean_x = tl.sum(h * x, axis=0) / sum_h

        best_err = 1.0e38
        best = tl.load(grid_ptr)
        for i in range(N_GRID):
            f = tl.load(grid_ptr + i)
            lo = center - half * f
            hi = center + half * f
            q = tl.floor((x - lo) / (hi - lo + eps) * qmax + 0.5)
            q = tl.minimum(tl.maximum(q, 0.0), qmax)
            mean_q = tl.sum(h * q, axis=0) / sum_h
            var_q = tl.sum(h * q * q, axis=0) / sum_h - mean_q * mean_q
            cov = tl.sum(h * q * x, axis=0) / sum_h - mean_q * mean_x
            r = cov / (var_q + eps) * (q - mean_q) + mean_x
            if ROUND_BF16:
                r = r.to(tl.bfloat16).to(tl.float32)
            d = r - x  # padded lanes carry h = 0
            err = tl.sum(h * d * d, axis=0) / sum_h
            better = err < best_err  # strict: ties keep the earlier factor
            best = tl.where(better, f, best)
            best_err = tl.where(better, err, best_err)
        tl.store(out_ptr + pid, best)


def affine_sweep(x: torch.Tensor, h: torch.Tensor, grid, qmax: int, round_bf16: bool, eps: float) -> torch.Tensor:
    """Best clipping factor per group of ``[..., group]`` fp32 CUDA tensors, shaped ``[..., 1]``."""
    group = x.shape[-1]
    x2 = x.reshape(-1, group).contiguous()
    h2 = h.reshape(-1, group).contiguous().float()
    grid_t = torch.tensor(grid, device=x.device, dtype=torch.float32)
    out = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(group)
    _sweep_kernel[(x2.shape[0],)](
        x2,
        h2,
        grid_t,
        out,
        float(qmax),
        float(eps),
        GROUP=group,
        N_GRID=len(grid),
        ROUND_BF16=round_bf16,
        BLOCK=block,
        # One warp keeps every reduction of a 128-wide group in warp shuffles (~4x faster).
        num_warps=max(1, min(4, block // 128)),
    )
    return out.view(*x.shape[:-1], 1)
