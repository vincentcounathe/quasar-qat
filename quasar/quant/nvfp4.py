"""NVFP4 lattice: E2M1 codes, E4M3 group scales and one FP32 per-tensor scale.

A deployed NVFP4 weight is ``S * e4m3(s_g / S) * e2m1(code)`` with ``S = ref / 448``.
Training fake-quantizes onto exactly this lattice, so ``ref`` must be the value
deployment will store:

* shard-local search sees only a rank's rows, so ``ref`` is MAX-reduced over the
  quantizer's process group;
* vLLM fuses q/k/v and gate/up into one layer and keeps the max of the
  partitions' tensor scales, so fused partitions share one ``ref`` (:class:`FusedRef`);
* compressed-tensors stores the divisor ``1/S``, so ``ref`` is nudged up to a value
  whose ``S`` survives that round trip exactly (:func:`ct_divisor_snap`).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .common import ste

E2M1_MAX = 6.0
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
E4M3_MIN = 2.0**-9  # smallest subnormal: group scales never collapse to zero
_CT_SNAP_MAX_ULPS = 64

# Partitions that vLLM fuses into one layer of a dense decoder.
FUSED_GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("mlp.gate_proj", "mlp.up_proj"),
)


def project_e2m1(y: torch.Tensor) -> torch.Tensor:
    """Nearest signed E2M1 level; exact midpoints round toward zero, ``|y| > 6`` saturates."""
    levels = torch.tensor(E2M1_LEVELS, device=y.device, dtype=y.dtype)
    mids = (levels[1:] + levels[:-1]) / 2
    return torch.sign(y) * levels[torch.bucketize(y.abs(), mids)]


def _e4m3(y: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E4M3 value (saturating at +-448), returned in ``y.dtype``."""
    return y.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(y.dtype)


def _second_level(ref: torch.Tensor) -> torch.Tensor:
    # The clamp only matters for an all-zero tensor (ref == 0), whose weights stay zero.
    return (ref / E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)


def two_level_scale(s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Positive per-group scales onto ``S * e4m3(s / S)`` (straight-through)."""
    second = _second_level(ref)
    y = s / second
    return second * ste(y, _e4m3(y.detach().clamp(E4M3_MIN, E4M3_MAX)))


def two_level_slope(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Fitted (signed, possibly zero) per-group multipliers onto the same lattice."""
    second = _second_level(ref)
    return second * _e4m3(a / second)


def _rn32_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Correctly rounded fp32 division on any device (CUDA's tensor/scalar division is not)."""
    return (num.double() / den.double()).float()


def ct_divisor_snap(ref: torch.Tensor) -> torch.Tensor:
    """Smallest ``ref' >= ref`` whose second-level scale compressed-tensors stores exactly.

    ``ref`` is a 0-d fp32 tensor. Accepts ``S = RN(ref/448)`` when ``S == RN(1/RN(1/S))``
    (the stored divisor round trip) and the device's own ``ref / 448`` equals ``S``.
    Moving up keeps ``ref`` an upper bound of the group scales.
    """
    one, e4m3_max = torch.ones_like(ref), torch.full_like(ref, E4M3_MAX)
    out = ref
    for _ in range(_CT_SNAP_MAX_ULPS):
        second = _rn32_div(out, e4m3_max)
        if bool((_rn32_div(one, _rn32_div(one, second)) == second) & (out / E4M3_MAX == second)):
            return out
        out = torch.nextafter(out, torch.full_like(out, float("inf")))
    raise RuntimeError(f"CT divisor snap did not converge within {_CT_SNAP_MAX_ULPS} ulps (ref={float(ref)!r})")


class FusedRef:
    """Per-tensor reference shared by the partitions of one fused deployment layer.

    Each member records its latest reference and uses the group max. Entries are
    overwritten, so the shared value tracks the current weights; a member
    quantized earlier in a step sees the others' references from their previous forward.
    """

    def __init__(self):
        self.refs: dict[int, float] = {}

    def update(self, quantizer, ref: torch.Tensor) -> torch.Tensor:
        self.refs[id(quantizer)] = float(ref)
        return ref.new_tensor(max(self.refs.values()))


def tensor_ref(scales: torch.Tensor, quantizer, factor: float = 1.0) -> torch.Tensor:
    """The per-tensor reference deployment will store: ``factor * max(scales)`` over the tensor.

    ``scales`` are this tensor's (or row shard's) per-group scales. The result is
    detached: the per-tensor scale is a deployment constant, not a trained
    quantity. It is recorded on ``quantizer.last_tensor_ref`` for the exporter.
    """
    ref = scales.detach().float().amax() * factor
    if quantizer.ref_process_group is not None:
        dist.all_reduce(ref, op=dist.ReduceOp.MAX, group=quantizer.ref_process_group)
    if quantizer.fused_ref is not None:
        ref = quantizer.fused_ref.update(quantizer, ref)
    ref = ct_divisor_snap(ref)
    quantizer.last_tensor_ref = ref
    return ref
