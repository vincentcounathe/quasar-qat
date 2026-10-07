"""The fake quantizer, the linear layer that applies it, and model-level wiring."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from .baselines import bitdistiller, denoising, lsq, lsq_init, standard, standard_nvfp4
from .common import to_blocks
from .config import DECODER_PROJ_RE, QuantConfig
from .nvfp4 import FUSED_GROUPS, FusedRef
from .quasar import quasar, saliency_weights


class FakeQuantizer(nn.Module):
    """``forward(w)`` returns the fake-quantized weight in ``w.dtype``.

    State used by training and export:

    * ``bypass``: the shard-local FSDP path already quantized the weight during
      the all-gather, so :class:`QuantLinear` uses it as is.
    * ``lsq_scale`` / ``lsq_beta``: fp32 LSQ parameters, ``[out, in / g, 1]``.
    * ``saliency`` and ``saliency_steps``: AdamW's ``exp_avg_sq`` of this weight (or
      its local shard) as of the last optimizer step, and how many steps that was.
    * ``last_tensor_ref`` (NVFP4): the per-tensor reference of the last forward;
      the deployed per-tensor scale is ``last_tensor_ref / 448``.
    * ``ref_process_group`` / ``fused_ref`` (NVFP4): make that reference a max
      over ranks / over the partitions of a fused layer (:mod:`quasar.quant.nvfp4`).
    """

    def __init__(self, config: QuantConfig):
        super().__init__()
        self.config = config
        self.bypass = False
        self.saliency: torch.Tensor | None = None
        self.saliency_steps = 0
        self.last_tensor_ref: torch.Tensor | None = None
        self.ref_process_group = None
        self.fused_ref: FusedRef | None = None
        self._search_cache = None

    def extra_repr(self) -> str:
        c = self.config
        return f"method={c.method}, format={c.format}, bits={c.bits}, group_size={c.group_size}"

    @torch.no_grad()
    def init_lsq(self, weight: torch.Tensor) -> None:
        """Create the LSQ step and offset (RTN init); call before FSDP and the optimizer."""
        step, offset = lsq_init(to_blocks(weight.detach(), self.config.group_size), self.config.qmax)
        self.lsq_scale, self.lsq_beta = nn.Parameter(step), nn.Parameter(offset)

    def update_saliency(self, exp_avg_sq: torch.Tensor) -> None:
        """Point at AdamW's second moment after an optimizer step (no bias correction: it cancels per group)."""
        self.saliency = exp_avg_sq
        self.saliency_steps += 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Fake-quantize ``w``."""
        cfg = self.config
        x = to_blocks(w, cfg.group_size)
        if cfg.method == "quasar":
            # Shard-local training gathers see the same weights until the next optimizer step.
            r = quasar(x, saliency_weights(self, w), self, reuse=self.training and self.bypass).reshape(w.shape)
            # Identity straight-through: value r, gradient passed to w unchanged.
            return r + (w - w.detach()) if torch.is_grad_enabled() and w.requires_grad else r
        if cfg.method == "standard":
            r = standard_nvfp4(x, self) if cfg.format == "nvfp4" else standard(x, cfg.qmax)
        elif cfg.method == "denoising":
            r = denoising(x, cfg.qmax)
        elif cfg.method == "lsq":
            r = lsq(x, self.lsq_scale, self.lsq_beta, cfg.qmax)
        else:
            r = bitdistiller(x, cfg.qmax)
        return r.reshape(w.shape)


class QuantLinear(nn.Linear):
    """``nn.Linear`` whose weight passes through ``self.quantizer`` in forward."""

    def __init__(self, linear: nn.Linear, config: QuantConfig):
        super().__init__(linear.in_features, linear.out_features, linear.bias is not None, device="meta")
        # Keep the original Parameter objects: ties, the optimizer and FSDP see the originals.
        self.weight, self.bias = linear.weight, linear.bias
        self.quantizer = FakeQuantizer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight if self.quantizer.bypass else self.quantizer(self.weight)
        return F.linear(x, w, self.bias)


def quant_linears(model: nn.Module) -> Iterator[tuple[str, QuantLinear]]:
    return ((n, m) for n, m in model.named_modules() if isinstance(m, QuantLinear))


def quantize_model(model: nn.Module, config: QuantConfig) -> list[str]:
    """Replace every decoder projection (``DECODER_PROJ_RE``) with a :class:`QuantLinear`.

    LSQ parameters are created here, so call this before FSDP wrapping and
    before building the optimizer. NVFP4 partitions that vLLM fuses into one
    layer (q/k/v, gate/up) share one per-tensor reference. Returns the replaced
    module names.
    """
    names = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not isinstance(m, QuantLinear) and DECODER_PROJ_RE.fullmatch(n)
    ]
    if not names:
        raise RuntimeError("quantize_model: no unquantized decoder projections match DECODER_PROJ_RE")
    fused: dict[tuple[str, tuple[str, ...]], FusedRef] = {}
    for name in names:
        parent, _, child = name.rpartition(".")
        q = QuantLinear(model.get_submodule(name), config)
        if config.method == "lsq":
            q.quantizer.init_lsq(q.weight)
        if config.format == "nvfp4":
            for members in FUSED_GROUPS:
                suffix = next((s for s in members if name.endswith("." + s)), None)
                if suffix:
                    q.quantizer.fused_ref = fused.setdefault((name[: -len(suffix)], members), FusedRef())
        setattr(model.get_submodule(parent), child, q)
    return names


@torch.no_grad()
def snapshot_saliency(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Hand AdamW's ``exp_avg_sq`` to every QUASAR quantizer; call after ``optimizer.step()``.

    A shard-local quantizer (:func:`enable_shard_local`) searches its rank's rows, so it
    takes the local shard of a sharded state; otherwise the module forward quantizes the
    full weight, so the state is gathered.
    """
    for _, m in quant_linears(model):
        if m.quantizer.config.method != "quasar":
            continue
        v = optimizer.state[m.weight]["exp_avg_sq"]
        if isinstance(v, DTensor):
            v = v.to_local() if m.quantizer.bypass else v.full_tensor()
        m.quantizer.update_saliency(v)
