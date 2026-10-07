"""Quantizer configuration and the method's fixed constants."""

from __future__ import annotations

import re
from dataclasses import dataclass

METHODS = ("standard", "lsq", "denoising", "bitdistiller", "quasar")
FORMATS = ("int", "nvfp4")

EPS = 1e-8  # guards divisions in the code maps and least-squares fits
# Saliency is AdamW's exp_avg_sq; the floor only keeps all-zero groups finite.
SALIENCY_FLOOR = 1e-30
# exp_avg_sq starts at zero and is dominated by the first few gradients, so the
# first snapshots are ignored (uniform saliency) until it has averaged a little.
SALIENCY_WARMUP = 10

# QUASAR clipping factors. INT shrinks the min/max range; for NVFP4 the largest
# factor also sets the per-tensor scale, so it changes the deployed lattice.
INT_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
NVFP4_GRID = (0.8, 0.9, 1.0, 1.25, 1.5)
NVFP4_GROUP = 16  # fixed by the format
INT_GROUP = 128

# Decoder-block projections that get quantized; lm_head, embeddings and norms stay BF16.
DECODER_PROJ_RE = re.compile(r"^model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj$")


@dataclass(frozen=True)
class QuantConfig:
    """Weight-only fake quantization of one model.

    ``format="int"``: asymmetric (affine) quantization with unsigned codes
    ``0..2^bits-1`` and a continuous offset per group of 128.
    ``format="nvfp4"``: E2M1 codes in groups of 16 with E4M3 group scales under
    one FP32 per-tensor scale (methods ``standard`` and ``quasar`` only).
    """

    method: str
    format: str = "int"
    bits: int = 4

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {self.format!r}")
        if self.format == "nvfp4" and (self.method not in ("standard", "quasar") or self.bits != 4):
            raise ValueError(f"nvfp4 supports 4-bit standard/quasar only, got {self.method!r} W{self.bits}")
        if self.format == "int" and self.bits not in (2, 3, 4):
            raise ValueError(f"int format supports bits 2/3/4, got {self.bits}")

    @property
    def group_size(self) -> int:
        return NVFP4_GROUP if self.format == "nvfp4" else INT_GROUP

    @property
    def grid(self) -> tuple[float, ...]:
        """QUASAR's clipping factors."""
        return NVFP4_GRID if self.format == "nvfp4" else INT_GRID

    @property
    def qmax(self) -> int:
        """Largest INT code (codes are ``0..qmax``)."""
        return 2**self.bits - 1
