"""Weight-only fake quantization for QAT: QUASAR and the paper's baselines.

Typical use::

    quantize_model(model, QuantConfig(method="quasar", bits=2))  # decoder projections -> QuantLinear
    enable_shard_local(model)                                    # QUASAR under FSDP2, before fully_shard
    ...
    optimizer.step()
    snapshot_saliency(model, optimizer)                          # AdamW exp_avg_sq -> saliency
"""

from .baselines import apply_bitdistiller_clip
from .config import DECODER_PROJ_RE, QuantConfig
from .fsdp import enable_shard_local
from .quantizer import (
    FakeQuantizer,
    QuantLinear,
    quant_linears,
    quantize_model,
    snapshot_saliency,
)

__all__ = [
    "DECODER_PROJ_RE",
    "FakeQuantizer",
    "QuantConfig",
    "QuantLinear",
    "apply_bitdistiller_clip",
    "enable_shard_local",
    "quant_linears",
    "quantize_model",
    "snapshot_saliency",
]
