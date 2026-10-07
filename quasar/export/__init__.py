"""Export trained models: bf16 materialization, the RTN baseline and NVFP4 packing.

* :func:`save_materialized` (in-run, every rank): plain bf16 HF checkpoint of the
  deployed weights; this is the INT artifact and the input of the NVFP4 packer.
* ``python -m quasar.export.rtn``: untrained Standard QAT (round-to-nearest).
* ``python -m quasar.export.nvfp4``: lossless compressed-tensors NVFP4 (W4A16) packing.
"""

from .materialize import save_materialized

__all__ = ["save_materialized"]
