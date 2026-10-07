"""Round-to-nearest baseline: an untrained Standard QAT model, materialized.

    python -m quasar.export.rtn --model Qwen/Qwen3-4B-Thinking-2507 --format int --bits 2 --out_dir rtn_w2
    python -m quasar.export.rtn --model Qwen/Qwen3-8B --format nvfp4 --out_dir rtn_nvfp4

INT writes the bf16 dequantized checkpoint (min/max affine groups of 128);
NVFP4 additionally writes the tensor references, so the result can be packed
with ``python -m quasar.export.nvfp4``.
"""

from __future__ import annotations

import argparse

import torch

from quasar.quant import QuantConfig, quantize_model

from .materialize import save_materialized


def rtn(model_path: str, out_dir: str, *, fmt: str = "int", bits: int = 4, device: str = "cpu") -> dict:
    from transformers import AutoModelForCausalLM

    config = QuantConfig("standard", fmt, bits)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
    quantize_model(model, config)
    receipt = {
        "method": "rtn",
        "format": fmt,
        "bits": bits,
        "group_size": config.group_size,
        "model_path": str(model_path),
    }
    return save_materialized(model, out_dir, model_path=model_path, receipt=receipt)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or directory")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--format", choices=("int", "nvfp4"), default="int")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    print(rtn(a.model, a.out_dir, fmt=a.format, bits=a.bits, device=a.device))


if __name__ == "__main__":
    main()
