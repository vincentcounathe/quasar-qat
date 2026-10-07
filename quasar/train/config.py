"""Training configuration: one dataclass, filled from a recipe YAML and CLI overrides.

    torchrun --nproc_per_node 8 -m quasar.train --recipe configs/heal_qwen3_4b_thinking.yaml --method quasar \\
        --bits 2 --train_data train.jsonl --eval_data eval.jsonl --out_dir runs/w2_quasar --learning_rate 5e-5

Every field can be set in the recipe or on the command line (CLI wins). Non-string
values are parsed as YAML, so ``--freeze false`` and ``--learning_rate "{2: 5e-5}"`` work.
``learning_rate`` may be a ``{bits: lr}`` mapping in the recipe; it is resolved
against ``bits`` after the overrides are applied. Fields and defaults: ``TrainConfig``
in ``quasar/train/config.py``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass, fields

import yaml

from quasar.quant import QuantConfig
from quasar.quant.config import METHODS as QUANT_METHODS

METHODS = (*QUANT_METHODS, "none")


@dataclass
class TrainConfig:
    # Model and data. Corpora are JSONL files of {"messages": [...]} rows.
    model_path: str = ""  # also the KD teacher (self-distillation from the BF16 model)
    tokenizer_path: str | None = None  # chat template + tokenizer; default model_path
    train_data: str = ""
    eval_data: str = ""
    out_dir: str = ""
    max_seq_len: int = 4096
    # Quantization ("none" trains the full-precision model).
    method: str = "quasar"
    format: str = "int"
    bits: int = 4  # group sizes and QUASAR's clipping grid are fixed per format (quasar.quant.config)
    # Objective: "kd" = KL(teacher || student) at T=1; "ce" = next-token cross-entropy.
    objective: str = "kd"
    freeze: bool = True  # train only the quantized projections (healing); False trains everything
    # Optimization: AdamW (0.9, 0.999), no weight decay, cosine schedule to zero.
    learning_rate: float | dict = 2e-5  # or {bits: lr}
    per_device_batch_size: int = 1
    grad_accum: int = 1
    max_steps: int = 4096
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    # fp32 weights and AdamW state (compute stays bf16). The INT recipes' learning rates were
    # tuned with bf16 master weights, whose rounding absorbs part of each small update.
    fp32_master: bool = False
    attn_implementation: str = "sdpa"
    seed: int = 1234
    # Logging and held-out evaluation (eval_steps 0: final evaluation only).
    logging_steps: int = 10
    eval_steps: int = 128

    def quant_config(self) -> QuantConfig | None:
        return None if self.method == "none" else QuantConfig(self.method, self.format, self.bits)

    def validate(self) -> TrainConfig:
        """Resolve ``learning_rate`` against ``bits`` and reject inconsistent settings."""
        for name in ("model_path", "train_data", "eval_data", "out_dir"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        for f in fields(self):  # a recipe value YAML reads as a number or date
            if f.type.startswith("str") and not isinstance(getattr(self, f.name), (str, type(None))):
                raise ValueError(f"{f.name} must be a string; quote it in the recipe")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.objective not in ("kd", "ce"):
            raise ValueError(f"objective must be 'kd' or 'ce', got {self.objective!r}")
        if self.freeze and self.method == "none":
            raise ValueError("freeze=true trains only quantized projections; method 'none' has none")
        self.quant_config()  # raises on an unsupported method/format/bits combination
        if isinstance(self.learning_rate, dict):
            if self.bits not in self.learning_rate:
                raise ValueError(f"learning_rate has no entry for bits={self.bits}: {self.learning_rate}")
            self.learning_rate = self.learning_rate[self.bits]
        self.learning_rate = float(self.learning_rate)
        positive = (
            "learning_rate",
            "max_seq_len",
            "per_device_batch_size",
            "grad_accum",
            "max_steps",
            "max_grad_norm",
            "logging_steps",
        )
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0 <= self.warmup_ratio < 1 or self.eval_steps < 0:
            raise ValueError("need 0 <= warmup_ratio < 1 and eval_steps >= 0")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``3e-5`` as a float (YAML 1.1 requires a dot)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float", re.compile(r"^[-+]?[0-9]+(\.[0-9]*)?[eE][-+]?[0-9]+$"), list("-+0123456789")
)


def parse_value(text: str):
    return yaml.load(text, Loader=_Loader)


def load_config(argv: list[str] | None = None) -> TrainConfig:
    """Recipe YAML (``--recipe``) merged with ``--<field> value`` overrides, validated."""
    ap = argparse.ArgumentParser(
        prog="python -m quasar.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--recipe", help="YAML file with any TrainConfig fields")
    for f in fields(TrainConfig):  # paths and names verbatim, everything else as YAML
        parse = str if f.type.startswith("str") else parse_value
        ap.add_argument(f"--{f.name}", type=parse, default=argparse.SUPPRESS, metavar="VALUE")
    overrides = vars(ap.parse_args(argv))
    values = {}
    if recipe := overrides.pop("recipe", None):
        with open(recipe, encoding="utf-8") as fh:
            values = yaml.load(fh, Loader=_Loader) or {}
    unknown = set(values) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown recipe keys: {sorted(unknown)}")
    values.update(overrides)
    return TrainConfig(**values).validate()
