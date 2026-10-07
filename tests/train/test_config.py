"""Recipe + CLI merging and validation."""

from pathlib import Path

import pytest

from quasar.quant import QuantConfig
from quasar.train.config import TrainConfig, load_config

REQUIRED = ["--model_path", "m", "--train_data", "t.jsonl", "--eval_data", "e.jsonl", "--out_dir", "out"]
RECIPES = sorted((Path(__file__).parents[2] / "configs").glob("*.yaml"))


def _recipe(tmp_path, text):
    path = tmp_path / "recipe.yaml"
    path.write_text(text)
    return ["--recipe", str(path)]


def test_recipe_then_cli_override(tmp_path):
    recipe = _recipe(
        tmp_path,
        """
model_path: Qwen/Qwen3-4B-Thinking-2507
train_data: train.jsonl
eval_data: eval.jsonl
out_dir: runs/x
learning_rate: {2: 5e-5, 3: 3e-5, 4: 2e-5}
max_seq_len: 4096
freeze: true
""",
    )
    cfg = load_config(recipe + ["--bits", "3", "--method", "lsq", "--out_dir", "2026"])
    assert (cfg.bits, cfg.method, cfg.learning_rate, cfg.out_dir) == (3, "lsq", 3e-5, "2026")
    assert cfg.model_path == "Qwen/Qwen3-4B-Thinking-2507" and cfg.freeze is True
    cfg = load_config(recipe + ["--bits", "2", "--learning_rate", "1e-4", "--freeze", "false"])
    assert cfg.learning_rate == 1e-4 and isinstance(cfg.learning_rate, float) and cfg.freeze is False


def test_values_parse_as_yaml_and_strings_stay_verbatim():
    cfg = load_config(
        REQUIRED + ["--out_dir", "0755", "--model_path", "1e3", "--warmup_ratio", "3e-2", "--format", "nvfp4"]
    )
    assert cfg.warmup_ratio == 0.03 and (cfg.out_dir, cfg.model_path) == ("0755", "1e3")
    assert cfg.quant_config() == QuantConfig("quasar", "nvfp4", 4)


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_shipped_recipes_validate(recipe):
    """Every cell the reproduction scripts launch resolves to a valid config."""
    io = ["--train_data", "t", "--eval_data", "e", "--out_dir", "o"]
    nvfp4 = "nvfp4" in recipe.stem
    methods = ["quasar", "standard"] if nvfp4 else ["quasar", "standard", "lsq", "denoising", "bitdistiller"]
    for method in methods:
        for bits in [4] if nvfp4 else [2, 3, 4]:
            cfg = load_config(["--recipe", str(recipe), "--method", method, "--bits", str(bits)] + io)
            assert isinstance(cfg.learning_rate, float)
    if "adapt" in recipe.stem:  # the full-precision reference
        assert load_config(["--recipe", str(recipe), "--method", "none"] + io).quant_config() is None


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--method", "gptq"], "method"),
        (["--objective", "mse"], "objective"),
        (["--method", "none"], "freeze"),
        (["--format", "nvfp4", "--method", "lsq"], "nvfp4"),
        (["--bits", "8"], "bits"),
        (["--grad_accum", "0"], "grad_accum"),
        (["--warmup_ratio", "1.5"], "warmup_ratio"),
    ],
)
def test_validation_rejects(extra, match):
    with pytest.raises(ValueError, match=match):
        load_config(REQUIRED + extra)


def test_missing_and_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="eval_data"):
        TrainConfig(model_path="m", train_data="t", out_dir="o").validate()
    with pytest.raises(ValueError, match="no entry for bits=2"):
        load_config(REQUIRED + ["--bits", "2", "--learning_rate", "{4: 1e-5}"])
    with pytest.raises(ValueError, match="unknown recipe keys"):
        load_config(_recipe(tmp_path, "lr: 1e-5\n") + REQUIRED)
    with pytest.raises(ValueError, match="quote it"):
        load_config(_recipe(tmp_path, "tokenizer_path: 2507\n") + REQUIRED)


def test_none_method_trains_full_precision():
    cfg = load_config(REQUIRED + ["--method", "none", "--freeze", "false", "--objective", "ce"])
    assert cfg.quant_config() is None
