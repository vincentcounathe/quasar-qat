"""Single-process materialization: the saved checkpoint computes what the quantized model computes."""

import json

import pytest
import torch
from helpers import save_base, tiny_model
from transformers import AutoModelForCausalLM

from quasar.export import save_materialized
from quasar.export.writer import SafetensorsReader
from quasar.quant import FakeQuantizer, QuantConfig, quant_linears, quantize_model
from quasar.quant.config import SALIENCY_WARMUP

INT_METHODS = ["standard", "lsq", "denoising", "bitdistiller", "quasar"]


def _train_state(model, method):
    """Stand-ins for training: a live saliency snapshot (QUASAR), moved LSQ parameters."""
    g = torch.Generator().manual_seed(1)
    for _, m in quant_linears(model):
        qz = m.quantizer
        if method == "quasar":
            for _ in range(SALIENCY_WARMUP):
                qz.update_saliency(torch.exp(torch.randn(m.weight.shape, generator=g) * 3 - 20))
        if method == "lsq":
            with torch.no_grad():
                qz.lsq_scale.mul_(1 + 0.1 * torch.rand(qz.lsq_scale.shape, generator=g))
                qz.lsq_beta.add_(0.01 * torch.randn(qz.lsq_beta.shape, generator=g))


def _materialize(model, tmp_path, base):
    out = tmp_path / "materialized"
    save_materialized(model, out, model_path=base, receipt={"method": "test"})
    return out


@pytest.mark.parametrize("method", INT_METHODS + ["none"])
def test_reloaded_checkpoint_matches_quantized_model(family, method, tmp_path):
    model = tiny_model(family)
    base = save_base(model, tmp_path / "base")
    if method != "none":
        quantize_model(model, QuantConfig(method, bits=2))
        _train_state(model, method)
    ids = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    out = _materialize(model, tmp_path, base)

    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).eval()
    with torch.no_grad():
        assert torch.equal(reloaded(input_ids=ids).logits, expected)
    keys = SafetensorsReader(out).keys()
    assert ("lm_head.weight" in keys) == (family == "llama")  # tied embeddings are written once
    assert not any(".quantizer." in k for k in keys)
    assert json.loads((out / "receipt.json").read_text()) == {
        "method": "test",
        "quantized_linears": 0 if method == "none" else 14,
    }
    for name in ["config.json", "generation_config.json", "tokenizer.json", "chat_template.jinja"]:
        assert (out / name).read_bytes() == (base / name).read_bytes()


def test_quasar_uses_the_live_saliency(tmp_path):
    model = tiny_model("llama")
    base = save_base(model, tmp_path / "base")
    quantize_model(model, QuantConfig("quasar", bits=2))
    _train_state(model, "quasar")
    reader = SafetensorsReader(_materialize(model, tmp_path, base))
    differs = 0
    for name, m in quant_linears(model):
        saved = reader[f"{name}.weight"]
        with torch.no_grad():
            assert torch.equal(saved, m.quantizer(m.weight))
            differs += int(not torch.equal(saved, FakeQuantizer(m.quantizer.config)(m.weight)))  # uniform saliency
    assert differs > 0


@pytest.mark.parametrize("method", ["standard", "quasar"])
def test_nvfp4_refs_are_final_and_shared(method, tmp_path):
    model = tiny_model("qwen3")
    base = save_base(model, tmp_path / "base")
    quantize_model(model, QuantConfig(method, "nvfp4"))
    _train_state(model, method)
    with torch.no_grad():
        model(input_ids=torch.zeros(1, 4, dtype=torch.long))  # references of these weights...
        for p in model.parameters():
            p.mul_(1.5)  # ...are stale after this "optimizer step"
    out = _materialize(model, tmp_path, base)
    refs = json.loads((out / "nvfp4_tensor_refs.json").read_text())
    reader = SafetensorsReader(out)
    assert len(refs) == 14
    for name, m in quant_linears(model):
        assert refs[f"{name}.weight"] == float(m.quantizer.last_tensor_ref).hex()
        with torch.no_grad():
            assert torch.equal(reader[f"{name}.weight"], m.quantizer(m.weight))
    for layer in range(2):
        p = f"model.layers.{layer}."
        assert len({refs[p + f"self_attn.{x}_proj.weight"] for x in "qkv"}) == 1
        assert len({refs[p + f"mlp.{x}_proj.weight"] for x in ("gate", "up")}) == 1


def test_untied_lm_head_under_tied_config_is_refused(tmp_path):
    model = tiny_model("qwen3")
    base = save_base(model, tmp_path / "base")
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(RuntimeError, match="tie_word_embeddings"):
        _materialize(model, tmp_path, base)
