"""Single-process pieces of the trainer: freezing, the optimizer schedule and the objective switch."""

import pytest
import torch
from helpers import tiny_model

from quasar.quant import QuantConfig, quant_linears, quantize_model
from quasar.train.config import TrainConfig
from quasar.train.objectives import kd_loss
from quasar.train.trainer import freeze_to_quantized, make_optimizer, objective


@pytest.mark.parametrize("method", ["quasar", "lsq"])
def test_freeze_keeps_only_quantized_projections(method):
    model = tiny_model()
    quantize_model(model, QuantConfig(method, bits=2))
    freeze_to_quantized(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {f"{n}.weight" for n, _ in quant_linears(model)}
    if method == "lsq":
        expected |= {f"{n}.quantizer.lsq_{k}" for n, _ in quant_linears(model) for k in ("scale", "beta")}
    assert trainable == expected and len(list(quant_linears(model))) == 14
    assert not model.model.embed_tokens.weight.requires_grad and model.lm_head.weight is model.model.embed_tokens.weight


def test_frozen_model_still_gets_gradients_with_checkpointing():
    model = tiny_model()
    model.gradient_checkpointing_enable()
    quantize_model(model, QuantConfig("standard", bits=4))
    freeze_to_quantized(model)
    model.train()
    ids = torch.randint(0, 259, (2, 16))
    model(input_ids=ids, labels=ids).loss.backward()
    assert all(m.weight.grad is not None and m.weight.grad.abs().sum() > 0 for _, m in quant_linears(model))


def test_schedule_warmup_then_cosine_to_zero():
    cfg = TrainConfig(learning_rate=2e-5, max_steps=4096, warmup_ratio=0.01)
    p = torch.nn.Parameter(torch.zeros(3))
    opt, sched = make_optimizer([p], cfg)
    assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["weight_decay"] == 0.0
    lrs = []
    for _ in range(cfg.max_steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.0 and lrs[1] == pytest.approx(2e-5 / 41) and lrs[41] == pytest.approx(2e-5)
    assert all(a >= b for a, b in zip(lrs[41:-1], lrs[42:], strict=True)) and lrs[-1] < 1e-10


def test_objective_kd_with_teacher_ce_without():
    torch.manual_seed(0)
    student, teacher = tiny_model(seed=0).float(), tiny_model(seed=1).float()
    ids = torch.randint(0, 259, (2, 12))
    labels = ids.clone()
    labels[:, :4] = -100
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    ce = objective(student, None, batch)
    assert ce.item() == pytest.approx(student(**batch).loss.item())
    kd = objective(student, teacher, batch)
    ref = kd_loss(student(input_ids=ids).logits, teacher(input_ids=ids).logits, labels)
    assert kd.item() == pytest.approx(ref.item(), rel=1e-6) and kd.requires_grad
