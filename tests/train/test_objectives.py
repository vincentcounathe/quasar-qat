"""KD loss and held-out metrics against per-token reference math."""

import math

import pytest
import torch
import torch.nn.functional as F

from quasar.train.objectives import heldout_metrics, heldout_sums, kd_loss


def _batch(dtype=torch.float32, b=3, t=29, v=53, seed=0):
    g = torch.Generator().manual_seed(seed)
    student = (torch.randn(b, t, v, generator=g) * 2).to(dtype)
    teacher = (student.float() + torch.randn(b, t, v, generator=g)).to(dtype)
    labels = torch.randint(0, v, (b, t), generator=g)
    labels[:, :5] = -100  # prompt
    labels[-1, t // 2 :] = -100  # padding
    return student, teacher, labels


def _per_token(student, teacher, labels):
    """Reference: per supervised position, KL(p_teacher || p_student), NLL and argmax agreement."""
    kls, nlls, agree = [], [], []
    for i in range(student.shape[0]):
        for j in range(student.shape[1] - 1):
            y = int(labels[i, j + 1])
            if y == -100:
                continue
            p, q = F.softmax(teacher[i, j].double(), -1), F.softmax(student[i, j].double(), -1)
            kls.append(float((p * (p.log() - q.log())).sum()))
            nlls.append(-math.log(float(q[y])))
            agree.append(int(student[i, j].argmax() == teacher[i, j].argmax()))
    return kls, nlls, agree


@pytest.mark.parametrize("chunk", [4, 7, 1024])
def test_kd_loss_is_token_mean_forward_kl(chunk):
    student, teacher, labels = _batch()
    kls, _, _ = _per_token(student, teacher, labels)
    assert float(kd_loss(student, teacher, labels, chunk=chunk)) == pytest.approx(sum(kls) / len(kls), rel=1e-5)


def test_kd_loss_gradient_is_softmax_difference():
    student, teacher, labels = _batch()
    s = student.clone().requires_grad_()
    kd_loss(s, teacher, labels, chunk=5).backward()
    mask = labels[:, 1:].ne(-100)
    expected = torch.zeros_like(student)
    expected[:, :-1] = (F.softmax(student[:, :-1], -1) - F.softmax(teacher[:, :-1], -1)) * mask[..., None] / mask.sum()
    torch.testing.assert_close(s.grad, expected, rtol=1e-4, atol=1e-7)


def test_kd_loss_bf16_logits_are_upcast():
    student, teacher, labels = _batch(torch.bfloat16)
    kls, _, _ = _per_token(student.float(), teacher.float(), labels)
    loss = kd_loss(student, teacher, labels)
    assert loss.dtype == torch.float32 and float(loss) == pytest.approx(sum(kls) / len(kls), rel=1e-5)


@pytest.mark.parametrize("chunk", [6, 1024])
def test_heldout_sums_and_metrics(chunk):
    student, teacher, labels = _batch(seed=1)
    kls, nlls, agree = _per_token(student, teacher, labels)
    sums = heldout_sums(student, labels, teacher, chunk=chunk)
    assert sums.dtype == torch.float64 and int(sums[0]) == len(nlls) and int(sums[3]) == sum(agree)
    assert float(sums[1]) == pytest.approx(sum(nlls), rel=1e-5) and float(sums[2]) == pytest.approx(sum(kls), rel=1e-5)
    m = heldout_metrics(sums, with_teacher=True)
    assert m["eval_ce"] == pytest.approx(sum(nlls) / len(nlls), rel=1e-5)
    assert m["eval_ppl"] == pytest.approx(math.exp(m["eval_ce"]))
    assert m["eval_kl"] == pytest.approx(sum(kls) / len(kls), rel=1e-5) and m["eval_top1"] == sum(agree) / len(agree)
    # CE matches the causal-LM loss the CE objective trains on.
    ce = F.cross_entropy(student[:, :-1].reshape(-1, student.shape[-1]), labels[:, 1:].reshape(-1))
    assert m["eval_ce"] == pytest.approx(float(ce), rel=1e-5)


def test_heldout_without_teacher_and_empty():
    student, _, labels = _batch()
    assert set(heldout_metrics(heldout_sums(student, labels), with_teacher=False)) == {"eval_ce", "eval_ppl"}
    with pytest.raises(RuntimeError):
        heldout_metrics(torch.zeros(4, dtype=torch.float64), with_teacher=False)
