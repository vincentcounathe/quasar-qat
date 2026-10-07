"""Training objective (KD) and held-out metrics, computed on next-token logits.

Logits at position ``t`` predict token ``t + 1``, as in the causal-LM loss, and
only positions whose target is supervised (``labels != -100``) count. Softmaxes
run in fp32, ``chunk`` positions at a time; ``kd_loss`` recomputes each chunk in
backward, so only the bf16 logits stay alive until then.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LOGIT_CHUNK = 1024


def _chunk_kl(s: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kl = F.kl_div(F.log_softmax(s.float(), dim=-1), F.softmax(t.float(), dim=-1), reduction="none").sum(dim=-1)
    return (kl * mask).sum()


def kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, chunk: int = LOGIT_CHUNK
) -> torch.Tensor:
    """Token mean of the forward KL(teacher || student) at temperature 1."""
    s, t = student_logits[:, :-1], teacher_logits[:, :-1]
    mask = labels[:, 1:].ne(-100)
    total = 0.0
    for i in range(0, s.shape[1], chunk):
        c = slice(i, i + chunk)
        total = total + checkpoint(
            _chunk_kl, s[:, c], t[:, c], mask[:, c], use_reentrant=False, preserve_rng_state=False
        )
    return total / mask.sum().clamp_min(1)


@torch.no_grad()
def heldout_sums(
    logits: torch.Tensor, labels: torch.Tensor, teacher_logits: torch.Tensor | None = None, chunk: int = LOGIT_CHUNK
) -> torch.Tensor:
    """float64 ``[tokens, sum NLL, sum KL(teacher || student), top-1 agreements]`` of one batch."""
    s, targets = logits[:, :-1], labels[:, 1:]
    tl = teacher_logits[:, :-1] if teacher_logits is not None else None
    mask = targets.ne(-100)
    sums = torch.zeros(4, dtype=torch.float64, device=logits.device)
    sums[0] = mask.sum()
    for i in range(0, s.shape[1], chunk):
        m = mask[:, i : i + chunk]
        log_q = F.log_softmax(s[:, i : i + chunk].float(), dim=-1)
        nll = -log_q.gather(-1, targets[:, i : i + chunk].clamp_min(0).unsqueeze(-1)).squeeze(-1)
        sums[1] += nll[m].sum()
        if tl is not None:
            t = tl[:, i : i + chunk].float()
            sums[2] += F.kl_div(log_q, F.softmax(t, dim=-1), reduction="none").sum(dim=-1)[m].sum()
            sums[3] += (log_q.argmax(dim=-1) == t.argmax(dim=-1))[m].sum()
    return sums


def heldout_metrics(sums: torch.Tensor, with_teacher: bool) -> dict[str, float]:
    """Token-weighted means of :func:`heldout_sums` totals; PPL is ``exp(CE)``."""
    n = float(sums[0])
    if n == 0:
        raise RuntimeError("held-out set has no supervised tokens")
    ce = float(sums[1]) / n
    out = {"eval_ce": ce, "eval_ppl": math.exp(ce) if ce < 50 else math.inf}
    if with_teacher:
        out.update(eval_kl=float(sums[2]) / n, eval_top1=float(sums[3]) / n)
    return out
