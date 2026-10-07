"""Triton sweep vs the PyTorch sweep (CUDA only)."""

import pytest
import torch

from quasar.quant import FakeQuantizer, QuantConfig, kernel
from quasar.quant.config import SALIENCY_WARMUP

HAS_GPU = torch.cuda.is_available() and kernel.TRITON_AVAILABLE
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not HAS_GPU, reason="needs CUDA + Triton")]


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_kernel_matches_torch_sweep(bits, dtype, monkeypatch):
    g = torch.Generator().manual_seed(bits)
    w = (torch.randn(512, 1024, generator=g) * 0.02).to(dtype).cuda()
    qz = FakeQuantizer(QuantConfig("quasar", bits=bits))
    for _ in range(SALIENCY_WARMUP):
        qz.update_saliency(torch.exp(torch.randn(w.shape, generator=g) * 3 - 20).cuda())
    with torch.no_grad():
        r_kernel = qz(w)
        monkeypatch.setattr(kernel, "TRITON_AVAILABLE", False)
        r_torch = qz(w)
    h = qz.saliency.reshape(-1, 128)
    err = lambda r: (h * (r.float() - w.float()).reshape(-1, 128) ** 2).sum(-1)  # noqa: E731
    same = (r_kernel.reshape(-1, 128) == r_torch.reshape(-1, 128)).all(-1).float().mean()
    # Groups may differ only where two factors tie up to rounding (in bf16 a near-tie can flip the
    # rounding of a dominant-saliency weight, so a single group's error can move a lot); the total
    # weighted error must stay within 1%.
    assert float(same) >= 0.98
    assert float(err(r_kernel).sum()) <= float(err(r_torch).sum()) * 1.01
