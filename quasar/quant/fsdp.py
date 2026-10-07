"""Shard-local QUASAR under FSDP2: search on each rank's rows before the all-gather.

FSDP2 shards weights by rows while quantization groups partition columns, so a
rank's rows hold whole groups and the per-group search needs no communication.
A tensor subclass hooks ``fsdp_pre_all_gather`` to fake-quantize the local shard
(under ``no_grad``), and FSDP2 all-gathers the result: each rank runs 1/world of
the search instead of every rank repeating it on the full weight. The backward
is an identity straight-through estimator: the gathered weight's gradient is
reduce-scattered onto the raw shard.

Every rank must hold the same number of rows (out_features divisible by the world
size): FSDP2 pads uneven shards, and a shard quantized before the all-gather cannot
be padded.
"""

from __future__ import annotations

import torch
import torch.utils._pytree as pytree
from torch import nn

from .quantizer import FakeQuantizer, quant_linears

_aten = torch.ops.aten
# Ops whose outputs must stay wrapped so FSDP2's sharding and copies keep the subclass.
_PRESERVE_OPS = {
    _aten.empty_like.default,
    _aten.new_zeros.default,
    _aten.slice.Tensor,
    _aten.copy_.default,
    _aten.view.default,
    _aten.as_strided.default,
    _aten._to_copy.default,
    _aten._pin_memory.default,
    _aten.split.Tensor,
    _aten.clone.default,
}


class ShardLocalWeight(torch.Tensor):
    """A weight that FSDP2 all-gathers in fake-quantized form.

    Every other op unwraps to the underlying tensor, so the optimizer updates
    the high-precision weight in place.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, qz: FakeQuantizer):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, qz: FakeQuantizer):
        self._tensor = tensor
        self._qz = qz

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == _aten.detach.default:
            return cls(args[0]._tensor, args[0]._qz)
        qz = next(t._qz for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, cls))
        args, kwargs = pytree.tree_map_only(cls, lambda t: t._tensor, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _PRESERVE_OPS:
            return out
        return pytree.tree_map_only(torch.Tensor, lambda t: cls(t, qz), out)

    def __tensor_flatten__(self):
        return ["_tensor"], {"qz": self._qz}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return ShardLocalWeight(inner_tensors["_tensor"], flatten_spec["qz"])

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        """Fake-quantize the local shard in the compute dtype, as the module forward would.

        The mesh's process group is stamped on the quantizer so an NVFP4
        per-tensor reference is reduced over all ranks.
        """
        param_dtype = mp_policy.param_dtype or self._tensor.dtype
        self._qz.ref_process_group = mesh.get_group()
        with torch.no_grad():
            w = self._qz(self._tensor.to(param_dtype))
        return (w.contiguous(),), ()

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        # The gather is already in param_dtype, so the unsharded weight is the gather
        # buffer itself; later gathers (``out`` given) refill it in place.
        if out is not None:
            return
        (w,) = all_gather_outputs
        return w, (w,)


def enable_shard_local(model: nn.Module) -> None:
    """Wrap every QUASAR weight for shard-local search; call after quantize_model, before fully_shard."""
    wrapped = 0
    for _, m in quant_linears(model):
        if m.quantizer.config.method == "quasar":
            m.weight = nn.Parameter(ShardLocalWeight(m.weight.data, m.quantizer), requires_grad=m.weight.requires_grad)
            m.quantizer.bypass = True
            wrapped += 1
    if not wrapped:
        raise RuntimeError("enable_shard_local: the model has no QUASAR quantizers")
