"""Pack a materialized NVFP4 run into a compressed-tensors checkpoint (W4A16).

    python -m quasar.export.nvfp4 --materialized RUN/materialized --out_dir RUN/nvfp4

The output uses the ``nvfp4-pack-quantized`` format that vLLM loads natively,
weight-only (Hopper serves it through the Marlin kernel). Per quantized linear:

* ``weight_packed`` uint8 ``[out, in/2]``: E2M1 codes, low nibble = even element, bit 3 = sign;
* ``weight_scale`` float8_e4m3fn ``[out, in/16]``: group scales in units of ``S``;
* ``weight_global_scale`` fp32: the divisor ``1/S`` (loaders compute ``S = 1/divisor``).

The export is lossless. Each group's codes and scale are recovered by replaying
the trainer's arithmetic, ``bf16(fp32(S * scale) * code)``, which must
reproduce every materialized weight bit for bit; the written tensors are then
decoded from disk and compared again. ``S`` cannot be recovered from the
weights, so it is read from the run's ``nvfp4_tensor_refs.json`` (``S = ref / 448``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from quasar.quant import DECODER_PROJ_RE
from quasar.quant.common import to_blocks
from quasar.quant.config import NVFP4_GROUP
from quasar.quant.nvfp4 import E2M1_LEVELS, E4M3_MAX, E4M3_MIN, FUSED_GROUPS, project_e2m1

from .files import copy_model_files
from .materialize import REFS_FILE
from .writer import SafetensorsReader, ShardedSafetensorsWriter

E2M1_CODES = E2M1_LEVELS + tuple(-v for v in E2M1_LEVELS)  # value of each 4-bit code


def _recover(w: torch.Tensor, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """4-bit codes ``[rows, in]`` and E4M3 group scales ``[rows, in/16]`` of a bf16 weight block.

    A group's largest weight is ``S * scale * level`` for some E2M1 level; each
    level's implied scale (and its E4M3 neighbours, which bf16 rounding can
    select) is replayed, and the first that reproduces all 16 weights exactly
    is kept. A negative fitted scale folds into the codes' signs.
    """
    x = to_blocks(w, NVFP4_GROUP)
    xf = x.float()
    amax = xf.abs().amax(dim=-1)
    done = amax == 0  # all-zero groups: code 0, scale 0
    codes = torch.zeros(x.shape, dtype=torch.uint8)
    scales = torch.zeros(amax.shape, dtype=torch.uint8)  # float8_e4m3fn bytes
    levels = torch.tensor(E2M1_LEVELS)
    for level in reversed(E2M1_LEVELS[1:]):
        guess = (amax / level / S).clamp(E4M3_MIN, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
        for step in (0, 1, -1):
            if bool(done.all()):
                break
            scale = (guess.to(torch.int16) + step).clamp(0, 0x7E).to(torch.uint8)
            a = (S * scale.view(torch.float8_e4m3fn).float()).unsqueeze(-1)
            safe = torch.where(a > 0, a, torch.ones_like(a))
            q = project_e2m1(xf / safe)
            take = ~done & (a[..., 0] > 0) & ((safe * q).to(torch.bfloat16) == x).all(dim=-1)
            code = (torch.searchsorted(levels, q.abs()) + 8 * (q < 0)).to(torch.uint8)
            codes = torch.where(take.unsqueeze(-1), code, codes)
            scales = torch.where(take, scale, scales)
            done |= take
    if not bool(done.all()):
        bad = (~done).nonzero()[:4].tolist()
        raise ValueError(
            f"{int((~done).sum())} groups are not on the NVFP4 lattice of S={float(S):.6e} (first [row, group]: {bad})"
        )
    return codes.reshape(w.shape), scales.view(torch.float8_e4m3fn)


def pack(w: torch.Tensor, S: torch.Tensor, rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weight_packed, weight_scale)`` of a bf16 ``[out, in]`` weight on the lattice of ``S``."""
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise ValueError(f"expected a 2-D bf16 weight, got {w.dtype} {tuple(w.shape)}")
    packed = torch.empty(w.shape[0], w.shape[1] // 2, dtype=torch.uint8)
    scales = torch.empty(w.shape[0], w.shape[1] // NVFP4_GROUP, dtype=torch.float8_e4m3fn)
    for r in range(0, w.shape[0], rows):  # bounds the fp32 temporaries
        codes, block_scales = _recover(w[r : r + rows], S)
        packed[r : r + rows] = (codes[:, 1::2] << 4) | codes[:, 0::2]
        scales[r : r + rows] = block_scales
    return packed, scales


def dequantize(packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """bf16 weight of one packed linear, computed as vLLM does at load."""
    lut = torch.tensor(E2M1_CODES, dtype=torch.float32)
    q = torch.stack([lut[(packed & 15).long()], lut[(packed >> 4).long()]], dim=-1).reshape(packed.shape[0], -1)
    S = torch.ones((), dtype=torch.float32) / global_scale.float()
    return ((scale.float() * S).repeat_interleave(NVFP4_GROUP, dim=-1) * q).to(torch.bfloat16)


def global_scale(S: torch.Tensor) -> torch.Tensor:
    """The stored divisor ``1/S``; refuses an ``S`` the loader's ``1/divisor`` would not give back."""
    one = torch.ones((), dtype=torch.float32)
    d = one / S
    if not torch.equal(one / d, S):
        raise ValueError(f"tensor scale {float(S)!r} does not survive the 1/S round trip")
    return d.reshape(1)


def _check_fused(refs: dict[str, str]) -> None:
    """vLLM keeps one tensor scale per fused q/k/v and gate/up layer, so partners must agree."""
    for members in FUSED_GROUPS:
        for key in refs:
            if key.endswith(f".{members[0]}.weight"):
                prefix = key[: -len(f"{members[0]}.weight")]
                if len({refs.get(f"{prefix}{m}.weight") for m in members}) != 1:
                    raise ValueError(f"tensor scales differ within the fused layer {prefix}{members}")


def quantization_config(ignore: list[str]) -> dict:
    """compressed-tensors ``quantization_config`` for NVFP4 weights and unquantized activations."""
    weights = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": NVFP4_GROUP,
        "dynamic": False,
        "block_structure": None,
        "actorder": None,
        "observer": "minmax",
        "observer_kwargs": {},
    }
    group = {
        "targets": ["Linear"],
        "weights": weights,
        "input_activations": None,
        "output_activations": None,
        "format": "nvfp4-pack-quantized",
    }
    return {
        "config_groups": {"group_0": group},
        "format": "nvfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "ignore": ignore,
        "kv_cache_scheme": None,
        "global_compression_ratio": None,
        "version": "0.17.0",
    }


def export(materialized: str | Path, out: str | Path) -> dict:
    """Pack every decoder projection of a materialized NVFP4 run into ``out``."""
    src = Path(materialized)
    refs = json.loads((src / REFS_FILE).read_text())
    reader = SafetensorsReader(src)
    scope = {k for k in reader.keys() if k.endswith(".weight") and DECODER_PROJ_RE.fullmatch(k[: -len(".weight")])}
    if set(refs) != scope:
        raise ValueError(f"{REFS_FILE} does not cover exactly the decoder projections: {sorted(scope ^ set(refs))[:4]}")
    _check_fused(refs)
    S = {k: torch.tensor(float.fromhex(v) / E4M3_MAX, dtype=torch.float32) for k, v in refs.items()}
    divisors = {k: global_scale(s) for k, s in S.items()}

    writer = ShardedSafetensorsWriter(out)
    ignore = {"lm_head"}
    for name in reader.keys():
        t = reader[name]
        module = name[: -len(".weight")]
        if name in S:
            packed, scale = pack(t, S[name])
            writer.add(f"{module}.weight_packed", packed)
            writer.add(f"{module}.weight_scale", scale)
            writer.add(f"{module}.weight_global_scale", divisors[name])
            continue
        if name.endswith(".weight") and t.dim() == 2:
            ignore.add(module)
        writer.add(name, t)
    writer.finalize()

    out = Path(out)
    written = SafetensorsReader(out)
    for name in S:
        m = name[: -len(".weight")]
        w = dequantize(written[f"{m}.weight_packed"], written[f"{m}.weight_scale"], written[f"{m}.weight_global_scale"])
        if not torch.equal(w, reader[name]):
            raise RuntimeError(f"{name}: the written checkpoint does not decode to the materialized weight")
    copy_model_files(out, src)
    shutil.copyfile(src / "receipt.json", out / "receipt.json")
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = quantization_config(sorted(ignore))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {"quantized_linears": len(S)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--materialized", required=True, help="materialized NVFP4 run directory to pack")
    p.add_argument("--out_dir", required=True)
    a = p.parse_args(argv)
    print(export(a.materialized, a.out_dir))


if __name__ == "__main__":
    main()
