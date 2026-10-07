"""RULER (NIAH multikey 1-3, variable tracking, SQuAD / HotpotQA QA) through lm-eval.

Raw completion (no chat template), greedy, 100 samples per task and context length,
one lm-eval run per length. The data are synthesized once per family on CPU, sized
with the family teacher's tokenizer, and shared by every checkpoint of the family.
The score is the mean over lengths of the 6-task mean.

    python -m quasar.eval.ruler --family qwen --data_dir RULER_DATA --pregen        # CPU only
    python -m quasar.eval.ruler --family qwen --data_dir RULER_DATA --model CKPT --out_dir OUT
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from quasar.eval import common as C
from quasar.eval.settings import (
    DECODE,
    GPU_MEMORY_UTILIZATION,
    RULER_CONTEXT_MARGIN,
    RULER_LENGTHS,
    RULER_LIMIT,
    RULER_TASKS,
    SEED,
    TEACHERS,
)

TASKS_DIR = Path(__file__).resolve().parent / "ruler_tasks"


def _env(family: str, data_dir: str) -> dict:
    return dict(
        os.environ,
        QUASAR_RULER_TOKENIZER=TEACHERS[family],
        QUASAR_RULER_DATA=str(Path(data_dir).resolve()),
        QUASAR_RULER_EMPTY_THINK="1" if DECODE[family].thinking else "0",
        TOKENIZERS_PARALLELISM="false",
    )


def pregen(family: str, data_dir: str) -> None:
    """Build (or load) every (task, length) cache on CPU, in a child process because
    the task module reads its settings from the environment at import."""
    script = TASKS_DIR / "quasar_ruler.py"
    subprocess.run([sys.executable, str(script), *map(str, RULER_LENGTHS)], env=_env(family, data_dir), check=True)


def lm_eval_command(model: str, length: int, out_dir: Path) -> list[str]:
    model_args = (
        f"pretrained={model},dtype=bfloat16,tensor_parallel_size=1,add_bos_token=True,seed={SEED},"
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION},max_model_len={length + RULER_CONTEXT_MARGIN}"
    )
    return [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(f"{t}_quasar" for t in RULER_TASKS),
        "--include_path",
        str(TASKS_DIR),
        "--metadata",
        json.dumps({"max_seq_lengths": [length]}),
        "--limit",
        str(RULER_LIMIT),
        "--batch_size",
        "auto",
        "--output_path",
        str(out_dir),
    ]


def read_scores(run_dir: Path, length: int) -> dict[str, float]:
    """Per-task score at ``length`` from lm-eval's newest results file."""
    newest = max(run_dir.rglob("results_*.json"), key=os.path.getmtime)
    results = C.read_json(newest)["results"]
    return {t: float(results[f"{t}_quasar"][f"{length},none"]) for t in RULER_TASKS}


def run(model: str, family: str, data_dir: str, out_dir: str) -> dict:
    pregen(family, data_dir)
    by_length = {}
    for length in RULER_LENGTHS:
        run_dir = Path(out_dir) / "ruler" / str(length)
        subprocess.run(lm_eval_command(model, length, run_dir), env=_env(family, data_dir), check=True)
        scores = read_scores(run_dir, length)
        by_length[str(length)] = {"score": sum(scores.values()) / len(scores), "tasks": scores}
    result = {
        "task": "ruler",
        "score": sum(v["score"] for v in by_length.values()) / len(by_length),
        "by_length": by_length,
        "settings": {"model": model, "family": family, "sizing_tokenizer": TEACHERS[family], "limit": RULER_LIMIT},
    }
    C.write_json(Path(out_dir) / "ruler.json", result)
    print(f"[ruler] score={result['score']:.4f}", flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(TEACHERS))
    ap.add_argument("--data_dir", required=True, help="RULER data cache of this family.")
    ap.add_argument("--pregen", action="store_true", help="Only build the data cache (CPU).")
    ap.add_argument("--model")
    ap.add_argument("--out_dir")
    args = ap.parse_args(argv)
    if args.pregen:
        pregen(args.family, args.data_dir)
    elif args.model and args.out_dir:
        run(args.model, args.family, args.data_dir, args.out_dir)
    else:
        ap.error("--model and --out_dir are required unless --pregen")


if __name__ == "__main__":
    main()
