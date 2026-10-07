"""Run one generation column of a paper table on one GPU: load items, sample with vLLM, score,
write results.

    python -m quasar.eval.run --table healing --family qwen --task aime25 --model CKPT --out_dir OUT

Decode profile and budgets are the table's setting for the family (``quasar.eval.settings``).
Writes ``OUT/<task>.json`` (``score`` = avg@k in [0, 1]) and the raw samples to
``OUT/<task>.gens.jsonl.gz``.
"""

from __future__ import annotations

import argparse
import dataclasses
from functools import partial
from pathlib import Path

from quasar.eval import common as C
from quasar.eval import lcb, longbench, mathbench, mcq
from quasar.eval.settings import DATASETS, DECODE, SEED, TABLES, Task

TASKS = {  # task -> (load items, judge one sample)
    **{b: (partial(mathbench.load, b), mathbench.judge) for b in mathbench.BENCHES},
    "mmlu_pro": (mcq.load_mmlu_pro, mcq.judge),
    "supergpqa": (mcq.load_supergpqa, mcq.judge),
    "lcb": (lcb.load, None),  # scored by LiveCodeBench's own evaluator (quasar.eval.lcb)
    "longbench_v2": (longbench.load, mcq.judge),
}


def score(task: str, items: list[dict], samples: list[list[str]]) -> dict:
    """avg@k of the task's judge plus per-item records (first sample's extraction)."""
    if task == "lcb":
        return lcb.summarize(items, samples)
    judge = TASKS[task][1]
    flags, per_item = [], []
    for item, ss in zip(items, samples):
        judged = [judge(item, s) for s in ss]
        flags.append([ok for _, ok in judged])
        per_item.append(
            {
                "id": item["id"],
                "gold": item["gold"],
                "extracted": judged[0][0],
                "n_correct": sum(flags[-1]),
                "k": len(judged),
            }
        )
    return {"score": C.avg_at_k(flags), "n_items": len(items), "per_item": per_item}


def run(task: str, family: str, model: str, out_dir: str, setting: Task, *, limit: int = 0) -> dict:
    from transformers import AutoTokenizer

    decode = DECODE[family]
    tokenizer = AutoTokenizer.from_pretrained(model)
    load = TASKS[task][0]
    if task == "longbench_v2":  # documents are cut to what the generation budget leaves of the window
        items = load(tokenizer, decode.thinking, setting.max_model_len - setting.max_new_tokens)
    else:
        items = load()
    items = items[:limit] if limit else items
    prompts = [C.render_prompt(tokenizer, it["messages"], decode.thinking) for it in items]
    print(f"[{task}] {len(items)} items x k={setting.k}, {setting}, {decode}", flush=True)
    llm = C.make_llm(model, max_model_len=setting.max_model_len)
    samples, stats = C.generate(llm, tokenizer, prompts, decode=decode, task=setting)
    C.save_generations(Path(out_dir) / f"{task}.gens.jsonl.gz", items, samples)
    settings = {
        "model": model,
        "family": family,
        "seed": SEED,
        "limit": limit,
        "decode": dataclasses.asdict(decode),
        "task": dataclasses.asdict(setting),
        "dataset": dataclasses.asdict(DATASETS[task]),
    }
    result = {"task": task, **score(task, items, samples), "generation": stats, "settings": settings}
    path = Path(out_dir) / f"{task}.json"
    C.write_json(path, result)
    print(f"[{task}] score={result['score']} n_items={result['n_items']} -> {path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, choices=sorted(TABLES))
    ap.add_argument("--family", required=True, choices=sorted(DECODE))
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True, help="HF checkpoint (its own tokenizer and chat template are used).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Debug: first N items only.")
    args = ap.parse_args(argv)
    setting = TABLES[args.table].get(args.family, {}).get(args.task)
    if setting is None:
        ap.error(f"{args.task} is not a column of the {args.table} table for {args.family}")
    run(args.task, args.family, args.model, args.out_dir, setting, limit=args.limit)


if __name__ == "__main__":
    main()
