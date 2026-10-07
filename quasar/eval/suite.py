"""Evaluate one checkpoint on one paper table, task after task on one GPU.

    python -m quasar.eval.suite --table healing --family qwen --model CKPT --out_dir OUT \\
        --heldout heldout.jsonl --ruler_data RULER_DATA --lcb_repo LiveCodeBench --lcb_python ENV/bin/python
    python -m quasar.eval.suite --table adaptation --family qwen --model CKPT --out_dir OUT --heldout eval.jsonl

Every table starts with the held-out metrics (perplexity; KL / top-1 against the
family's BF16 reference except for adaptation), then its generation columns
(healing adds RULER). Every task runs in its own process (so vLLM returns the GPU on
exit) and writes ``OUT/<task>.json``; finished tasks are skipped on a re-run.
``OUT/summary.json`` holds the table row: accuracies x100 and their average (KL /
top-1 / perplexity are not averaged).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from quasar.eval.common import read_json, write_json
from quasar.eval.settings import HELDOUT_MAX_SEQ_LEN, TABLES, TEACHER_REVISIONS, TEACHERS


def score(out: Path, task: str):
    """The task's score, or None if it has not been produced (LiveCodeBench: scored) yet."""
    path = out / f"{task}.json"
    return read_json(path).get("score") if path.is_file() else None


def steps(args: argparse.Namespace) -> list[tuple[str, list[str], bool]]:
    """(name, command, done) for every task of the table, in table order."""
    out, m = Path(args.out_dir), [sys.executable, "-m"]
    common = ["--model", args.model, "--out_dir", args.out_dir]
    cmd = m + [
        "quasar.eval.heldout",
        *common,
        "--data",
        str(args.heldout),
        "--max_seq_len",
        str(HELDOUT_MAX_SEQ_LEN[args.table]),
    ]
    if args.table != "adaptation":
        cmd += ["--teacher", TEACHERS[args.family]]
        cmd += ["--teacher_revision", TEACHER_REVISIONS[args.family]] if args.family in TEACHER_REVISIONS else []
    plan = [("heldout", cmd, (out / "heldout.json").is_file())]
    for task in TABLES[args.table][args.family]:
        done = (out / f"{task}.gens.jsonl.gz").is_file() and (out / f"{task}.json").is_file()
        plan.append(
            (
                task,
                m + ["quasar.eval.run", "--task", task, "--family", args.family, "--table", args.table, *common],
                done,
            )
        )
        if task == "lcb" and args.lcb_repo:
            plan.append(
                (
                    "lcb_score",
                    m
                    + [
                        "quasar.eval.lcb",
                        "--out_dir",
                        args.out_dir,
                        "--lcb_repo",
                        args.lcb_repo,
                        "--lcb_python",
                        args.lcb_python,
                    ],
                    score(out, "lcb") is not None,
                )
            )
    if args.table == "healing":
        plan.append(
            (
                "ruler",
                m + ["quasar.eval.ruler", "--family", args.family, "--data_dir", str(args.ruler_data), *common],
                score(out, "ruler") is not None,
            )
        )
    return [s for s in plan if not args.tasks or s[0].removesuffix("_score") in args.tasks]


def summarize(table: str, family: str, out: Path) -> dict:
    held = read_json(out / "heldout.json") if (out / "heldout.json").is_file() else {}
    if table == "adaptation":
        row = {"ppl": held.get("eval_ppl")}
    else:
        row = {"kl": held.get("eval_kl"), "top1": 100 * held["eval_top1"] if "eval_top1" in held else None}
    columns = list(TABLES[table][family]) + (["ruler"] if table == "healing" else [])
    for task in columns:
        s = score(out, task)
        row[task] = None if s is None else 100 * s
    accs = [row[t] for t in columns]
    row["avg"] = None if None in accs else sum(accs) / len(accs)
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, choices=sorted(TABLES))
    ap.add_argument("--family", required=True, choices=sorted(TEACHERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--heldout", type=Path, help="Held-out chat jsonl for perplexity (and KL / top-1).")
    ap.add_argument("--ruler_data", type=Path, help="RULER data cache of this family (healing).")
    ap.add_argument("--lcb_repo", help="LiveCodeBench checkout; without it LCB is generated but not scored.")
    ap.add_argument("--lcb_python", default=sys.executable)
    ap.add_argument("--tasks", type=lambda s: set(s.split(",")), help="Comma-separated subset of the table's tasks.")
    args = ap.parse_args(argv)
    if args.family not in TABLES[args.table]:
        ap.error(f"no {args.table} table for family {args.family}")

    todo = [(name, cmd) for name, cmd, done in steps(args) if not done]
    needs = {"heldout": ("heldout", args.heldout), "ruler": ("ruler_data", args.ruler_data)}
    for name, _ in todo:
        if name in needs and needs[name][1] is None:
            ap.error(f"{name} needs --{needs[name][0]}")
    for name, cmd in todo:
        print(f"[suite] {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    out = Path(args.out_dir)
    row = summarize(args.table, args.family, out)
    write_json(out / "summary.json", {"table": args.table, "family": args.family, "model": args.model, "row": row})
    print("[suite] " + "  ".join(f"{k}={'-' if v is None else f'{v:.3f}'}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
