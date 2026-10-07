"""LiveCodeBench v6 code generation: prompts and code extraction of the official
``lcb_runner`` generic-chat protocol, scored by LiveCodeBench's own execution-based
``custom_evaluator``.

Generation runs through ``python -m quasar.eval.run --task lcb``. Scoring needs a
LiveCodeBench checkout installed in its own environment (``pip install -e .``)::

    python -m quasar.eval.lcb --out_dir OUT --lcb_repo LiveCodeBench --lcb_python ENV/bin/python

The evaluator executes model-written code without a sandbox: run it on a disposable
Linux machine (it refuses to run on macOS).
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from quasar.data.common import iter_jsonl
from quasar.eval import common as C
from quasar.eval.common import strip_thinking
from quasar.eval.settings import DATASETS, LCB_FILES, LCB_TIMEOUT_S

# lcb_runner/prompts/code_generation.py.
SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem specification) and "
    "will generate a correct Python program that matches the specification and passes all tests."
)
WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your "
    "code within delimiters."
)
WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
    "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
    "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def build_messages(question: str, starter_code: str = "") -> list[dict]:
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += f"### Format: {WITH_STARTER}\n```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}]


def extract_code(sample: str) -> str:
    """Official extractor on the answer after ``</think>``: the text between the last
    two lines containing a code fence ("" when there are fewer than two)."""
    lines = strip_thinking(sample).split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[fences[-2] + 1 : fences[-1]]) if len(fences) >= 2 else ""


def load() -> list[dict]:
    """The release_v6 problems at the pinned dataset revision, sorted by id."""
    from huggingface_hub import hf_hub_download

    spec = DATASETS["lcb"]
    rows = []
    for fname in LCB_FILES:
        rows += iter_jsonl(hf_hub_download(spec.repo, fname, repo_type="dataset", revision=spec.revision))
    rows.sort(key=lambda r: str(r["question_id"]))
    return [
        {"id": str(r["question_id"]), "messages": build_messages(r["question_content"], r.get("starter_code") or "")}
        for r in rows
    ]


def summarize(items: list[dict], samples: list[list[str]]) -> dict:
    """Generation-side summary; the score comes from the official evaluator."""
    n_empty = sum(not extract_code(s) for ss in samples for s in ss)
    return {"score": None, "n_items": len(items), "n_empty_code": n_empty}


def score(out_dir: str, lcb_repo: str, lcb_python: str) -> dict:
    """Run ``lcb_runner.runner.custom_evaluator`` on the saved generations and record
    pass@1 (mean over problems of the fraction of passing samples)."""
    if platform.system() == "Darwin":
        sys.exit("[lcb] refusing to execute model-generated code on macOS")
    out = Path(out_dir)
    items, samples = C.load_generations(out / "lcb.gens.jsonl.gz")
    codes = [{"question_id": it["id"], "code_list": [extract_code(s) for s in ss]} for it, ss in zip(items, samples)]
    custom = (out / "lcb_custom_outputs.json").resolve()
    C.write_json(custom, codes)
    subprocess.run(
        [
            lcb_python,
            "-m",
            "lcb_runner.runner.custom_evaluator",
            "--custom_output_file",
            str(custom),
            "--scenario",
            "codegeneration",
            "--release_version",
            "release_v6",
            "--timeout",
            str(LCB_TIMEOUT_S),
        ],
        cwd=lcb_repo,
        check=True,
    )
    graded = C.read_json(str(custom)[: -len(".json")] + "_codegeneration_output_eval_all.json")
    flags = [[bool(g) for g in r["graded_list"]] for r in graded]
    per_item = [{"id": str(r["question_id"]), "n_correct": sum(f), "k": len(f)} for r, f in zip(graded, flags)]
    result = C.read_json(out / "lcb.json")
    result.update(score=C.avg_at_k(flags), per_item=per_item)
    C.write_json(out / "lcb.json", result)
    print(f"[lcb] pass@1 {result['score']:.4f} over {len(per_item)} problems", flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="Directory holding lcb.json and lcb.gens.jsonl.gz.")
    ap.add_argument("--lcb_repo", required=True, help="LiveCodeBench checkout.")
    ap.add_argument("--lcb_python", default=sys.executable, help="Python of the LiveCodeBench environment.")
    args = ap.parse_args(argv)
    score(args.out_dir, args.lcb_repo, args.lcb_python)


if __name__ == "__main__":
    main()
