"""Shared evaluation helpers: benchmark datasets, thinking-aware prompts, one vLLM engine/generate
path, avg@k and result files.

vLLM (optional ``eval`` extra) is imported lazily, so scoring and tests run without it.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quasar.data import THINK_OPEN
from quasar.eval.settings import DATASETS, GPU_MEMORY_UTILIZATION, SEED, Decode, Task

THINK_CLOSE = "</think>"


def hf_rows(task: str, split: str | None = None):
    """Rows of ``DATASETS[task]`` at its revision (None: the Hub's current snapshot); several configs
    are concatenated (AIME'25)."""
    from datasets import concatenate_datasets, load_dataset

    spec = DATASETS[task]
    configs = spec.config if isinstance(spec.config, tuple) else (spec.config,)
    return concatenate_datasets(
        [load_dataset(spec.repo, c, split=split or spec.split, revision=spec.revision) for c in configs]
    )


def strip_thinking(text: str) -> str:
    """Text after the LAST ``</think>``; the whole text when the trace never closed.

    Answers are read from this span so a tentative answer inside the reasoning cannot
    shadow the final one.
    """
    return text.rsplit(THINK_CLOSE, 1)[-1]


def render_prompt(tokenizer: Any, messages: list[dict], thinking: bool) -> str:
    """Chat-template ``messages`` with the generation header.

    In thinking mode the prompt always ends with an open ``<think>``: thinking-native
    templates already emit it, base/instruct students get it appended so every
    checkpoint decodes in the mode it was trained for.
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if thinking and not text.rstrip().endswith(THINK_OPEN.rstrip()):
        text += THINK_OPEN
    return text


def encode(tokenizer: Any, prompts: Sequence[str]) -> list[list[int]]:
    """Token ids without added special tokens: the template already emits BOS, and a
    second one (Llama-3) would shift every score."""
    return [list(ids) for ids in tokenizer(list(prompts), add_special_tokens=False)["input_ids"]]


def make_llm(model: str, *, max_model_len: int):
    """A single-GPU bf16 vLLM engine."""
    from vllm import LLM

    return LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=SEED,
    )


def generate(
    llm, tokenizer: Any, prompts: Sequence[str], *, decode: Decode, task: Task
) -> tuple[list[list[str]], dict]:
    """``task.k`` samples per prompt (input order) plus generation statistics.

    Seeds are per request, so batching does not change the samples. Each prompt's
    budget is clipped to what its length leaves of ``task.max_model_len``, minus 8 tokens.
    """
    from vllm import SamplingParams

    token_ids = encode(tokenizer, prompts)
    temperature = decode.temperature if task.temperature is None else task.temperature
    params = [
        SamplingParams(
            n=task.k,
            temperature=temperature,
            top_p=decode.top_p,
            top_k=decode.top_k,
            seed=SEED,
            # 8 tokens of slack below the window, as in the paper's SuperGPQA/MMLU-Pro runs.
            max_tokens=min(task.max_new_tokens, task.max_model_len - len(ids) - 8),
        )
        for ids in token_ids
    ]
    outs = llm.generate([{"prompt_token_ids": ids} for ids in token_ids], params)
    samples = [[c.text for c in o.outputs] for o in outs]
    completions = [c for o in outs for c in o.outputs]
    n = len(completions)
    stats = {
        "n_samples": n,
        "frac_length_stop": sum(c.finish_reason == "length" for c in completions) / n,
        "mean_gen_tokens": sum(len(c.token_ids) for c in completions) / n,
        "frac_think_closed": sum(THINK_CLOSE in c.text for c in completions) / n,
    }
    return samples, stats


def avg_at_k(flags: Sequence[Sequence[bool]]) -> float:
    """Mean over items of the fraction of correct samples (pass@1 estimated from k)."""
    return sum(sum(f) / len(f) for f in flags) / len(flags)


# -------------------------------------------------------------------- result files
def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_generations(path: str | Path, items: Sequence[dict], samples: Sequence[Sequence[str]]) -> None:
    """One line per item: its scoring fields (prompt dropped) and its raw samples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for item, ss in zip(items, samples):
            row = {k: v for k, v in item.items() if k != "messages"}
            fh.write(json.dumps({**row, "samples": list(ss)}, ensure_ascii=False) + "\n")


def load_generations(path: str | Path) -> tuple[list[dict], list[list[str]]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows, [row.pop("samples") for row in rows]
