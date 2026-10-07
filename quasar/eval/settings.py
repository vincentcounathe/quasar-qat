"""Evaluation settings of the paper tables, in one place.

Every number a table cell depends on (teacher, dataset revision, decode profile,
samples per item, token budgets) is defined here and recorded in each result file.
Datasets with ``revision=None`` were evaluated on the Hugging Face snapshot current
at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass

SEED = 1234
GPU_MEMORY_UTILIZATION = 0.85

# BF16 reference per model family: the KL / top-1 reference of quasar.eval.heldout (pinned
# below) and the RULER sizing tokenizer; training loads the recipe's model_path, the same
# model. ``qwen3_8b`` is the NVFP4 table's model (no pinned revision).
TEACHERS = {
    "qwen": "Qwen/Qwen3-4B-Thinking-2507",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
}
TEACHER_REVISIONS = {
    "qwen": "768f209d9ea81521153ed38c47d515654e938aea",
    "llama": "0e9e39f249a16976918f6564b8830bc894c89659",
}


@dataclass(frozen=True)
class Decode:
    """One sampling profile per model family (Qwen3's recommended thinking decode)."""

    thinking: bool
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


DECODE = {"qwen": Decode(thinking=True), "llama": Decode(thinking=False), "qwen3_8b": Decode(thinking=True)}


@dataclass(frozen=True)
class Task:
    """Samples per item (score = avg@k) and token budgets of one table column."""

    k: int
    max_new_tokens: int
    max_model_len: int
    temperature: float | None = None  # overrides the family profile (LiveCodeBench's official 0.2)


@dataclass(frozen=True)
class HFDataset:
    repo: str
    split: str
    config: str | tuple[str, ...] | None = None
    revision: str | None = None


DATASETS = {
    "math500": HFDataset("HuggingFaceH4/MATH-500", "test"),
    "gsm8k": HFDataset("openai/gsm8k", "test", config="main"),
    "aime24": HFDataset("Maxwell-Jia/AIME_2024", "train"),
    "aime25": HFDataset("opencompass/AIME2025", "test", config=("AIME2025-I", "AIME2025-II")),
    "hmmt25": HFDataset("MathArena/hmmt_feb_2025", "train"),
    "hmmt26": HFDataset("MathArena/hmmt_feb_2026", "train", revision="02fba4f74d8e68e73e66a02d540fd979c05c274c"),
    "mmlu_pro": HFDataset("TIGER-Lab/MMLU-Pro", "test"),
    "supergpqa": HFDataset("m-a-p/SuperGPQA", "train", revision="4430d4458112c7d4497fdcf94d7cc223313d6acf"),
    "lcb": HFDataset("livecodebench/code_generation_lite", "test", revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505"),
    "longbench_v2": HFDataset("THUDM/LongBench-v2", "train"),
}
MMLU_PRO_SHOTS = 5
SUPERGPQA_ITEMS = 4000  # stratified by discipline, drawn with SEED
LCB_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl")  # release_v6
LCB_TIMEOUT_S = 6

# Generation columns of each table, in table order.
_LCB_T = 0.2
TABLES: dict[str, dict[str, dict[str, Task]]] = {
    "healing": {
        "qwen": {
            "hmmt26": Task(32, 32768, 40960),
            "aime25": Task(32, 38912, 43008),
            "math500": Task(4, 32768, 36864),
            "mmlu_pro": Task(1, 16384, 24576),
            "supergpqa": Task(1, 32768, 32768),
            "lcb": Task(1, 32768, 40960, temperature=_LCB_T),
            "longbench_v2": Task(1, 32768, 131072),
        },
        "llama": {
            "math500": Task(4, 8192, 12288),
            "mmlu_pro": Task(1, 4096, 12288),
            "supergpqa": Task(1, 8192, 16384),
            "lcb": Task(1, 8192, 16384, temperature=_LCB_T),
            "longbench_v2": Task(1, 1024, 131072),
        },
    },
    # Qwen3-4B-Base students (context 32768) in thinking mode.
    "adaptation": {
        "qwen": {b: Task(8, 28672, 32768) for b in ("math500", "gsm8k", "aime24", "aime25", "hmmt25")},
    },
    # Packed NVFP4 checkpoints served by vLLM within Qwen3-8B's 40,960-token window.
    "nvfp4": {
        "qwen3_8b": {
            "hmmt26": Task(32, 32768, 40960),
            "aime25": Task(32, 38912, 40960),
            "supergpqa": Task(1, 32768, 40960),
        },
    },
}

# Held-out metrics (quasar.eval.heldout): perplexity, plus KL / top-1 against the family's
# BF16 reference except for adaptation. Sequence length: the recipe's max_seq_len (healing 4096,
# adaptation 18432); NVFP4 trains at 8192 but its KL / top-1 is measured at 2048, so its
# heldout.json is not the run's in-loop evaluation.
HELDOUT_MAX_SEQ_LEN = {"healing": 4096, "adaptation": 18432, "nvfp4": 2048}

# RULER: 6 tasks x 5 context lengths, 100 samples each, greedy, raw completion.
RULER_TASKS = ("niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "ruler_vt", "ruler_qa_squad", "ruler_qa_hotpot")
RULER_LENGTHS = (4096, 8192, 16384, 32768, 65536)
RULER_LIMIT = 100
RULER_CONTEXT_MARGIN = 512  # max_model_len = L + margin
