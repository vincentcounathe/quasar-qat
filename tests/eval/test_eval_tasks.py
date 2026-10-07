"""LongBench-v2, LiveCodeBench, RULER and the generation runner (no GPU, no network)."""

import json

from quasar.eval import common as C
from quasar.eval import lcb, longbench, mcq, ruler, run
from quasar.eval.settings import DECODE, RULER_TASKS, TABLES, Task

# ---------------------------------------------------------------- LongBench-v2
ROW = {
    "context": " The sky is blue. ",
    "question": "What color is the sky? ",
    "choice_A": "Red",
    "choice_B": "Blue",
    "choice_C": "Green",
    "choice_D": "Yellow",
}


def test_longbench_prompt():
    p = longbench.build_prompt(ROW)
    assert "$" not in p
    assert "<text>\nThe sky is blue.\n</text>" in p and "this question: What color is the sky?\n" in p
    assert "(A) Red\n(B) Blue\n(C) Green\n(D) Yellow" in p
    assert p.endswith('"The correct answer is (insert answer here)".')


def test_truncate_middle_matches_official():
    ids = list(range(100))
    assert longbench.truncate_middle(ids, 100) == ids
    assert longbench.truncate_middle(ids, 10) == list(range(5)) + list(range(95, 100))
    assert longbench.truncate_middle(ids, 7) == ids[:3] + ids[-4:]


# --------------------------------------------------------------- LiveCodeBench
def test_lcb_messages():
    sys_msg, user = lcb.build_messages("Sum two numbers from stdin.")
    assert sys_msg == {"role": "system", "content": lcb.SYSTEM_MESSAGE}
    assert user["content"].startswith("### Question:\nSum two numbers from stdin.\n\n")
    assert f"### Format: {lcb.WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n" in user["content"]
    assert user["content"].endswith("### Answer: (use the provided format with backticks)\n\n")
    starter = "class Solution:\n    def f(self): pass"
    with_starter = lcb.build_messages("q", starter)[1]["content"]
    assert f"### Format: {lcb.WITH_STARTER}\n```python\n{starter}\n```\n\n" in with_starter


def test_lcb_extract_code():
    assert lcb.extract_code("```python\nprint(1 + 1)\n```") == "print(1 + 1)"
    two = "```python\nx = 1\n```\nbetter:\n```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    assert lcb.extract_code(two) == "a, b = map(int, input().split())\nprint(a + b)"
    assert lcb.extract_code("<think>```python\nwrong()\n```</think>\n```\nright()\n```") == "right()"
    assert lcb.extract_code("dangling fence\n```python\nprint(1)") == ""
    assert lcb.extract_code("no code") == "" and lcb.extract_code("") == ""


# ----------------------------------------------------------------------- RULER
def test_ruler_command_and_scores(tmp_path):
    cmd = ruler.lm_eval_command("ckpt", 8192, tmp_path)
    assert cmd[1:4] == ["-m", "lm_eval", "run"]
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "pretrained=ckpt" in model_args and "max_model_len=8704" in model_args and "add_bos_token=True" in model_args
    assert cmd[cmd.index("--limit") + 1] == "100"
    assert json.loads(cmd[cmd.index("--metadata") + 1]) == {"max_seq_lengths": [8192]}
    assert cmd[cmd.index("--tasks") + 1].split(",") == [f"{t}_quasar" for t in RULER_TASKS]
    results = {f"{t}_quasar": {"8192,none": 0.5 + i / 100} for i, t in enumerate(RULER_TASKS)}
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "results_0.json").write_text(json.dumps({"results": results}))
    assert ruler.read_scores(tmp_path, 8192) == {t: 0.5 + i / 100 for i, t in enumerate(RULER_TASKS)}


def test_ruler_yamls_exist():
    for t in RULER_TASKS:
        text = (ruler.TASKS_DIR / f"{t}_quasar.yaml").read_text()
        assert f"task: {t}_quasar" in text and "quasar_ruler." in text


# ---------------------------------------------------------------------- runner
def test_every_table_column_is_runnable():
    for table in TABLES.values():
        for family, columns in table.items():
            assert family in DECODE and set(columns) <= set(run.TASKS)


def test_score_and_saved_generations(tmp_path):
    items = [
        {"id": "a", "gold": "B", "n_options": 4, "messages": [{"role": "user", "content": "q"}]},
        {"id": "b", "gold": "C", "n_options": 4, "messages": [{"role": "user", "content": "q"}]},
    ]
    samples = [["the answer is (B)", "the answer is (A)"], ["<think>x</think>Answer: C", "nothing"]]
    res = run.score("mmlu_pro", items, samples)
    assert res["score"] == 0.5 and res["n_items"] == 2
    assert res["per_item"][0] == {"id": "a", "gold": "B", "extracted": "B", "n_correct": 1, "k": 2}
    C.save_generations(tmp_path / "mmlu_pro.gens.jsonl.gz", items, samples)
    loaded, loaded_samples = C.load_generations(tmp_path / "mmlu_pro.gens.jsonl.gz")
    assert loaded_samples == samples and "messages" not in loaded[0]


def test_lcb_summary_defers_scoring():
    res = run.score("lcb", [{"id": "1"}], [["```python\nprint(1)\n```", "no code"]])
    assert res == {"score": None, "n_items": 1, "n_empty_code": 1}


def test_run_end_to_end_with_a_fake_engine(tmp_path, monkeypatch):
    """Plumbing of run(): prompts, per-request budgets, sampling knobs, scoring and files."""
    import sys
    import types

    seen = {}

    class Completion:
        def __init__(self, text):
            self.text, self.finish_reason, self.token_ids = text, "stop", [0] * 5

    class LLM:
        def __init__(self, **kwargs):
            seen["engine"] = kwargs

        def generate(self, prompts, params):
            seen["prompts"], seen["params"] = prompts, params
            return [
                types.SimpleNamespace(outputs=[Completion("<think>x</think> the answer is (B)")] * p.n) for p in params
            ]

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            return "<|user|>" + "".join(m["content"] for m in messages) + "<|assistant|>"

        def __call__(self, texts, add_special_tokens=True):
            return {"input_ids": [[ord(c) for c in t] for t in texts]}

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda path: Tokenizer())
    items = [
        {"id": str(i), "gold": g, "n_options": 4, "messages": [{"role": "user", "content": "q" * 10 * i}]}
        for i, g in enumerate("BC", start=1)
    ]
    monkeypatch.setitem(run.TASKS, "mmlu_pro", (lambda: items, mcq.judge))
    res = run.run("mmlu_pro", "qwen", "ckpt", str(tmp_path), Task(k=2, max_new_tokens=150, max_model_len=200))

    assert seen["prompts"][0]["prompt_token_ids"][-8:] == [ord(c) for c in "<think>\n"]  # thinking seeded
    p0, p1 = seen["params"]
    assert (p0.n, p0.temperature, p0.top_p, p0.top_k, p0.seed) == (2, 0.6, 0.95, 20, 1234)
    assert p0.max_tokens == 150 and p1.max_tokens == 200 - len(seen["prompts"][1]["prompt_token_ids"]) - 8 < 150
    assert seen["engine"]["max_model_len"] == 200 and seen["engine"]["dtype"] == "bfloat16"
    assert res["score"] == 0.5 and res["generation"]["frac_think_closed"] == 1.0
    assert C.read_json(tmp_path / "mmlu_pro.json")["settings"]["task"]["k"] == 2
    assert C.load_generations(tmp_path / "mmlu_pro.gens.jsonl.gz")[1][0] == ["<think>x</think> the answer is (B)"] * 2
