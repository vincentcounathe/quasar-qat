"""Prompts, avg@k, held-out metrics and the suite summary (no GPU, no network)."""

import argparse
import math

import pytest

from quasar.eval import common as C
from quasar.eval import heldout, suite
from quasar.eval.settings import DECODE, TABLES


class FakeTokenizer:
    """Records chat-template calls; Llama-like BOS that ``add_special_tokens`` would prepend."""

    bos_token_id = 1

    def __init__(self, rendered):
        self.rendered, self.calls = rendered, []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(kwargs)
        return self.rendered

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [([1] if add_special_tokens else []) + [ord(c) for c in t] for t in texts]}


def test_strip_thinking():
    assert C.strip_thinking("<think>a</think>mid</think>The answer.") == "The answer."
    assert C.strip_thinking("no block") == "no block"


def test_render_prompt_opens_thinking_once():
    plain = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"
    assert C.render_prompt(FakeTokenizer(plain), [], thinking=True) == plain + "<think>\n"
    assert C.render_prompt(FakeTokenizer(plain + "<think>\n"), [], thinking=True) == plain + "<think>\n"
    tok = FakeTokenizer(plain)
    assert C.render_prompt(tok, [], thinking=False) == plain
    assert tok.calls == [{"enable_thinking": False}]


def test_encode_adds_no_second_bos():
    assert C.encode(FakeTokenizer(""), ["ab"]) == [[ord("a"), ord("b")]]


def test_decode_profiles():
    assert DECODE["qwen"].thinking and not DECODE["llama"].thinking
    assert (DECODE["qwen"].temperature, DECODE["qwen"].top_p, DECODE["qwen"].top_k) == (0.6, 0.95, 20)
    assert TABLES["healing"]["qwen"]["lcb"].temperature == 0.2


def test_avg_at_k():
    assert C.avg_at_k([[True, False, True, False], [True], [False, False]]) == pytest.approx((0.5 + 1 + 0) / 3)


# ------------------------------------------------------------- held-out metrics
def test_heldout_cli_against_itself(tiny_checkpoint, tmp_path):
    model, _, rows = tiny_checkpoint
    heldout.main(
        [
            "--model",
            model,
            "--teacher",
            model,
            "--data",
            rows,
            "--max_seq_len",
            "256",
            "--out_dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    res = C.read_json(tmp_path / "heldout.json")
    assert res["n_rows"] == 9 and res["eval_top1"] == 1.0 and res["eval_kl"] == pytest.approx(0.0, abs=1e-6)
    assert res["eval_ppl"] == pytest.approx(math.exp(res["eval_ce"]))


# ------------------------------------------------------------------------ suite
def test_suite_summary(tmp_path):
    C.write_json(tmp_path / "heldout.json", {"eval_kl": 0.05, "eval_top1": 0.9, "eval_ppl": 3.0})
    for i, task in enumerate(TABLES["healing"]["llama"]):
        C.write_json(tmp_path / f"{task}.json", {"score": 0.1 * (i + 1)})
    row = suite.summarize("healing", "llama", tmp_path)
    assert row["kl"] == 0.05 and row["top1"] == pytest.approx(90) and row["ruler"] is None and row["avg"] is None
    C.write_json(tmp_path / "ruler.json", {"score": 0.6})
    row = suite.summarize("healing", "llama", tmp_path)
    assert row["avg"] == pytest.approx(100 * (0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6) / 6)


def test_suite_adaptation_summary(tmp_path):
    C.write_json(tmp_path / "heldout.json", {"eval_ce": 0.5, "eval_ppl": math.exp(0.5)})
    for task in TABLES["adaptation"]["qwen"]:
        C.write_json(tmp_path / f"{task}.json", {"score": 0.5})
    row = suite.summarize("adaptation", "qwen", tmp_path)
    assert row["ppl"] == pytest.approx(math.exp(0.5)) and row["avg"] == pytest.approx(50)


def test_suite_steps_skip_finished_and_filter(tmp_path):
    args = argparse.Namespace(
        table="healing",
        family="qwen",
        model="m",
        out_dir=str(tmp_path),
        heldout="h.jsonl",
        ruler_data="rd",
        lcb_repo="lcb",
        lcb_python="py",
        tasks=None,
    )
    plan = suite.steps(args)
    assert [name for name, _, _ in plan] == [
        "heldout",
        "hmmt26",
        "aime25",
        "math500",
        "mmlu_pro",
        "supergpqa",
        "lcb",
        "lcb_score",
        "longbench_v2",
        "ruler",
    ]
    assert plan[0][1][-6:] == [
        "--max_seq_len",
        "4096",
        "--teacher",
        "Qwen/Qwen3-4B-Thinking-2507",
        "--teacher_revision",
        "768f209d9ea81521153ed38c47d515654e938aea",
    ]
    C.write_json(tmp_path / "heldout.json", {"eval_kl": 0.1, "eval_top1": 0.9})
    assert dict((n, d) for n, _, d in suite.steps(args))["heldout"] is True
    args.tasks = {"lcb", "ruler"}
    assert [n for n, _, _ in suite.steps(args)] == ["lcb", "lcb_score", "ruler"]


def test_suite_nvfp4_and_adaptation_steps(tmp_path):
    args = argparse.Namespace(
        table="nvfp4",
        family="qwen3_8b",
        model="m",
        out_dir=str(tmp_path),
        heldout="h.jsonl",
        ruler_data=None,
        lcb_repo=None,
        lcb_python="py",
        tasks=None,
    )
    plan = suite.steps(args)
    assert [name for name, _, _ in plan] == ["heldout", "hmmt26", "aime25", "supergpqa"]
    assert plan[0][1][-4:] == ["--max_seq_len", "2048", "--teacher", "Qwen/Qwen3-8B"]
    args.table, args.family = "adaptation", "qwen"
    assert "--teacher" not in suite.steps(args)[0][1]
