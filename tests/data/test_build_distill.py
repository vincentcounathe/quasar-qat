import json

from quasar.data import build_distill


def _rec(prompt, content, reasoning="r", finish="stop", **extra):
    """A sample_teacher record."""
    asst = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    return (
        json.dumps(
            {"index": 0, "messages": [{"role": "user", "content": prompt}, asst], "finish_reason": finish, **extra}
        )
        + "\n"
    )


def test_split_think():
    split = build_distill.split_think
    inline = {"role": "assistant", "content": "step one\n</think>\n\nThe answer.", "reasoning_content": ""}
    assert split(inline) == {"role": "assistant", "content": "The answer.", "reasoning_content": "step one"}
    tagged = {"role": "assistant", "content": "<think>\nstep one\n</think>\n\nThe answer.", "reasoning_content": ""}
    assert split(tagged)["reasoning_content"] == "step one"
    already = {"role": "assistant", "content": "a </think> b", "reasoning_content": "kept"}
    assert split(already) is already
    plain = {"role": "assistant", "content": "no reasoning here", "reasoning_content": ""}
    assert split(plain) is plain


def test_filters(toy_tokenizer):
    lines = [
        _rec("p0", "ok"),
        _rec("p1", "cut off", finish="length"),
        _rec("p1", "second sample of p1 passes"),  # the first *passing* sample of a prompt is kept
        _rec(" p0 ", "duplicate of p0 after stripping"),
        _rec("p2", "   "),
        _rec("p3", "thinking</think>", reasoning=""),  # inline reasoning, empty answer
        _rec("held out", "ok"),
        "{not json\n",
        _rec("p4", "word " * 100),
        _rec("p5 " + "long prompt " * 15, "ok"),
        _rec("p6", "fine answer"),
    ]
    kept, counts = build_distill.filter_records(
        lines, toy_tokenizer, exclude={"held out"}, max_seq_len=80, max_prompt_len=20
    )
    assert [m[0]["content"] for m in kept] == ["p0", "p1", "p6"]
    assert kept[1][1]["content"] == "second sample of p1 passes"
    assert dict(counts) == {
        "read": 11,
        "kept": 3,
        "kept_tokens": counts["kept_tokens"],
        "not_stop": 1,
        "duplicate": 1,
        "empty_answer": 2,
        "excluded": 1,
        "bad_json": 1,
        "too_long": 1,
        "prompt_too_long": 1,
    }


def test_cli_split_is_seeded_and_disjoint(toy_tokenizer, tmp_path):
    src = tmp_path / "samples.jsonl"
    src.write_text("".join(_rec(f"prompt {i}", f"answer {i}") for i in range(50)) + _rec("prompt 3", "dup"))
    held = tmp_path / "held.jsonl"
    held.write_text(_rec("prompt 7", "x"))

    def build(out, seed):
        build_distill.main(
            [
                "--src",
                str(src),
                "--out_dir",
                str(out),
                "--tokenizer",
                "toy",
                "--exclude",
                str(held),
                "--eval_rows",
                "5",
                "--seed",
                str(seed),
            ]
        )
        return tuple(
            [json.loads(line)["messages"][0]["content"] for line in (out / name).read_text().splitlines()]
            for name in ("train.jsonl", "eval.jsonl")
        )

    train, ev = build(tmp_path / "a", 1)
    assert len(train) == 44 and len(ev) == 5 and not set(train) & set(ev)
    assert "prompt 7" not in train + ev
    assert build(tmp_path / "b", 1) == (train, ev) and build(tmp_path / "c", 2) != (train, ev)
    stats = json.loads((tmp_path / "a" / "stats.json").read_text())
    assert stats["train"] == 44 and stats["counts"]["duplicate"] == 1
