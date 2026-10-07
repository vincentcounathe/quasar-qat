import json

import pytest

from quasar.data import MATH_INSTRUCTION, build_math

BOXED = "So the answer is \\boxed{4}."
BENCH = [
    "Let x be a real number such that x squared plus three x equals ten. Find the sum of all possible values of x.",
    "A bag holds five red and seven blue marbles; two are drawn without replacement. What is the probability that "
    "both are red? Express it as m/n where m and n are relatively prime positive integers and find m+n.",
]


@pytest.mark.parametrize(
    "problem, solution, reason",
    [
        ("p", f"<think> r</think>\n{BOXED}", None),
        ("p <think>", f"<think> r</think>{BOXED}", "problem_markers"),
        ("p", f"<think>r<think></think>{BOXED}", "think_tags"),
        ("p", f"</think>r<think>{BOXED}", "think_tags"),
        ("p", "<think>r</think>  ", "empty_answer"),
        ("p", "<think>r</think> the answer is \\boxed 4", "no_boxed"),
        ("p", f"<think>r</think>{BOXED}<|im_end|>", "solution_markers"),
    ],
)
def test_rejection(problem, solution, reason):
    assert build_math.rejection(problem, solution) == reason


def test_to_messages_matches_eval_prompt():
    user, asst = build_math.to_messages("What is 2+2?  \n", "<think>  \n\nreason</think>\n\\boxed{4}")
    assert user == {"role": "user", "content": f"What is 2+2?\n\n{MATH_INSTRUCTION}"}
    assert asst == {"role": "assistant", "content": "<think>\nreason</think>\n\\boxed{4}"}


def test_contamination():
    flagged = build_math.Contamination(BENCH)
    assert flagged(BENCH[0].upper().replace(" ", "  ") + "!")  # a reformatted copy
    assert flagged(BENCH[0][:80] + " and then a different ending about circles and triangles in a plane")
    unrelated = (
        "Three circles are tangent to a line and to each other. The radii are distinct integers; "
        "the answer can be written as m/n where m and n are relatively prime positive integers."
    )
    assert not flagged(unrelated)  # shares only competition boilerplate
    assert not flagged("Find x.")


def _row(problem, think="reasoning", answer=BOXED):
    return {"problem": problem, "generated_solution": f"<think> {think}</think>\n{answer}"}


def test_build_dedups_and_long_rows_do_not_use_up_problems(toy_tokenizer):
    stream = [
        _row("p0"),
        _row("p1", think="very " * 50),  # too long, a later trace of p1 still counts
        _row("p0", think="another trace"),
        _row("p1"),
        _row(BENCH[1]),
        _row("p2", answer="no box"),
        _row("p3"),
        _row("p4"),
    ]
    kept, counts = build_math.build(stream, toy_tokenizer, build_math.Contamination(BENCH), max_seq_len=40, rows=3)
    assert [m[0]["content"].split("\n")[0] for m in kept] == ["p0", "p1", "p3"]
    assert (counts["too_long"], counts["duplicate"], counts["contaminated"], counts["no_boxed"]) == (1, 1, 1, 1)
    assert counts["read"] == 7  # stopped at --rows


def test_cli(toy_tokenizer, tmp_path, monkeypatch):
    import datasets

    rows = [_row(f"problem {i}") for i in range(30)] + [_row(BENCH[0])]

    def fake_load_dataset(name, config=None, *args, split=None, streaming=False, revision=None, **kwargs):
        if streaming:
            assert (name, split, revision) == (build_math.OMR_REPO, "cot", build_math.OMR_REVISION)
            return datasets.Dataset.from_list(rows).to_iterable_dataset(num_shards=2)
        problems = BENCH if name == "HuggingFaceH4/MATH-500" else ["unrelated text"]
        return datasets.Dataset.from_dict({field: problems for field in ("problem", "Problem", "question")})

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    build_math.main(
        ["--tokenizer", "toy", "--max_seq_len", "100", "--rows", "1000", "--eval_rows", "4", "--out_dir", str(tmp_path)]
    )
    read = lambda name: [json.loads(line)["messages"] for line in (tmp_path / name).read_text().splitlines()]  # noqa: E731
    train, ev = read("train.jsonl"), read("eval.jsonl")
    assert len(train) == 26 and len(ev) == 4
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["counts"]["contaminated"] == 1 and stats["train"] == 26
