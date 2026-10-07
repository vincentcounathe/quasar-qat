import json

import pytest

from quasar.data import prompts


def _conv(*turns):
    return [{"from": who, "value": text} for who, text in turns]


ROWS = [
    {
        "conversations": _conv(("human", "What is 2+2? Explain briefly."), ("gpt", "two and two make four. " * 4)),
        "source": "math",
    },
    {"conversations": _conv(("gpt", "no user turn")), "source": "chat"},
    {
        "conversations": _conv(
            ("system", "sys"),
            ("human", "  "),
            ("user", "Name a prime number please."),
            ("human", "again"),
            ("assistant", "short"),
        ),
        "source": "chat",
    },
    {"conversations": _conv(("human", "  What is 2+2? Explain briefly.\n"), ("gpt", "")), "source": "math"},
] + [
    {
        "conversations": _conv(("human", f"Question number {i}, tell me something."), ("gpt", f"answer {i} " * 20)),
        "source": "code",
    }
    for i in range(40)
]


@pytest.fixture
def opb(monkeypatch):
    import datasets

    def fake_load_dataset(name, *args, revision=None, **kwargs):
        assert name == prompts.OPB_REPO and revision == prompts.OPB_REVISION
        return datasets.Dataset.from_list(ROWS)

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)


def test_first_exchange():
    assert prompts.first_exchange(ROWS[0]["conversations"])[0] == "What is 2+2? Explain briefly."
    assert prompts.first_exchange(ROWS[1]["conversations"]) is None
    assert prompts.first_exchange(ROWS[2]["conversations"]) == ("Name a prime number please.", "short")
    assert prompts.first_exchange(_conv(("human", "only"))) == ("only", None)


def test_records_and_pair_bounds():
    assert prompts.to_record(7, ROWS[0]) == {"prompt": "What is 2+2? Explain briefly.", "source": "math", "orig_row": 7}
    assert prompts.to_record(0, ROWS[0], pairs=True)["messages"][1]["role"] == "assistant"
    assert prompts.to_record(2, ROWS[2], pairs=True) is None  # reply shorter than 64 characters
    assert prompts.to_record(3, ROWS[3], pairs=True) is None  # blank reply


def test_sample_is_seeded_nested_and_in_dataset_order(opb):
    big = prompts.sample(30, seed=5)
    assert big == prompts.sample(30, seed=5) and big != prompts.sample(30, seed=6)
    assert [r["orig_row"] for r in big] == sorted(r["orig_row"] for r in big)
    small = prompts.sample(10, seed=5)
    assert {r["orig_row"] for r in small} <= {r["orig_row"] for r in big}
    with pytest.raises(SystemExit):
        prompts.sample(len(ROWS), seed=5)  # more than the eligible rows


def test_exclusion_matches_stripped_prompt_text(opb, tmp_path):
    held_out = tmp_path / "eval.jsonl"
    held_out.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2? Explain briefly."},
                    {"role": "assistant", "content": "4"},
                ]
            }
        )
        + "\n"
        + json.dumps({"prompt": "Question number 3, tell me something."})
        + "\n"
    )
    out = tmp_path / "slice.jsonl"
    prompts.main(["--n", "40", "--exclude", str(held_out), "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["orig_row"] for r in rows} == {2} | set(range(4, 44)) - {7}  # rows 0 and 3 share the prompt


def test_pairs_cli(opb, tmp_path):
    out = tmp_path / "pairs.jsonl"
    prompts.main(["--pairs", "--n", "41", "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["orig_row"] for r in rows] == [0] + list(range(4, 44))
    assert all(set(r) == {"messages", "source", "orig_row"} for r in rows)
