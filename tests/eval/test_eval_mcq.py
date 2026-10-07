import pytest

from quasar.eval import mcq
from quasar.eval.mcq import extract_choice


@pytest.mark.parametrize(
    "text,n,expected",
    [
        # explicit statements; the last one wins
        ("The answer is (A). Hmm, no: the answer is (C).", 4, "C"),
        ("answer is B", 10, "B"),
        ("The answer is **D**.", 4, "D"),
        ("the answer is $\\boxed{J}$", 10, "J"),
        ("THE ANSWER IS: (E)", 10, "E"),
        ("Final Answer: C", 4, "C"),
        ("ANSWER: E", 10, "E"),
        ("Answer: **E**", 10, "E"),
        ("...reasoning...\nAnswer: F", 10, "F"),
        ("The correct answer is D.", 10, "D"),
        ("The correct option is (C).", 4, "C"),
        ("The correct letter choice is **B**.", 4, "B"),
        ("The answer should be (C).", 4, "C"),
        ("The best option is $\\boxed{B}$:", 10, "B"),
        ("### Final Answer  $$ \\boxed{D} $$", 4, "D"),
        ("\\boxed{\\text{(B)}}", 4, "B"),
        ("\\boxed{\\textbf{D}}", 4, "D"),
        ("\\boxed{(A)}", 4, "A"),
        ("\\boxed{B)}", 4, "B"),
        ("the answer is (A). Wait. \\boxed{C}", 4, "C"),
        ("\\boxed{C} ... no, the answer is (A).", 4, "A"),
        ("Answer: B. Option C was tempting (D)", 4, "B"),  # an explicit statement beats later mentions
        ("Answer: D. Option B is a common distractor.", 10, "D"),
        ("The correct answer is (B)", 4, "B"),  # LongBench-v2's official form
        # lowercase / articles / pronouns never name an option
        ("the answer is a matter of taste. Answer: C", 4, "C"),
        ("the answer is a bit tricky", 4, None),
        ("the answer is e.", 4, None),
        ("the answer is Bob", 4, None),
        ("Answer: I think the correct choice is (C).", 10, "C"),
        ("Final answer: I will go with (C).", 10, "C"),
        ("Answer: I see. (C).", 10, "C"),
        ("Final Answer: I", 10, "I"),
        ("The answer is A because the reaction proceeds via SN2.", 4, "A"),  # last resort
        # fallbacks in the closing characters
        ("I pick (B) then maybe (D)", 4, "D"),
        ("So it must be **Option C**.", 4, "C"),
        ("the answer is not (C) but (D)", 4, "D"),
        ("(A) " + "x" * 400, 4, None),
        ("x" * 400 + " (A) ", 4, "A"),
        # option count
        ("The answer is (J).", 4, None),
        ("The answer is (J). Also (B)", 4, "B"),
        ("\\boxed{J}", 4, None),
        ("nothing here", 4, None),
        ("", 4, None),
    ],
)
def test_extract_choice(text, n, expected):
    assert extract_choice(text, n) == expected


def test_extract_choice_reads_after_thinking():
    assert extract_choice("<think>I think the answer is (A)</think>\nThe answer is (B).", 4) == "B"
    assert extract_choice("<think>the answer is (A)</think>", 4) is None
    assert extract_choice("<think>maybe the answer is (C) or", 4) == "C"  # unclosed trace is read whole


def test_judge():
    item = {"gold": "B", "n_options": 4}
    assert mcq.judge(item, "<think>x</think> the answer is (B)") == ("B", True)
    assert mcq.judge(item, "Answer: C") == ("C", False)


# ------------------------------------------------------------------- prompts
VAL = [
    {
        "category": "math",
        "question": "1+1?",
        "options": ["1", "2"],
        "cot_content": "A: Let's think step by step. 1+1=2. The answer is (B).",
        "answer": "B",
    },
    {
        "category": "math",
        "question": "2+2?",
        "options": ["3", "4"],
        "cot_content": "A: Let's think step by step. 4. The answer is (B).",
        "answer": "B",
    },
]


def test_mmlu_pro_cot_example():
    assert mcq.format_cot_example(VAL[0], True) == (
        "Question:\n1+1?\nOptions:\nA. 1\nB. 2\nAnswer: Let's think step by step. 1+1=2. The answer is (B).\n\n"
    )
    assert mcq.format_cot_example(VAL[0], False).endswith("Options:\nA. 1\nB. 2\nAnswer: Let's think step by step.")


def test_mmlu_pro_prompt():
    p = mcq.mmlu_pro_prompt({"category": "math", "question": "3+3?", "options": ["5", "6", "7"]}, VAL)
    assert p.startswith(
        "The following are multiple choice questions (with answers) about math. Think step by step "
        'and then finish your answer with "the answer is (X)" where X is the correct letter '
        "choice.\n\n\n\nQuestion:\n"
    )
    assert p.count("Question:\n") == 3 and "A: Let's think" not in p
    assert p.endswith("Question:\n3+3?\nOptions:\nA. 5\nB. 6\nC. 7\nAnswer: Let's think step by step.")


def test_supergpqa_prompt():
    p = mcq.supergpqa_prompt("What is 2+2?", ["3", "4"])
    assert p.startswith("Answer the following multiple choice question.")
    assert "'Answer: $LETTER' (without quotes)" in p and "A, B, C, D, E, F, G, H, I, or J" in p
    assert p.endswith("What is 2+2?\nA) 3\nB) 4")


# -------------------------------------------------------- stratified subsample
def _items(spec):
    out = []
    for disc, n in spec.items():
        out += [{"id": f"u{len(out) + i:04d}", "discipline": disc} for i in range(n)]
    return out


def _counts(items):
    c = {}
    for it in items:
        c[it["discipline"]] = c.get(it["discipline"], 0) + 1
    return c


def test_subsample_deterministic_proportional_ordered():
    items = _items({"Science": 50, "Law": 30, "History": 20})
    a = mcq.stratified_subsample(items, 10, seed=1234)
    assert a == mcq.stratified_subsample(items, 10, seed=1234)
    assert a != mcq.stratified_subsample(items, 10, seed=999)
    assert _counts(a) == {"Science": 5, "Law": 3, "History": 2}
    assert [x["id"] for x in a] == sorted(x["id"] for x in a)


def test_subsample_largest_remainder():
    # 7 of {60, 25, 15}: quotas 4.2 / 1.75 / 1.05 -> 4 / 1 / 1, the spare one to the largest remainder.
    sub = mcq.stratified_subsample(_items({"A": 60, "B": 25, "C": 15}), 7, seed=3)
    assert _counts(sub) == {"A": 4, "B": 2, "C": 1}


def test_extract_choice_is_linear_on_degenerate_text():
    # Collapsed models emit long runs of whitespace / markup; the extractor must not backtrack on them.
    assert extract_choice("the answer is" + " " * 50000 + "x", 10) is None
    assert extract_choice("Answer:" + "*" * 50000 + "x", 10) is None
    assert extract_choice("\\boxed{" + " " * 50000 + "x", 10) is None
