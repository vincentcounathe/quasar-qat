import pytest

from quasar.eval import mathbench as M


def test_gsm8k_gold():
    assert M._gsm8k_gold({"answer": "so 3 + 4 = 7\n#### 1,234"}) == "1234"


def test_last_boxed():
    assert M.last_boxed("so \\boxed{1} then \\boxed{2}") == "2"
    assert M.last_boxed("x = \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert M.last_boxed("\\boxed{a{b{c}}d}") == "a{b{c}}d"
    assert M.last_boxed("earlier \\boxed{5}, then \\boxed{\\frac{1}{2") == "5"  # unclosed final box
    assert M.last_boxed("\\boxed 56") is None
    assert M.last_boxed("\\boxed{}") == ""
    assert M.last_boxed("no box") is None and M.last_boxed("") is None


def test_extract_answer_reads_after_thinking():
    assert M.extract_answer("<think>\\boxed{999}</think>Thus \\boxed{-\\frac{1}{21}}.") == "-\\frac{1}{21}"
    assert M.extract_answer("<think>maybe \\boxed{3}</think> so it is 4") is None
    assert M.extract_answer("<think>deep... so \\boxed{7} maybe") == "7"  # trace never closed


def test_normalize():
    assert M.normalize(" $420,261$ ") == "420261"
    assert M.normalize("\\dfrac{1}{2}") == "\\frac{1}{2}"
    assert M.normalize("48.") == "48"
    assert M.normalize("{48}") == "48"
    assert M.normalize("\\left(1,2\\right)") == "(12)"


@pytest.fixture
def mv():
    pytest.importorskip("math_verify")


@pytest.mark.parametrize(
    "gold,sample,ok",
    [
        ("48", "reasoning...\n\nThe answer is \\boxed{48}.", True),
        ("48", "The answer is \\boxed{49}.", False),
        ("-\\frac{1}{21}", "<think>\\boxed{999}</think>Thus \\boxed{-\\dfrac{1}{21}}.", True),
        ("\\frac{3}{4}", "<think>x</think>The probability is \\boxed{0.75}.", True),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "<think>x</think> \\boxed{(3, \\frac{\\pi}{2})}", True),
        ("4", "<think>maybe \\boxed{3}</think> final \\boxed{4}", True),
        ("3", "<think>maybe \\boxed{3}</think> final \\boxed{4}", False),  # answers inside thinking never count
        ("7", "<think>deep... so \\boxed{7} maybe", True),  # an unclosed trace is read whole
        ("420261", "<think>x</think>The count is 420,261.", True),  # no box: Math-Verify reads the answer span
    ],
)
def test_is_correct(mv, gold, sample, ok):
    assert M.is_correct(gold, sample) is ok


def test_unparsable_gold_falls_back_to_exact_match(mv):
    gold = "\\text{Monday}"
    assert M.is_correct(gold, "<think>x</think> \\boxed{\\text{Monday}}")
    assert not M.is_correct(gold, "<think>x</think> \\boxed{\\text{Tuesday}}")


def test_judge(mv):
    assert M.judge({"gold": "2"}, "<think>1+1</think> \\boxed{2}") == ("2", True)
