from scripts.prepare_math_train import _extract_boxed_answer


def test_extract_boxed_answer_simple():
    assert _extract_boxed_answer("Therefore the answer is $\\boxed{7}$.") == "7"


def test_extract_boxed_answer_nested():
    assert _extract_boxed_answer("So $x = \\boxed{\\frac{3}{4}}$.") == "\\frac{3}{4}"


def test_extract_boxed_answer_last_occurrence():
    text = "Draft: \\boxed{1}. Final: \\boxed{2}."
    assert _extract_boxed_answer(text) == "2"


def test_extract_boxed_answer_missing_returns_none():
    assert _extract_boxed_answer("no boxed answer") is None


def test_extract_boxed_answer_unterminated_returns_none():
    assert _extract_boxed_answer("\\boxed{unterminated") is None
