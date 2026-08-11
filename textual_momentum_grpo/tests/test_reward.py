from tmgrpo.reward import check_answer, extract_boxed_answer


def test_extract_boxed_answer_simple():
    assert extract_boxed_answer("The answer is \\boxed{42}.") == "42"


def test_extract_boxed_answer_nested_braces():
    assert extract_boxed_answer("So x = \\boxed{\\frac{1}{2}} is the answer.") == "\\frac{1}{2}"


def test_extract_boxed_answer_last_occurrence():
    text = "First I thought \\boxed{1} but actually \\boxed{2}."
    assert extract_boxed_answer(text) == "2"


def test_extract_boxed_answer_missing():
    assert extract_boxed_answer("no boxed answer here") is None


def test_extract_boxed_answer_unbalanced_braces():
    assert extract_boxed_answer("\\boxed{unterminated") is None


def test_check_answer_exact_match():
    assert check_answer("The answer is \\boxed{5}.", "5") is True


def test_check_answer_equivalent_fraction():
    assert check_answer("\\boxed{1/2}", "\\frac{1}{2}") is True


def test_check_answer_mismatch():
    assert check_answer("\\boxed{5}", "6") is False


def test_check_answer_raw_prediction_without_boxed():
    # No boxed answer -> falls back to treating the whole prediction string as the candidate.
    assert check_answer("5", "5") is True
