from scripts.prepare_openr1_train import row_to_record, sample_rows


def test_row_to_record_converts_problem_and_answer():
    row = {"problem": "What is 2+2?", "answer": "4", "solution": "...", "generations": ["..."]}

    assert row_to_record(row) == {"prompt": [{"role": "user", "content": "What is 2+2?"}], "label": "4"}


def test_row_to_record_strips_whitespace():
    row = {"problem": "  2+2?  ", "answer": "  4  "}

    assert row_to_record(row) == {"prompt": [{"role": "user", "content": "2+2?"}], "label": "4"}


def test_row_to_record_none_when_problem_blank():
    assert row_to_record({"problem": "  ", "answer": "4"}) is None


def test_row_to_record_none_when_answer_blank():
    assert row_to_record({"problem": "2+2?", "answer": ""}) is None


def test_row_to_record_none_when_answer_missing():
    assert row_to_record({"problem": "2+2?"}) is None


def test_sample_rows_returns_all_when_no_limit():
    rows = [1, 2, 3]
    assert sample_rows(rows, limit=None, seed=0) == rows


def test_sample_rows_returns_all_when_limit_exceeds_size():
    rows = [1, 2, 3]
    assert sample_rows(rows, limit=10, seed=0) == rows


def test_sample_rows_subsamples_to_limit():
    rows = list(range(100))
    sampled = sample_rows(rows, limit=10, seed=0)
    assert len(sampled) == 10
    assert set(sampled) <= set(rows)


def test_sample_rows_deterministic_for_same_seed():
    rows = list(range(100))
    assert sample_rows(rows, limit=10, seed=0) == sample_rows(rows, limit=10, seed=0)


def test_sample_rows_differs_across_seeds():
    rows = list(range(100))
    assert sample_rows(rows, limit=10, seed=0) != sample_rows(rows, limit=10, seed=1)
