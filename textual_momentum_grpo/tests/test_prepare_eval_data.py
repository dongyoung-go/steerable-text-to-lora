from scripts.prepare_eval_data import _sample_jsonl


def test_sample_jsonl_returns_all_rows_when_fewer_than_n(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    dst = tmp_path / "dst.jsonl"

    n = _sample_jsonl(src, dst, n=200, seed=0)

    assert n == 3
    assert dst.read_text() == src.read_text()


def test_sample_jsonl_samples_fixed_size_and_is_seed_deterministic(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("".join(f'{{"a": {i}}}\n' for i in range(1000)))
    dst1 = tmp_path / "dst1.jsonl"
    dst2 = tmp_path / "dst2.jsonl"

    n1 = _sample_jsonl(src, dst1, n=200, seed=0)
    n2 = _sample_jsonl(src, dst2, n=200, seed=0)

    assert n1 == n2 == 200
    assert dst1.read_text() == dst2.read_text()  # same seed -> same sample


def test_sample_jsonl_different_seed_differs(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("".join(f'{{"a": {i}}}\n' for i in range(1000)))
    dst1 = tmp_path / "dst1.jsonl"
    dst2 = tmp_path / "dst2.jsonl"

    _sample_jsonl(src, dst1, n=200, seed=0)
    _sample_jsonl(src, dst2, n=200, seed=1)

    assert dst1.read_text() != dst2.read_text()
