import json

from scripts.check_overlap import _load_problem_texts, _normalize


def test_normalize_collapses_whitespace_and_lowercases():
    assert _normalize("  What   is\n2+2?  ") == "what is 2+2?"


def test_load_problem_texts_extracts_last_user_turn(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {"prompt": [{"role": "user", "content": "problem A"}], "label": "1"},
        {
            "prompt": [
                {"role": "user", "content": "ignored earlier turn"},
                {"role": "user", "content": "problem B"},
            ],
            "label": "2",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    texts = _load_problem_texts(path)

    assert texts == ["problem A", "problem B"]


def test_load_problem_texts_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {"prompt": [{"role": "user", "content": "x"}], "label": "1"}
    path.write_text(f"\n{json.dumps(row)}\n\n")

    assert _load_problem_texts(path) == ["x"]
