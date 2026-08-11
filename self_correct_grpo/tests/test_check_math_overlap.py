import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_math_overlap.py"


def _write_jsonl(path: Path, problems: list[str]) -> None:
    with path.open("w") as f:
        for i, problem in enumerate(problems):
            f.write(json.dumps({"prompt": [{"role": "user", "content": problem}], "label": str(i)}) + "\n")


def test_no_overlap_exits_zero(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    _write_jsonl(train, ["What is 2+2?", "Solve for x: x^2 = 4."])
    _write_jsonl(eval_, ["What is 3+3?"])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(train), "--eval", str(eval_)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "OK: no exact-string overlap." in result.stdout


def test_overlap_exits_nonzero(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    shared = "What is 2+2?"
    _write_jsonl(train, [shared, "Solve for x: x^2 = 4."])
    _write_jsonl(eval_, [shared])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(train), "--eval", str(eval_)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stderr


def test_overlap_is_whitespace_and_case_insensitive(tmp_path):
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    _write_jsonl(train, ["What   is 2+2?\n"])
    _write_jsonl(eval_, ["what is 2+2?"])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(train), "--eval", str(eval_)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
