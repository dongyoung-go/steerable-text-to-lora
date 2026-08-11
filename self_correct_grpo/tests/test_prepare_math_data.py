import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_math_data.py"


def test_adds_data_source_metadata(tmp_path):
    train_src = tmp_path / "train_raw.jsonl"
    eval_src = tmp_path / "eval_raw.jsonl"
    out_dir = tmp_path / "prepared"

    train_src.write_text(json.dumps({"prompt": [{"role": "user", "content": "2+2?"}], "label": "4"}) + "\n")
    eval_src.write_text(json.dumps({"prompt": [{"role": "user", "content": "3+3?"}], "label": "6"}) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train_src),
            "--eval",
            str(eval_src),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    train_rows = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()]
    eval_rows = [json.loads(line) for line in (out_dir / "eval.jsonl").read_text().splitlines()]

    assert train_rows == [
        {"prompt": [{"role": "user", "content": "2+2?"}], "label": "4", "metadata": {"data_source": "math"}}
    ]
    assert eval_rows == [
        {"prompt": [{"role": "user", "content": "3+3?"}], "label": "6", "metadata": {"data_source": "math"}}
    ]


def test_preserves_existing_metadata_fields(tmp_path):
    src = tmp_path / "in.jsonl"
    out_dir = tmp_path / "prepared"
    src.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "x"}],
                "label": "y",
                "metadata": {"subset": "algebra"},
            }
        )
        + "\n"
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(src), "--eval", str(src), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    row = json.loads((out_dir / "train.jsonl").read_text().splitlines()[0])
    assert row["metadata"] == {"subset": "algebra", "data_source": "math"}
