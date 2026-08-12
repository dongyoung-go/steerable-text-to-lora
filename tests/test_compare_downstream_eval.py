"""scripts/compare_downstream_eval.py -- cross-run diff of two downstream eval JSONs."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compare_downstream_eval import join_key, main  # noqa: E402


def _write_result(path, *, task_name, t2l_acc, base_acc):
    result = {
        "per_task": {
            task_name: {
                "t2l_train_desc": {"accuracy": t2l_acc, "n": 10, "n_correct": int(t2l_acc * 10)},
                "base": {"accuracy": base_acc, "n": 10, "n_correct": int(base_acc * 10)},
            }
        },
        "overall": {"t2l_train_desc": t2l_acc, "base": base_acc},
        "comparisons": {
            "per_task": {task_name: {"t2l_train_desc_minus_base": t2l_acc - base_acc}},
            "macro": {"t2l_train_desc_minus_base": t2l_acc - base_acc},
        },
    }
    path.write_text(json.dumps(result))


def test_join_key_strips_version_infix():
    assert join_key("textgrad_repro_v3_aqua_d9", "v3_", "v5_") == "textgrad_repro_aqua_d9"
    assert join_key("textgrad_repro_v5_aqua_d9", "v3_", "v5_") == "textgrad_repro_aqua_d9"


def test_compares_two_runs(tmp_path, capsys):
    file_a = tmp_path / "v3.json"
    file_b = tmp_path / "v5.json"
    _write_result(file_a, task_name="textgrad_repro_v3_aqua_d9", t2l_acc=0.3, base_acc=0.2)
    _write_result(file_b, task_name="textgrad_repro_v5_aqua_d9", t2l_acc=0.5, base_acc=0.2)

    old_argv = sys.argv
    sys.argv = [
        "compare_downstream_eval.py", str(file_a), str(file_b),
        "--labels", "v3", "v5",
    ]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    output = out.getvalue()
    assert "t2l_train_desc" in output
    assert "textgrad_repro_aqua_d9" in output
    assert "0.3000" in output and "0.5000" in output


def test_skips_tasks_without_a_match(tmp_path, capsys):
    file_a = tmp_path / "v3.json"
    file_b = tmp_path / "v5.json"
    _write_result(file_a, task_name="textgrad_repro_v3_aqua_d9", t2l_acc=0.3, base_acc=0.2)
    _write_result(file_b, task_name="textgrad_repro_v5_other_task_d0", t2l_acc=0.5, base_acc=0.2)

    old_argv = sys.argv
    sys.argv = ["compare_downstream_eval.py", str(file_a), str(file_b)]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert "aqua_d9" not in out.getvalue()
