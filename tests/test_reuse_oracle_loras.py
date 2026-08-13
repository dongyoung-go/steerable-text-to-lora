"""scripts/reuse_oracle_loras.py -- symlink one namespace's oracle LoRAs into another's."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from reuse_oracle_loras import main  # noqa: E402


def _write_task(root, name, n_rows=2):
    task_dir = root / name
    task_dir.mkdir()
    jsonl_path = task_dir / f"{name}.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"question": f"q{i}", "response": f"r{i}"}) + "\n")
    metadata = {
        "descriptions": ["desc 0"],
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "response",
        "system_message": "",
        "user_prompt_template": "{question}",
    }
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)
    return task_dir


def _write_oracle(root, name):
    oracle_dir = root / name
    oracle_dir.mkdir(parents=True)
    (oracle_dir / "adapter_model.safetensors").write_bytes(b"fake")
    return oracle_dir


def _write_canon(root, name):
    root.mkdir(parents=True, exist_ok=True)
    canon_path = root / f"{name}.pt"
    canon_path.write_bytes(b"fake")
    return canon_path


def test_links_matching_tasks(tmp_path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task(tasks_root, "textgrad_repro_v5_aqua_d0")
    _write_task(tasks_root, "gepa_repro_v5_aqua_d1")

    source_oracle = tmp_path / "oracle_loras_v3"
    source_canon = tmp_path / "oracle_loras_canon_v3"
    _write_oracle(source_oracle, "textgrad_repro_v3_aqua_d0")
    _write_canon(source_canon, "textgrad_repro_v3_aqua_d0")
    _write_oracle(source_oracle, "gepa_repro_v3_aqua_d1")
    _write_canon(source_canon, "gepa_repro_v3_aqua_d1")

    out_oracle = tmp_path / "oracle_loras_v5"
    out_canon = tmp_path / "oracle_loras_canon_v5"

    main_args = [
        "--tasks-root", str(tasks_root),
        "--train-tasks", "textgrad_repro_v5_*", "gepa_repro_v5_*",
        "--source-oracle-dir", str(source_oracle),
        "--source-canon-dir", str(source_canon),
        "--out-oracle-dir", str(out_oracle),
        "--out-canon-dir", str(out_canon),
        "--from-substr", "_v5_",
        "--to-substr", "_v3_",
    ]
    import contextlib
    import io

    old_argv = sys.argv
    sys.argv = ["reuse_oracle_loras.py", *main_args]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    link = out_oracle / "textgrad_repro_v5_aqua_d0"
    assert link.is_symlink()
    assert link.resolve() == (source_oracle / "textgrad_repro_v3_aqua_d0").resolve()
    canon_link = out_canon / "textgrad_repro_v5_aqua_d0.pt"
    assert canon_link.is_symlink()
    assert canon_link.resolve() == (source_canon / "textgrad_repro_v3_aqua_d0.pt").resolve()

    link2 = out_oracle / "gepa_repro_v5_aqua_d1"
    assert link2.resolve() == (source_oracle / "gepa_repro_v3_aqua_d1").resolve()


def test_missing_source_errors(tmp_path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task(tasks_root, "textgrad_repro_v5_aqua_d0")

    out_oracle = tmp_path / "oracle_loras_v5"
    out_canon = tmp_path / "oracle_loras_canon_v5"

    args = [
        "--tasks-root", str(tasks_root),
        "--train-tasks", "textgrad_repro_v5_*",
        "--source-oracle-dir", str(tmp_path / "oracle_loras_v3"),
        "--source-canon-dir", str(tmp_path / "oracle_loras_canon_v3"),
        "--out-oracle-dir", str(out_oracle),
        "--out-canon-dir", str(out_canon),
        "--from-substr", "_v5_",
        "--to-substr", "_v3_",
    ]
    import contextlib
    import io

    old_argv = sys.argv
    sys.argv = ["reuse_oracle_loras.py", *args]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main()
    finally:
        sys.argv = old_argv

    assert rc == 1
    assert not (out_oracle / "textgrad_repro_v5_aqua_d0").exists()


def test_splits_t_holdout_tasks_are_skipped_not_missing(tmp_path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task(tasks_root, "textgrad_repro_v5_aqua_d0")
    _write_task(tasks_root, "textgrad_repro_v5_aqua_d1")

    source_oracle = tmp_path / "oracle_loras_v3"
    source_canon = tmp_path / "oracle_loras_canon_v3"
    _write_oracle(source_oracle, "textgrad_repro_v3_aqua_d0")
    _write_canon(source_canon, "textgrad_repro_v3_aqua_d0")
    # No source oracle/canon for textgrad_repro_v3_aqua_d1 -- it's a T-holdout task, so
    # train_oracle_loras.py never trained one for it; that's expected, not an error.

    splits_path = tmp_path / "splits_v5.json"
    splits_path.write_text(
        json.dumps(
            {
                "q_frac": 0.1,
                "d_holdout": {},
                "t_holdout": ["textgrad_repro_v5_aqua_d1"],
                "seed": 0,
            }
        )
    )

    out_oracle = tmp_path / "oracle_loras_v5"
    out_canon = tmp_path / "oracle_loras_canon_v5"

    args = [
        "--tasks-root", str(tasks_root),
        "--train-tasks", "textgrad_repro_v5_*",
        "--splits", str(splits_path),
        "--source-oracle-dir", str(source_oracle),
        "--source-canon-dir", str(source_canon),
        "--out-oracle-dir", str(out_oracle),
        "--out-canon-dir", str(out_canon),
        "--from-substr", "_v5_",
        "--to-substr", "_v3_",
    ]
    import contextlib
    import io

    old_argv = sys.argv
    sys.argv = ["reuse_oracle_loras.py", *args]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert (out_oracle / "textgrad_repro_v5_aqua_d0").is_symlink()
    assert not (out_oracle / "textgrad_repro_v5_aqua_d1").exists()


def test_refuses_to_clobber_real_directory(tmp_path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task(tasks_root, "textgrad_repro_v5_aqua_d0")

    source_oracle = tmp_path / "oracle_loras_v3"
    source_canon = tmp_path / "oracle_loras_canon_v3"
    _write_oracle(source_oracle, "textgrad_repro_v3_aqua_d0")
    _write_canon(source_canon, "textgrad_repro_v3_aqua_d0")

    out_oracle = tmp_path / "oracle_loras_v5"
    out_canon = tmp_path / "oracle_loras_canon_v5"
    # A real (non-symlink) directory already at the destination -- must not be clobbered.
    (out_oracle / "textgrad_repro_v5_aqua_d0").mkdir(parents=True)

    args = [
        "--tasks-root", str(tasks_root),
        "--train-tasks", "textgrad_repro_v5_*",
        "--source-oracle-dir", str(source_oracle),
        "--source-canon-dir", str(source_canon),
        "--out-oracle-dir", str(out_oracle),
        "--out-canon-dir", str(out_canon),
        "--from-substr", "_v5_",
        "--to-substr", "_v3_",
    ]
    old_argv = sys.argv
    sys.argv = ["reuse_oracle_loras.py", *args]
    try:
        try:
            main()
        except FileExistsError:
            pass
        else:
            raise AssertionError("expected FileExistsError")
    finally:
        sys.argv = old_argv
