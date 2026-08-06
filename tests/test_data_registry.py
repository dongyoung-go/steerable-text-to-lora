"""Glob-based task discovery. See docs/03_training_validation.md §1."""

from __future__ import annotations

import json
import warnings

import pytest
import yaml

from steerable_t2l.data.registry import discover_tasks, domains


def _write_task(root, name, *, domain=None, n_rows=3, n_desc=1):
    task_dir = root / name
    task_dir.mkdir()
    jsonl_path = task_dir / f"{name}.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"question": f"q{i}", "response": f"r{i}"}) + "\n")

    metadata = {
        "descriptions": [f"desc {i}" for i in range(n_desc)],
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "response",
        "system_message": "",
        "user_prompt_template": "{question}",
    }
    if domain is not None:
        metadata["domain"] = domain
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)
    return task_dir


def test_discover_tasks_by_glob(tmp_path):
    _write_task(tmp_path, "math_00", domain="math")
    _write_task(tmp_path, "math_01", domain="math")
    _write_task(tmp_path, "code_00", domain="code")

    tasks = discover_tasks(tmp_path, ["math_*"])
    assert [t.name for t in tasks] == ["math_00", "math_01"]


def test_discover_tasks_dedups_overlapping_patterns(tmp_path):
    _write_task(tmp_path, "math_00")
    tasks = discover_tasks(tmp_path, ["math_*", "math_0*"])
    assert len(tasks) == 1


def test_discover_tasks_skips_dir_without_metadata(tmp_path):
    (tmp_path / "not_a_task").mkdir()
    _write_task(tmp_path, "math_00")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tasks = discover_tasks(tmp_path, ["*"])
        assert any("metadata.yaml" in str(warning.message) for warning in w)
    assert [t.name for t in tasks] == ["math_00"]


def test_domains_groups_by_metadata_domain(tmp_path):
    _write_task(tmp_path, "math_00", domain="math")
    _write_task(tmp_path, "code_00", domain="code")
    _write_task(tmp_path, "other_00")

    tasks = discover_tasks(tmp_path, ["*"])
    grouped = domains(tasks)
    assert [t.name for t in grouped["math"]] == ["math_00"]
    assert None in grouped
    assert [t.name for t in grouped[None]] == ["other_00"]


def test_metadata_rejects_nonempty_system_message(tmp_path):
    task_dir = _write_task(tmp_path, "bad_00")
    meta = yaml.safe_load((task_dir / "metadata.yaml").read_text())
    meta["system_message"] = "not empty"
    (task_dir / "metadata.yaml").write_text(yaml.safe_dump(meta))

    from steerable_t2l.data.metadata import TaskMetadata

    with pytest.raises(ValueError, match="system_message"):
        TaskMetadata.from_yaml(task_dir / "metadata.yaml")
