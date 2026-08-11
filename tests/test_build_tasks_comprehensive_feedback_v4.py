"""scripts/build_tasks_from_comprehensive_feedback_v4.py: joins forward_outputs.jsonl val rows
to a comprehensive-feedback chain by iteration and groups by feedback text, mirroring
build_tasks_from_textgrad_repro_v3.py's prompt-text grouping but keyed by feedback instead."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_tasks_from_comprehensive_feedback_v4.py"
    spec = importlib.util.spec_from_file_location("build_tasks_from_comprehensive_feedback_v4", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_tasks_from_comprehensive_feedback_v4"] = module
    spec.loader.exec_module(module)
    return module


build_tasks = _load_module()


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_group_val_rows_by_feedback_skips_non_val_and_missing_iterations(tmp_path):
    forward_outputs = tmp_path / "forward_outputs.jsonl"
    _write_jsonl(forward_outputs, [
        {"iteration": -1, "split": "val", "question": "q_baseline", "model_response": "r0",
         "gold_answer": "a", "correct": True},
        {"iteration": 0, "split": "train", "question": "q_train", "model_response": "rt",
         "gold_answer": "a", "correct": True},
        {"iteration": 0, "split": "val", "question": "q0", "model_response": "r1",
         "gold_answer": "a", "correct": True},
        {"iteration": 1, "split": "val", "question": "q1", "model_response": "r2",
         "gold_answer": "a", "correct": True},
    ])
    # iteration -1 (baseline) has no feedback entry -- must be skipped, not KeyError.
    feedback_by_iteration = {0: "feedback A", 1: "feedback A"}

    groups, order = build_tasks.group_val_rows_by_feedback(forward_outputs, feedback_by_iteration)

    assert order == ["feedback A"]
    assert len(groups["feedback A"]) == 2
    assert {r["question"] for r in groups["feedback A"]} == {"q0", "q1"}


def test_reverted_round_pools_with_parent_via_identical_feedback_text(tmp_path):
    # Simulates the exact scenario generate_comprehensive_feedback_v4.py produces: iteration 0 is
    # accepted (feedback advances to "A"), iteration 1 is reverted (feedback stays "A", byte-
    # identical to iteration 0's) -- these must land in ONE group, not two.
    forward_outputs = tmp_path / "forward_outputs.jsonl"
    _write_jsonl(forward_outputs, [
        {"iteration": 0, "split": "val", "question": "q0", "model_response": "correct 0",
         "gold_answer": "42", "correct": True},
        {"iteration": 1, "split": "val", "question": "q1", "model_response": "correct 1",
         "gold_answer": "42", "correct": True},
        {"iteration": 2, "split": "val", "question": "q2", "model_response": "correct 2",
         "gold_answer": "42", "correct": True},
    ])
    feedback_by_iteration = {0: "A", 1: "A", 2: "B"}

    groups, order = build_tasks.group_val_rows_by_feedback(forward_outputs, feedback_by_iteration)

    assert order == ["A", "B"]
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 1


def test_build_one_writes_task_dir_with_feedback_as_description(tmp_path):
    src_dir = tmp_path / "src"
    forward_outputs = src_dir / "forward_outputs.jsonl"
    _write_jsonl(forward_outputs, [
        {"iteration": 0, "split": "val", "question": f"q{i}", "model_response": f"r{i}",
         "gold_answer": "42", "correct": True}
        for i in range(3)
    ])
    feedback_path = tmp_path / "feedback" / "comprehensive_feedback_v4.jsonl"
    _write_jsonl(feedback_path, [{"iteration": 0, "comprehensive_feedback": "generalized guidance text"}])

    jsonl_out = tmp_path / "jsonl_out"
    tasks_out = tmp_path / "tasks_out"

    summaries = build_tasks.build_one(
        src_dir, feedback_path, "gsm8k", jsonl_out, tasks_out,
        filter_correct=True, min_samples=1,
    )

    assert len(summaries) == 1
    assert summaries[0]["dropped_min_samples"] is False

    task_dir = tasks_out / "comprehensive_feedback_v4_gsm8k_d0"
    metadata = yaml.safe_load((task_dir / "metadata.yaml").read_text())
    assert metadata["descriptions"] == ["generalized guidance text"]
    assert metadata["response_field"] == "response"
    assert metadata["domain"] == "gsm8k"

    rows = [json.loads(line) for line in open(jsonl_out / "gsm8k_d0.jsonl")]
    assert len(rows) == 3
    assert {r["question"] for r in rows} == {"q0", "q1", "q2"}


def test_build_one_drops_group_with_empty_feedback_text(tmp_path):
    src_dir = tmp_path / "src"
    forward_outputs = src_dir / "forward_outputs.jsonl"
    _write_jsonl(forward_outputs, [
        {"iteration": 0, "split": "val", "question": "q0", "model_response": "r0",
         "gold_answer": "42", "correct": True},
    ])
    # Simulates round-0-reverted: no feedback ever accumulated -- generate_comprehensive_feedback_v4.py
    # would still emit a row (reverted=True, comprehensive_feedback="") but
    # group_val_rows_by_feedback's own `if not feedback: continue` guard excludes empty feedback
    # text from ever forming a group.
    feedback_path = tmp_path / "feedback" / "comprehensive_feedback_v4.jsonl"
    _write_jsonl(feedback_path, [{"iteration": 0, "comprehensive_feedback": ""}])

    groups, order = build_tasks.group_val_rows_by_feedback(
        forward_outputs, build_tasks.load_feedback_by_iteration(feedback_path)
    )

    assert groups == {}
    assert order == []


def test_build_one_min_samples_drops_group(tmp_path):
    src_dir = tmp_path / "src"
    forward_outputs = src_dir / "forward_outputs.jsonl"
    _write_jsonl(forward_outputs, [
        {"iteration": 0, "split": "val", "question": "q0", "model_response": "r0",
         "gold_answer": "42", "correct": True},
    ])
    feedback_path = tmp_path / "feedback" / "comprehensive_feedback_v4.jsonl"
    _write_jsonl(feedback_path, [{"iteration": 0, "comprehensive_feedback": "guidance"}])

    summaries = build_tasks.build_one(
        src_dir, feedback_path, "gsm8k", tmp_path / "jsonl_out", tmp_path / "tasks_out",
        filter_correct=True, min_samples=5,
    )

    assert len(summaries) == 1
    assert summaries[0]["dropped_min_samples"] is True
    assert not (tmp_path / "tasks_out" / "comprehensive_feedback_v4_gsm8k_d0").exists()
