"""scripts/run_ablation.py: pure post-hoc comparison of two SFT checkpoints' histories."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_run_ablation_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ablation"] = module
    spec.loader.exec_module(module)
    return module


run_ablation = _load_run_ablation_module()


def _write_checkpoint(path, history):
    torch.save({"history": history, "stage": "sft", "step": history[-1]["step"] if history else 0}, path)


def test_latest_common_step_picks_max_shared_step():
    scratch = [{"step": 100}, {"step": 200}]
    warmstart = [{"step": 100}, {"step": 200}, {"step": 300}]
    assert run_ablation.latest_common_step(scratch, warmstart) == 200


def test_latest_common_step_none_when_disjoint():
    scratch = [{"step": 100}]
    warmstart = [{"step": 50}]
    assert run_ablation.latest_common_step(scratch, warmstart) is None


def test_compare_falls_back_to_last_entries_when_no_common_step():
    scratch = [{"step": 10, "steering_margin": {"t0": {"vs_gibberish/train_descs": 0.1}}}]
    warmstart = [{"step": 999, "steering_margin": {"t0": {"vs_gibberish/train_descs": 0.5}}}]
    result = run_ablation.compare(scratch, warmstart)
    assert result["step"] is None
    assert result["scratch"]["step"] == 10
    assert result["warmstart"]["step"] == 999


def test_compare_and_format_at_common_step():
    # steering_margin is per-task ({task_name: {denom: value} | "n/a"}), matching
    # validation.run_validation's real output shape -- NOT a flat {denom: value} dict.
    scratch = [{
        "step": 200,
        "steering_margin": {"t0": {"vs_gibberish/train_descs": 0.05}},
        "overall": {"base": 2.0},
    }]
    warmstart = [{
        "step": 200,
        "steering_margin": {"t0": {"vs_gibberish/train_descs": 0.4}},
        "overall": {"base": 2.0},
    }]
    result = run_ablation.compare(scratch, warmstart)
    assert result["step"] == 200
    assert "0.4000" in run_ablation.format_margin(result["warmstart"])
    assert "0.0500" in run_ablation.format_margin(result["scratch"])
    assert "base=2.0000" in run_ablation.format_overall(result["scratch"])


def test_format_margin_averages_across_tasks_and_skips_na():
    entry = {
        "steering_margin": {
            "t0": {"vs_gibberish/train_descs": 0.2, "vs_other_task/train_descs": 0.1},
            "t1": {"vs_gibberish/train_descs": 0.4},
            "t2": "n/a",  # e.g. a task with no other-task/gibberish comparison available
        }
    }
    formatted = run_ablation.format_margin(entry)
    # (0.2 + 0.4) / 2 == 0.3, averaged over the 2 tasks that actually have this denominator.
    assert "vs_gibberish/train_descs=0.3000 (n=2)" in formatted
    assert "vs_other_task/train_descs=0.1000 (n=1)" in formatted


def test_format_margin_reports_na_when_every_task_is_na():
    entry = {"steering_margin": {"t0": "n/a", "t1": "n/a"}}
    assert run_ablation.format_margin(entry) == "n/a"


def test_load_history_roundtrips_through_a_checkpoint_file(tmp_path):
    empty_path = tmp_path / "empty.pt"
    _write_checkpoint(empty_path, [])
    nonempty_path = tmp_path / "nonempty.pt"
    _write_checkpoint(nonempty_path, [{"step": 1, "steering_margin": {"t0": "n/a"}, "overall": {}}])

    assert run_ablation.load_history(str(empty_path)) == []
    assert len(run_ablation.load_history(str(nonempty_path))) == 1
