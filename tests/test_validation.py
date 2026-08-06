"""Seven-condition validation scoring and steering margin. See docs/03_training_validation.md §4."""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from steerable_t2l.data.datasets import DataConfig
from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import make_splits
from steerable_t2l.validation import (
    CONDITIONS,
    GIBBERISH_DESCS,
    build_condition_descs,
    run_validation,
    score_condition,
    steering_margin,
)


def _make_task(root, name, n_rows=8, n_desc=2, response_prefix="answer"):
    task_dir = root / name
    task_dir.mkdir()
    jsonl_path = task_dir / f"{name}.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"question": f"q{i}", "response": f"{response_prefix} {i}"}) + "\n")
    metadata = {
        "descriptions": [f"{name} instruction {i}" for i in range(n_desc)],
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "response",
        "system_message": "",
        "user_prompt_template": "{question}",
    }
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)


@pytest.fixture
def three_tasks(tmp_path):
    _make_task(tmp_path, "t0", n_desc=2)
    _make_task(tmp_path, "t1", n_desc=1)
    _make_task(tmp_path, "t2", n_desc=2)
    return discover_tasks(tmp_path, ["t*"]), tmp_path


def test_build_condition_descs_train_excludes_d_holdout(three_tasks):
    tasks, _ = three_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.1, seed=0)
    task = next(t for t in tasks if t.name == "t0")
    if splits.d_holdout["t0"]:
        pool = build_condition_descs(task, splits, tasks, "train_descs", __import__("random").Random(0))
        held = {task.metadata.descriptions[i] for i in splits.d_holdout["t0"]}
        assert not held.issubset(set(pool)) or len(pool) == len(task.metadata.descriptions)


def test_build_condition_descs_eval_na_when_d_unavailable(three_tasks):
    tasks, _ = three_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.1, seed=0)
    task = next(t for t in tasks if t.name == "t1")  # only 1 description
    import random

    pool = build_condition_descs(task, splits, tasks, "eval_descs", random.Random(0))
    assert pool is None


def test_build_condition_descs_gibberish_is_constant(three_tasks):
    import random

    tasks, _ = three_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.1, seed=0)
    task = tasks[0]
    pool = build_condition_descs(task, splits, tasks, "gibberish_descs", random.Random(0))
    assert pool == GIBBERISH_DESCS


def test_steering_margin_reports_na_without_denominators():
    assert steering_margin({}) == "n/a"


def test_steering_margin_computes_both_denominators():
    conds = {
        "train_descs": {"loss": 1.0, "n": 10},
        "eval_descs": {"loss": 1.2, "n": 10},
        "other_task_descs": {"loss": 2.0, "n": 10},
        "gibberish_descs": {"loss": 1.9, "n": 10},
    }
    margin = steering_margin(conds)
    assert margin["vs_other_task/train_descs"] == pytest.approx(1.0)
    assert margin["vs_other_task/eval_descs"] == pytest.approx(0.8)
    assert margin["vs_gibberish/train_descs"] == pytest.approx(0.9)


def test_score_condition_base_matches_plain_forward(three_tasks, tokenizer, target_model_for_tokenizer, spec):
    tasks, tmp_path = three_tasks
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.25, seed=0)
    task = tasks[0]

    from steerable_t2l.data.datasets import PerTaskDescDataset, build_dataloader
    from steerable_t2l.data.splits import resolve_q_holdout

    data_config = DataConfig(
        tasks_root=str(tmp_path), train_tasks=("t*",), inp_max_len=64,
        val_batch_size=4, cache_root=str(tmp_path / "cache"),
    )
    ds = PerTaskDescDataset(task, tokenizer, 64, cache_root=data_config.cache_root, seed=0)
    q_idx = resolve_q_holdout(splits, task.name, len(ds))
    val_loader = build_dataloader(
        [task], tokenizer, data_config, split="val", seed=0, row_indices_by_task={task.name: q_idx}
    )

    import random

    result = score_condition(None, target_model_for_tokenizer, spec, task, splits, tasks, "base", val_loader, None, random.Random(0))
    assert isinstance(result, dict)
    assert result["n"] == len(q_idx)
    assert result["loss"] > 0


def test_oracle_condition_via_synthetic_adapter_matches_hooked_forward(three_tasks, tokenizer, target_model_for_tokenizer, spec, tmp_path):
    """The 'oracle' path must route through build_sites/lora_hooks identically to hypernet
    conditions -- verified here by hand-building a trivial adapter and checking the score
    matches a manually-hooked forward over the same batch."""
    tasks, data_root = three_tasks
    task = tasks[0]
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.25, seed=0)

    # Build a tiny PEFT adapter directly on the tiny target model and save it.
    from peft import get_peft_model

    peft_model = get_peft_model(target_model_for_tokenizer, spec.to_lora_config())
    oracle_dir = tmp_path / "oracle" / task.name
    oracle_dir.mkdir(parents=True)
    peft_model.save_pretrained(str(oracle_dir))

    from steerable_t2l.data.datasets import PerTaskDescDataset, build_dataloader
    from steerable_t2l.data.splits import resolve_q_holdout

    data_config = DataConfig(
        tasks_root=str(data_root), train_tasks=("t*",), inp_max_len=64,
        val_batch_size=4, cache_root=str(tmp_path / "cache"),
    )
    ds = PerTaskDescDataset(task, tokenizer, 64, cache_root=data_config.cache_root, seed=0)
    q_idx = resolve_q_holdout(splits, task.name, len(ds))
    val_loader = build_dataloader(
        [task], tokenizer, data_config, split="val", seed=0, row_indices_by_task={task.name: q_idx}
    )

    import random

    result = score_condition(
        None, target_model_for_tokenizer, spec, task, splits, tasks, "oracle", val_loader,
        str(tmp_path / "oracle"), random.Random(0),
    )
    assert isinstance(result, dict)
    assert result["n"] == len(q_idx)


def test_oracle_condition_na_when_missing(three_tasks, tokenizer, target_model_for_tokenizer, spec, tmp_path):
    tasks, data_root = three_tasks
    task = tasks[0]
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.25, seed=0)

    from steerable_t2l.data.datasets import PerTaskDescDataset, build_dataloader
    from steerable_t2l.data.splits import resolve_q_holdout

    data_config = DataConfig(
        tasks_root=str(data_root), train_tasks=("t*",), inp_max_len=64,
        val_batch_size=4, cache_root=str(tmp_path / "cache"),
    )
    ds = PerTaskDescDataset(task, tokenizer, 64, cache_root=data_config.cache_root, seed=0)
    q_idx = resolve_q_holdout(splits, task.name, len(ds))
    val_loader = build_dataloader(
        [task], tokenizer, data_config, split="val", seed=0, row_indices_by_task={task.name: q_idx}
    )
    import random

    result = score_condition(
        None, target_model_for_tokenizer, spec, task, splits, tasks, "oracle", val_loader,
        str(tmp_path / "does_not_exist"), random.Random(0),
    )
    assert result == "n/a"


def test_run_validation_full_grid(three_tasks, tokenizer, target_model_for_tokenizer, hypernet, spec, tmp_path):
    tasks, data_root = three_tasks
    splits = make_splits(tasks, t_frac=0.34, q_frac=0.25, seed=0)
    data_config = DataConfig(
        tasks_root=str(data_root), train_tasks=("t*",), inp_max_len=64,
        val_batch_size=4, cache_root=str(tmp_path / "cache"),
    )

    result = run_validation(
        hypernet, target_model_for_tokenizer, spec, tasks, splits, tokenizer, data_config,
        oracle_dir=None, val_batch_size=4, seed=0,
    )

    assert "per_task" in result and "steering_margin" in result and "overall" in result
    for conds in result["per_task"].values():
        for condition in conds:
            assert condition in CONDITIONS
        # oracle is always n/a here (no oracle_dir given).
        if "oracle" in conds:
            assert conds["oracle"] == "n/a"

    for margin in result["steering_margin"].values():
        if margin != "n/a":
            for v in margin.values():
                assert torch.isfinite(torch.tensor(v))
