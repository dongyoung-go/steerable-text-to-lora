"""Q/D/T split determinism and D-axis graceful degradation. See docs/03_training_validation.md §4."""

from __future__ import annotations

import warnings

from steerable_t2l.data.metadata import TaskMetadata
from steerable_t2l.data.registry import Task
from steerable_t2l.data.splits import d_axis_available, make_splits, resolve_q_holdout


def _task(name, n_desc=1, domain=None):
    metadata = TaskMetadata(
        descriptions=tuple(f"desc {i}" for i in range(n_desc)),
        ds_kwargs={"path": "json", "data_files": f"{name}.jsonl", "split": "train"},
        response_field="response",
        user_prompt_template="{question}",
        domain=domain,
    )
    return Task(name=name, dir=None, metadata=metadata)


def test_make_splits_is_deterministic():
    tasks = [_task(f"t{i}", n_desc=2) for i in range(10)]
    s1 = make_splits(tasks, t_frac=0.2, q_frac=0.1, seed=42)
    s2 = make_splits(tasks, t_frac=0.2, q_frac=0.1, seed=42)
    assert s1.to_dict() == s2.to_dict()


def test_make_splits_different_seed_differs():
    tasks = [_task(f"t{i}", n_desc=2) for i in range(10)]
    s1 = make_splits(tasks, t_frac=0.2, q_frac=0.1, seed=0)
    s2 = make_splits(tasks, t_frac=0.2, q_frac=0.1, seed=1)
    assert s1.t_holdout != s2.t_holdout or s1.d_holdout != s2.d_holdout


def test_t_holdout_excludes_from_d_holdout():
    tasks = [_task(f"t{i}", n_desc=2) for i in range(10)]
    splits = make_splits(tasks, t_frac=0.3, q_frac=0.1, seed=0)
    assert set(splits.t_holdout).isdisjoint(splits.d_holdout.keys())


def test_single_description_task_degrades_gracefully():
    tasks = [_task("only_one", n_desc=1)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        splits = make_splits(tasks, t_frac=0.0, q_frac=0.1, seed=0)
        assert any("only 1 description" in str(warning.message) for warning in w)
    assert splits.d_holdout["only_one"] == []
    assert not d_axis_available(splits, "only_one")


def test_multi_description_task_holds_out_at_least_one():
    tasks = [_task("many", n_desc=4)]
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.1, seed=0)
    assert len(splits.d_holdout["many"]) >= 1
    assert d_axis_available(splits, "many")


def test_resolve_q_holdout_is_deterministic_and_bounded():
    tasks = [_task("t0", n_desc=2)]
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.2, seed=7)
    idx1 = resolve_q_holdout(splits, "t0", n_rows=50)
    idx2 = resolve_q_holdout(splits, "t0", n_rows=50)
    assert idx1 == idx2
    assert all(0 <= i < 50 for i in idx1)
    assert len(idx1) == round(50 * 0.2)


def test_resolve_q_holdout_empty_for_zero_rows():
    tasks = [_task("t0")]
    splits = make_splits(tasks, t_frac=0.0, q_frac=0.1, seed=0)
    assert resolve_q_holdout(splits, "t0", n_rows=0) == []
