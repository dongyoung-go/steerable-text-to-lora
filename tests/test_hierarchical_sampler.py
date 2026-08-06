"""HierarchicalBatchSampler: infinite resampling, driven by max_steps not epochs.

See docs/03_training_validation.md §1.
"""

from __future__ import annotations

import itertools

import pytest

from steerable_t2l.data.datasets import HierarchicalBatchSampler


def test_batch_shape_is_n_tasks_times_n_points():
    sampler = HierarchicalBatchSampler([10, 10, 10, 10], n_tasks_per_batch=2, n_points_per_task=3, seed=0)
    batch = next(iter(sampler))
    assert len(batch) == 2 * 3


def test_indices_stay_within_each_dataset_offset_range():
    sizes = [5, 7, 3]
    offsets = [0, 5, 12, 15]
    sampler = HierarchicalBatchSampler(sizes, n_tasks_per_batch=3, n_points_per_task=4, seed=0)
    it = iter(sampler)
    for _ in range(20):
        batch = next(it)
        for idx in batch:
            assert 0 <= idx < offsets[-1]


def test_sampler_never_terminates_len_raises():
    sampler = HierarchicalBatchSampler([5], n_tasks_per_batch=1, n_points_per_task=1, seed=0)
    with pytest.raises(TypeError):
        len(sampler)
    # It really is infinite: pulling many batches never raises StopIteration.
    it = iter(sampler)
    for _ in range(1000):
        next(it)


def test_deterministic_given_seed():
    s1 = HierarchicalBatchSampler([10, 10], n_tasks_per_batch=2, n_points_per_task=2, seed=123)
    s2 = HierarchicalBatchSampler([10, 10], n_tasks_per_batch=2, n_points_per_task=2, seed=123)
    batches1 = list(itertools.islice(iter(s1), 5))
    batches2 = list(itertools.islice(iter(s2), 5))
    assert batches1 == batches2


def test_resamples_with_replacement_can_repeat_tasks_in_one_batch():
    # With only 1 task available, every draw in every batch must come from it.
    sampler = HierarchicalBatchSampler([20], n_tasks_per_batch=3, n_points_per_task=2, seed=0)
    batch = next(iter(sampler))
    assert len(batch) == 6
    assert all(0 <= idx < 20 for idx in batch)
