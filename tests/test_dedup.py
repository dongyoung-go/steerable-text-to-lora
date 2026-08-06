"""Batch-level description deduplication is numerically free.

Within a training batch every sample drawn from the same task shares a description, so the
backbone (and the heads) should run on the unique instructions only. That optimization is
worth doing precisely because it is exact -- this test is what makes it safe to rely on.
"""

from __future__ import annotations

import torch

from steerable_t2l.hypernet import dedup
from tests.conftest import DESCS


def test_dedup_preserves_first_appearance_order():
    descs = ["b", "a", "b", "c", "a"]
    uniq, inverse = dedup(descs)
    assert uniq == ["b", "a", "c"]
    assert inverse.tolist() == [0, 1, 0, 2, 1]
    assert [uniq[i] for i in inverse.tolist()] == descs


def test_dedup_of_distinct_descs_is_identity():
    uniq, inverse = dedup(DESCS)
    assert uniq == DESCS
    assert inverse.tolist() == [0, 1, 2]


def test_deduped_forward_matches_naive(hypernet, spec):
    batch = [DESCS[0], DESCS[1], DESCS[0], DESCS[0], DESCS[2]]

    naive = hypernet(batch)
    deduped = hypernet.generate_for_batch(batch)

    for module in spec.target_modules:
        for i in range(2):
            torch.testing.assert_close(deduped[module][i], naive[module][i], rtol=1e-5, atol=1e-6)


def test_duplicate_rows_are_identical(hypernet, spec):
    batch = [DESCS[0], DESCS[1], DESCS[0]]
    out = hypernet.generate_for_batch(batch)
    for module in spec.target_modules:
        A, _ = out[module]
        assert torch.equal(A[0], A[2]), f"{module}: duplicated description gave different weights"


def test_deduped_backward_matches_naive(spec, tokenizer):
    """The index_select expansion must scatter-add gradients back into the unique rows."""
    from steerable_t2l.testing import tiny_hypernet

    batch = [DESCS[0], DESCS[1], DESCS[0], DESCS[0]]

    def grads(use_dedup: bool):
        torch.manual_seed(1234)
        model = tiny_hypernet(spec, tokenizer, zero_init=False, seed=0)
        model.zero_grad(set_to_none=True)
        out = model.generate_for_batch(batch) if use_dedup else model(batch)
        # A scalar that touches every sample and every module.
        loss = sum((A.pow(2).sum() + B.pow(2).sum()) for A, B in out.values())
        loss.backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    naive, deduped = grads(False), grads(True)
    assert set(naive) == set(deduped)
    for name in naive:
        torch.testing.assert_close(deduped[name], naive[name], rtol=1e-4, atol=1e-6, msg=name)
