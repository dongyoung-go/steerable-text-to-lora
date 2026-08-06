"""The (layer, module, role) <-> query-token index convention, and that the heads honour it.

A silent mismatch here would let every head read a plausible-looking but wrong token: the
model would train, the loss would go down, and every generated adapter would be assembled
from the wrong (layer, module) representations.
"""

from __future__ import annotations

import torch

from steerable_t2l.target_spec import N_ROLES, ROLE_A, ROLE_B


def test_index_is_a_bijection(spec):
    seen = {}
    for layer in range(spec.n_layers):
        for module in spec.target_modules:
            for role in (ROLE_A, ROLE_B):
                idx = spec.query_index(layer, module, role)
                assert idx not in seen, f"collision at {idx}: {seen[idx]} vs {(layer, module, role)}"
                seen[idx] = (layer, module, role)
    assert sorted(seen) == list(range(spec.n_queries))


def test_index_tensors_match_scalar_index(spec):
    layer_idx, module_idx, role_idx = spec.query_indices()
    assert layer_idx.shape == module_idx.shape == role_idx.shape == (spec.n_queries,)

    for idx in range(spec.n_queries):
        module = spec.target_modules[module_idx[idx]]
        assert spec.query_index(int(layer_idx[idx]), module, int(role_idx[idx])) == idx


def test_query_base_indices(spec):
    for module in spec.target_modules:
        base = spec.query_base_indices(module)
        assert base.shape == (spec.n_layers,)
        for layer in range(spec.n_layers):
            assert int(base[layer]) == spec.query_index(layer, module, ROLE_A)
            assert int(base[layer]) + 1 == spec.query_index(layer, module, ROLE_B)


def test_layer_major_contiguity(spec):
    """All queries for one layer are contiguous, so a layer's block can be sliced."""
    layer_idx, _, _ = spec.query_indices()
    per_layer = spec.n_modules * N_ROLES
    for layer in range(spec.n_layers):
        block = layer_idx[layer * per_layer : (layer + 1) * per_layer]
        assert (block == layer).all()


def test_heads_read_the_token_they_claim_to(hypernet, spec):
    """Feed a hidden state that encodes its own index; check each head sees the right one.

    Bypasses the backbone entirely and calls heads_forward on a synthetic ``h`` in which
    token ``i`` is the one-hot-ish vector ``i``. A head reading the wrong token would
    produce a different output than the same head applied directly to the expected token.
    """
    d = hypernet.d_model
    h = torch.zeros(1, spec.n_queries, d)
    for i in range(spec.n_queries):
        h[0, i, i % d] = float(i + 1)

    per_module = hypernet.heads_forward(h)

    for module in spec.target_modules:
        head = hypernet.heads[module]
        A, B = per_module[module]
        for layer in range(spec.n_layers):
            a_tok = h[:, spec.query_index(layer, module, ROLE_A)]
            b_tok = h[:, spec.query_index(layer, module, ROLE_B)]
            want_A, want_B = head(a_tok.unsqueeze(1), b_tok.unsqueeze(1))
            torch.testing.assert_close(A[:, layer : layer + 1], want_A)
            torch.testing.assert_close(B[:, layer : layer + 1], want_B)


def test_generated_shapes(hypernet, spec):
    per_module = hypernet(["step by step"])
    for module in spec.target_modules:
        A, B = per_module[module]
        assert A.shape == (1, spec.n_layers, spec.r, spec.in_features[module])
        assert B.shape == (1, spec.n_layers, spec.out_features[module], spec.r)
