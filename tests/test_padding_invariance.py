"""A description's generated LoRA must not depend on what else is in the batch.

Guards the explicit ``position_ids``. The transformers default is ``arange(L + n_queries)``,
which counts pad tokens: with left padding, an instruction batched beside a longer one would
receive shifted RoPE phases on its query tokens and emit a *different* LoRA. Nothing raises;
results just quietly stop being reproducible across batch compositions.
"""

from __future__ import annotations

import torch

SHORT = "Answer only."
LONG = (
    "You will answer a mathematical reasoning question by explicitly showing every "
    "arithmetic operation, unit conversion and intermediate value, verifying each step "
    "against the constraints stated in the problem before producing the final answer."
)


def test_alone_equals_batched_with_longer_neighbour(hypernet):
    alone = hypernet.encode([SHORT])
    batched = hypernet.encode([SHORT, LONG])
    torch.testing.assert_close(alone[0], batched[0], rtol=1e-5, atol=1e-6)


def test_position_in_batch_does_not_matter(hypernet):
    first = hypernet.encode([SHORT, LONG])
    second = hypernet.encode([LONG, SHORT])
    torch.testing.assert_close(first[0], second[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(first[1], second[0], rtol=1e-5, atol=1e-6)


def test_generated_weights_are_batch_invariant(hypernet, spec):
    alone = hypernet([SHORT])
    batched = hypernet([SHORT, LONG])
    for module in spec.target_modules:
        torch.testing.assert_close(alone[module][0][0], batched[module][0][0], rtol=1e-5, atol=1e-6)


def test_padded_positions_are_masked_out(hypernet):
    """Sanity: the pad tokens themselves carry no information into the queries."""
    input_ids, mask = hypernet.tokenize([SHORT, LONG])
    assert (mask[0, : (mask[0] == 0).sum()] == 0).all(), "expected LEFT padding"

    # Overwrite the pad ids with garbage; the encoding of row 0 must not move.
    before = hypernet.encode_tokenized(input_ids, mask)
    scrambled = input_ids.clone()
    scrambled[mask == 0] = 12345
    after = hypernet.encode_tokenized(scrambled, mask)
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


def test_left_padding_is_configured(hypernet):
    assert hypernet.tokenizer.padding_side == "left"
