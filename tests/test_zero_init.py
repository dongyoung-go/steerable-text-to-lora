"""The zero-init contract: at step 0 the generated LoRA is an *exactly* zero delta.

If this fails, training does not start from the base model, and the step-0 loss is not the
frozen model's loss -- which silently invalidates every comparison against a no-adapter
baseline.
"""

from __future__ import annotations

import math

import torch

from steerable_t2l.hooks import build_sites, delta_weights, lora_hooks
from steerable_t2l.hypernet import peft_lora_a_init
from steerable_t2l.testing import tiny_hypernet
from tests.conftest import DESCS


def test_B_is_exactly_zero(hypernet, spec):
    per_module = hypernet(DESCS)
    for module in spec.target_modules:
        _, B = per_module[module]
        # Exactly zero, not allclose: out_B has both weight and bias zeroed, so B == 0 for
        # every input regardless of what the refiner and shared decoder produce.
        assert (B == 0).all(), f"{module}: B is not exactly zero"


def test_delta_w_is_exactly_zero(hypernet, spec):
    per_module = hypernet(DESCS)
    for module in spec.target_modules:
        A, B = per_module[module]
        dW = delta_weights(A, B, spec.scaling)
        assert (dW == 0).all(), f"{module}: deltaW is not exactly zero"


def test_A_matches_peft_style_init(hypernet, spec):
    """A is a constant equal to PEFT's Kaiming init: bounded by 1/sqrt(fan_in), nonzero."""
    per_module = hypernet(DESCS)
    for module in spec.target_modules:
        A, _ = per_module[module]
        bound = 1.0 / math.sqrt(spec.in_features[module])
        assert (A.abs() <= bound + 1e-6).all(), f"{module}: A exceeds the Kaiming bound"
        assert (A != 0).any(), f"{module}: A is all zero -- the head bias did not land"

        # Constant across samples and layers: out_A.weight is zeroed, so only the bias speaks.
        torch.testing.assert_close(A[0, 0].expand_as(A), A)


def test_peft_lora_a_init_bound():
    w = peft_lora_a_init(64, 8, torch.Generator().manual_seed(0))
    assert w.shape == (8, 64)
    bound = 1.0 / math.sqrt(64)
    assert w.abs().max() <= bound
    assert w.abs().max() > 0.5 * bound  # actually spread over the range, not degenerate


def test_hooked_logits_are_bitwise_identical(hypernet, spec, target_model):
    """The whole point: with hooks attached at init, the target model is untouched."""
    input_ids = torch.randint(0, 200, (len(DESCS), 12))
    with torch.no_grad():
        baseline = target_model(input_ids=input_ids).logits

    per_module = hypernet(DESCS)
    sites = build_sites(spec, per_module)
    with lora_hooks(target_model, sites, spec.scaling), torch.no_grad():
        hooked = target_model(input_ids=input_ids).logits

    assert torch.equal(baseline, hooked), "hooks perturbed the model at init"


def test_zero_init_can_be_disabled(spec, tokenizer):
    """A warm-started run must not have its trained heads re-zeroed."""
    model = tiny_hypernet(spec, tokenizer, zero_init=False)
    per_module = model(DESCS)
    nonzero = [m for m, (_, B) in per_module.items() if (B != 0).any()]
    assert nonzero, "zero_init=False still produced an all-zero B everywhere"
