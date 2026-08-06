"""No trainable parameter is *permanently* starved of gradient.

This is the gate test, and the invariant is subtler than "everything gets a gradient".

The zero-init contract sets ``out_B.weight`` and ``out_B.bias`` to zero, so at step 0:

    dL/dA          proportional to  B == 0                  -> out_A is idle
    dL/d(bottleneck output)  =  out_A.weight^T dL/dA  +  out_B.weight^T dL/dB
                             =        0               +          0            -> idle

i.e. **only ``out_B``'s weight and bias move on the first step** -- exactly as in standard
LoRA, where B=0 means only B moves first. Everything upstream (bottleneck, shared decoder,
refiner, query bank, backbone LoRA) unblocks on step 1 once ``out_B.weight`` is nonzero.

The failure this guards against is that unblocking never happening: if the shared decoder or
the bottleneck emitted zeros, ``dL/d(out_B.weight) = dL/dB (x) bottleneck_activation`` would
also be zero and B would stay pinned at zero forever. Nothing raises -- the loss curve just
stays flat -- so this test is the only thing between that failure and a wasted training run.
Do not relax it.
"""

from __future__ import annotations

import pytest
import torch

from steerable_t2l.hooks import build_sites, lora_hooks
from tests.conftest import DESCS


def _backward(hypernet, spec, target_model, seed: int) -> None:
    hypernet.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 200, (len(DESCS), 10))
    sites = build_sites(spec, hypernet.generate_for_batch(DESCS))
    with lora_hooks(target_model, sites, spec.scaling):
        out = target_model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()


def _group_norms(hypernet) -> dict[str, float]:
    return {
        name: sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
        for name, params in hypernet.parameter_groups().items()
    }


@pytest.fixture
def after_one_step(hypernet, spec, target_model):
    """Backward once, take an optimizer step, backward again -- the steady state."""
    _backward(hypernet, spec, target_model, seed=1)
    optimizer = torch.optim.SGD([p for p in hypernet.parameters() if p.requires_grad], lr=1e-2)
    optimizer.step()
    _backward(hypernet, spec, target_model, seed=2)
    return hypernet


# -- step 0: the intended lag -----------------------------------------------------------


def test_step_zero_moves_out_B(hypernet, spec, target_model):
    """The one parameter that must escape the zero-init, for every module."""
    _backward(hypernet, spec, target_model, seed=1)
    for module in spec.target_modules:
        head = hypernet.heads[module]
        assert head.out_B.weight.grad is not None
        assert float(head.out_B.weight.grad.abs().sum()) > 0, (
            f"{module}: out_B.weight has zero gradient -- B is pinned at zero forever. "
            "Check that the shared decoder does not emit zeros and that the bottleneck is "
            "not zero-initialized."
        )
        assert float(head.out_B.bias.grad.abs().sum()) > 0


def test_step_zero_upstream_is_idle(hypernet, spec, target_model):
    """Documented, not accidental: everything above out_B waits one step.

    Asserted so that a change which makes these live at step 0 -- e.g. dropping the zero-init
    -- is noticed rather than silently accepted.
    """
    _backward(hypernet, spec, target_model, seed=1)
    norms = _group_norms(hypernet)
    for group in ("backbone_lora", "queries", "refiner", "shared_decoder"):
        assert norms[group] == 0.0, f"{group} was expected to be idle at step 0, got {norms[group]}"
    assert norms["heads"] > 0


# -- step 1 onward: everything must be live ---------------------------------------------


def test_every_group_gets_gradient(after_one_step):
    for name, params in after_one_step.parameter_groups().items():
        assert params, f"group {name!r} has no trainable parameters"
        assert _group_norms(after_one_step)[name] > 0, (
            f"group {name!r} is still receiving zero gradient after an optimizer step -- "
            "the path from the task loss to it is severed"
        )


def test_head_bottleneck_unblocks(after_one_step, spec):
    for module in spec.target_modules:
        head = after_one_step.heads[module]
        assert float(head.bottleneck.weight.grad.abs().sum()) > 0, f"{module}: dead bottleneck"


def test_out_A_unblocks(after_one_step, spec):
    for module in spec.target_modules:
        head = after_one_step.heads[module]
        assert float(head.out_A.weight.grad.abs().sum()) > 0, f"{module}: out_A never becomes live"


def test_query_factors_all_get_gradient(after_one_step):
    """Each compositional factor separately -- a dead e_role would collapse A and B."""
    queries = after_one_step.queries
    for name in ("q_base", "e_layer", "e_module", "e_role"):
        param = getattr(queries, name)
        param = param if name == "q_base" else param.weight
        assert param.grad is not None and float(param.grad.abs().sum()) > 0, f"dead {name}"


# -- preconditions and freezing ---------------------------------------------------------


def test_shared_decoder_output_is_not_zero(hypernet):
    """The precondition behind out_B's gradient, checked directly."""
    h = hypernet.encode(DESCS).detach()
    assert float(h.abs().mean()) > 0, "shared decoder emitted zeros -- out_B could never learn"
    assert torch.isfinite(h).all()


def test_target_model_stays_frozen(hypernet, spec, target_model):
    _backward(hypernet, spec, target_model, seed=1)
    assert all(p.grad is None for p in target_model.parameters())


def test_backbone_base_weights_stay_frozen(after_one_step):
    leaked = [
        name
        for name, p in after_one_step.backbone.named_parameters()
        if not p.requires_grad and p.grad is not None
    ]
    assert not leaked, f"frozen backbone weights received gradient: {leaked[:3]}"
