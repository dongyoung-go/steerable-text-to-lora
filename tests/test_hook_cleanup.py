"""Hook handles are always removed, including when the wrapped forward raises.

A leaked handle would keep injecting a stale adapter -- one belonging to a previous batch,
and detached from the current graph -- into every subsequent forward. The reference
implementation removes handles on the success path only.
"""

from __future__ import annotations

import pytest
import torch

from steerable_t2l.hooks import build_sites, get_layers, lora_hooks, qualified_name, resolve_module


def _hook_count(model) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


def _sites(spec):
    per_module = {
        m: (
            torch.zeros(1, spec.n_layers, spec.r, spec.in_features[m]),
            torch.zeros(1, spec.n_layers, spec.out_features[m], spec.r),
        )
        for m in spec.target_modules
    }
    return build_sites(spec, per_module)


def test_hooks_registered_then_removed(spec, target_model):
    before = _hook_count(target_model)
    with lora_hooks(target_model, _sites(spec), spec.scaling):
        assert _hook_count(target_model) == before + spec.n_layers * spec.n_modules
    assert _hook_count(target_model) == before


def test_hooks_removed_when_forward_raises(spec, target_model):
    before = _hook_count(target_model)
    with pytest.raises(RuntimeError, match="boom"), lora_hooks(target_model, _sites(spec), spec.scaling):
        raise RuntimeError("boom")
    assert _hook_count(target_model) == before, "handles leaked on the exception path"


def test_hooks_removed_on_partial_registration_failure(spec, target_model):
    """If site N fails to resolve, the N-1 already-registered hooks must still come off."""
    from steerable_t2l.hooks import Site

    sites = _sites(spec) + [Site(0, "does_not_exist", torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))]
    before = _hook_count(target_model)
    with pytest.raises(AttributeError):
        with lora_hooks(target_model, sites, spec.scaling):
            pass  # pragma: no cover - never reached
    assert _hook_count(target_model) == before


def test_module_name_resolution(target_model):
    assert qualified_name("q_proj") == "self_attn.q_proj"
    assert qualified_name("down_proj") == "mlp.down_proj"

    layer = get_layers(target_model)[0]
    assert isinstance(resolve_module(layer, "q_proj"), torch.nn.Linear)
    assert isinstance(resolve_module(layer, "gate_proj"), torch.nn.Linear)
    with pytest.raises(AttributeError, match="has no no_such_proj"):
        resolve_module(layer, "no_such_proj")


def test_get_layers_unwraps_peft(spec, target_model):
    from peft import get_peft_model

    n = len(get_layers(target_model))
    peft_model = get_peft_model(target_model, spec.to_lora_config())
    assert len(get_layers(peft_model)) == n
