"""Hook injection agrees with PEFT for the same A/B.

This is the cross-check that catches a transposed matrix, a wrong ``scaling``, or an
``use_rslora`` mismatch. Every one of those trains perfectly well and then evaluates at
baseline, because the adapter that gets saved is not the adapter that was trained.
"""

from __future__ import annotations

import pytest
import torch
from peft import get_peft_model

from steerable_t2l.hooks import Site, build_sites, delta_weights, lora_hooks


def _random_ab(spec, bs: int, generator: torch.Generator):
    per_module = {}
    for module in spec.target_modules:
        A = torch.randn(bs, spec.n_layers, spec.r, spec.in_features[module], generator=generator) * 0.1
        B = torch.randn(bs, spec.n_layers, spec.out_features[module], spec.r, generator=generator) * 0.1
        per_module[module] = (A, B)
    return per_module


def test_hook_matches_peft(spec, target_model, target_config):
    """Inject via hooks, then load the same weights as a PEFT adapter; logits must match."""
    generator = torch.Generator().manual_seed(7)
    per_module = _random_ab(spec, bs=1, generator=generator)
    input_ids = torch.randint(0, target_config.vocab_size, (1, 9))

    with lora_hooks(target_model, build_sites(spec, per_module), spec.scaling), torch.no_grad():
        hooked = target_model(input_ids=input_ids).logits

    peft_model = get_peft_model(target_model, spec.to_lora_config())
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if not hasattr(module, "lora_A") or "default" not in getattr(module, "lora_A", {}):
                continue
            # ...layers.<i>.self_attn.<module>
            parts = name.split(".")
            layer = int(parts[parts.index("layers") + 1])
            target = parts[-1]
            A, B = per_module[target]
            module.lora_A["default"].weight.copy_(A[0, layer])
            module.lora_B["default"].weight.copy_(B[0, layer])
        peft_out = peft_model(input_ids=input_ids).logits

    torch.testing.assert_close(hooked, peft_out, rtol=1e-4, atol=1e-5)


def test_hook_matches_explicit_delta_w(spec, target_model, target_config):
    """A second, independent check: hooks vs. adding B@A*scaling straight into the weights."""
    generator = torch.Generator().manual_seed(11)
    per_module = _random_ab(spec, bs=1, generator=generator)
    input_ids = torch.randint(0, target_config.vocab_size, (1, 7))

    with lora_hooks(target_model, build_sites(spec, per_module), spec.scaling), torch.no_grad():
        hooked = target_model(input_ids=input_ids).logits

    from steerable_t2l.hooks import get_layers, resolve_module

    originals = {}
    with torch.no_grad():
        for module, (A, B) in per_module.items():
            for layer_idx, layer in enumerate(get_layers(target_model)):
                linear = resolve_module(layer, module)
                originals[(module, layer_idx)] = linear.weight.detach().clone()
                linear.weight.add_(delta_weights(A[0, layer_idx], B[0, layer_idx], spec.scaling))
        merged = target_model(input_ids=input_ids).logits
        for (module, layer_idx), w in originals.items():
            resolve_module(get_layers(target_model)[layer_idx], module).weight.copy_(w)

    torch.testing.assert_close(hooked, merged, rtol=1e-4, atol=1e-5)


def test_per_sample_weights(spec, target_model, target_config):
    """Each row of the batch must get *its own* adapter -- this is what enables multi-task batches."""
    generator = torch.Generator().manual_seed(3)
    per_module = _random_ab(spec, bs=2, generator=generator)
    # Zero out sample 1's adapter entirely, leaving sample 0's intact.
    for _, B in per_module.values():
        B[1].zero_()

    input_ids = torch.randint(0, target_config.vocab_size, (2, 6))
    with torch.no_grad():
        baseline = target_model(input_ids=input_ids).logits
    with lora_hooks(target_model, build_sites(spec, per_module), spec.scaling), torch.no_grad():
        hooked = target_model(input_ids=input_ids).logits

    torch.testing.assert_close(hooked[1], baseline[1], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(hooked[0], baseline[0]), "sample 0's adapter had no effect"


def test_build_sites_rejects_wrong_rank(spec):
    bad = {m: (torch.zeros(1, spec.r, 4), torch.zeros(1, 4, spec.r)) for m in spec.target_modules}
    with pytest.raises(ValueError, match=r"expected A \[bs, n_layers"):
        build_sites(spec, bad)


def test_gradients_reach_the_generated_weights(spec, target_model, target_config):
    A = torch.zeros(1, spec.n_layers, spec.in_features["q_proj"], spec.r, requires_grad=True)
    B = torch.randn(1, spec.n_layers, spec.r, spec.out_features["q_proj"], requires_grad=True)
    sites = [Site(layer, "q_proj", A[:, layer], B[:, layer]) for layer in range(spec.n_layers)]

    input_ids = torch.randint(0, target_config.vocab_size, (1, 5))
    with lora_hooks(target_model, sites, spec.scaling):
        target_model(input_ids=input_ids, labels=input_ids).loss.backward()

    assert A.grad is not None and float(A.grad.abs().sum()) > 0
