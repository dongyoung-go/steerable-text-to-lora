"""SVD canonicalization: gauge invariance and the PEFT-convention round-trip.

See docs/03_training_validation.md §2 -- "a canonicalization test: two random
reparameterizations (R⁻¹A, BR) of the same ΔW must canonicalize to identical (A_c, B_c)".
"""

from __future__ import annotations

import torch

from steerable_t2l.hooks import delta_weights
from steerable_t2l.oracle.canonicalize import (
    canonicalize_adapter,
    canonicalize_state_dict,
    fix_svd_signs,
    load_and_canonicalize_oracle,
)


def test_canonicalize_is_gauge_invariant():
    torch.manual_seed(0)
    r, in_f, out_f = 8, 32, 24
    A = torch.randn(r, in_f, dtype=torch.float64)
    B = torch.randn(out_f, r, dtype=torch.float64)
    R = torch.randn(r, r, dtype=torch.float64)  # any invertible r x r (almost surely so)

    A2 = torch.linalg.solve(R, A)
    B2 = B @ R

    Ac1, Bc1, S1 = canonicalize_adapter(A, B)
    Ac2, Bc2, S2 = canonicalize_adapter(A2, B2)

    torch.testing.assert_close(Ac1, Ac2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(Bc1, Bc2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(S1, S2, atol=1e-4, rtol=1e-4)


def test_canonicalize_round_trips_delta_weights():
    torch.manual_seed(1)
    r, in_f, out_f = 8, 32, 24
    A = torch.randn(r, in_f, dtype=torch.float64)
    B = torch.randn(out_f, r, dtype=torch.float64)

    A_canon, B_canon, _ = canonicalize_adapter(A, B)

    dW_canon = delta_weights(A_canon.unsqueeze(0), B_canon.unsqueeze(0), 1.0)[0]
    dW_orig = delta_weights(A.float().unsqueeze(0), B.float().unsqueeze(0), 1.0)[0]
    torch.testing.assert_close(dW_canon, dW_orig, atol=1e-3, rtol=1e-3)


def test_fix_svd_signs_makes_largest_entry_positive():
    torch.manual_seed(2)
    U = torch.randn(10, 4, dtype=torch.float64)
    Vh = torch.randn(4, 6, dtype=torch.float64)
    U2, Vh2 = fix_svd_signs(U, Vh)
    for i in range(4):
        idx = int(U2[:, i].abs().argmax())
        assert U2[idx, i] > 0


def test_fix_svd_signs_preserves_the_product():
    torch.manual_seed(3)
    U = torch.randn(10, 4, dtype=torch.float64)
    Vh = torch.randn(4, 6, dtype=torch.float64)
    U2, Vh2 = fix_svd_signs(U, Vh)
    torch.testing.assert_close(U2 @ Vh2, U @ Vh, atol=1e-10, rtol=1e-10)


def test_canonicalize_state_dict_and_load_from_peft_adapter(tmp_path, spec, target_model):
    from peft import get_peft_model

    peft_model = get_peft_model(target_model, spec.to_lora_config())
    oracle_dir = tmp_path / "oracle_00"
    oracle_dir.mkdir()
    peft_model.save_pretrained(str(oracle_dir))

    per_module = load_and_canonicalize_oracle(str(oracle_dir), spec)
    assert set(per_module) == set(spec.target_modules)
    for module, (A, B) in per_module.items():
        assert A.shape == (spec.n_layers, spec.r, spec.in_features[module])
        assert B.shape == (spec.n_layers, spec.out_features[module], spec.r)

    # The canonicalized adapter must compute the same ΔW as the original PEFT weights.
    import peft as peft_lib

    raw = peft_lib.load_peft_weights(str(oracle_dir))
    canon, spectra = canonicalize_state_dict(raw, spec)
    assert spectra  # spectrum logged per (module, layer)

    for module in spec.target_modules:
        for layer in range(spec.n_layers):
            from steerable_t2l.hooks import qualified_name
            from steerable_t2l.oracle.canonicalize import PEFT_KEY_PREFIX

            stem = f"{PEFT_KEY_PREFIX}.{layer}.{qualified_name(module)}"
            A_orig = raw[f"{stem}.lora_A.weight"]
            B_orig = raw[f"{stem}.lora_B.weight"]
            A_c = canon[f"{stem}.lora_A.weight"]
            B_c = canon[f"{stem}.lora_B.weight"]

            dW_orig = delta_weights(A_orig.unsqueeze(0), B_orig.unsqueeze(0), 1.0)[0]
            dW_canon = delta_weights(A_c.unsqueeze(0), B_c.unsqueeze(0), 1.0)[0]
            torch.testing.assert_close(dW_canon, dW_orig.float(), atol=1e-3, rtol=1e-3)
