"""Magnitude-normalized L1 reconstruction loss. See docs/03_training_validation.md §2, Stage C."""

from __future__ import annotations

import torch

from steerable_t2l.trainers.recon import normalized_l1, recon_loss


def test_normalized_l1_matches_hand_computed_value():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 1.5, 1.5])
    loss, normalizer = normalized_l1(pred, target)
    expected_normalizer = target.abs().mean()
    expected_loss = (pred - target).abs().mean() / expected_normalizer
    torch.testing.assert_close(normalizer, expected_normalizer)
    torch.testing.assert_close(loss, expected_loss)


def test_normalized_l1_normalizer_is_detached():
    pred = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0], requires_grad=True)
    loss, normalizer = normalized_l1(pred, target)
    assert not normalizer.requires_grad
    loss.backward()
    assert pred.grad is not None
    # Gradient flows into target's numerator term but not through the (detached) normalizer.
    assert target.grad is not None


def test_normalized_l1_zero_when_pred_equals_target():
    x = torch.tensor([1.0, -2.0, 3.0])
    loss, _ = normalized_l1(x, x)
    assert loss.item() == 0.0


def test_recon_loss_averages_module_role_components():
    per_module = {
        "q_proj": (torch.zeros(2, 3), torch.zeros(2, 3)),
        "v_proj": (torch.ones(2, 3), torch.ones(2, 3)),
    }
    target_A = {"q_proj": torch.zeros(2, 3), "v_proj": torch.full((2, 3), 2.0)}
    target_B = {"q_proj": torch.zeros(2, 3), "v_proj": torch.full((2, 3), 2.0)}

    # q_proj: pred==target==0 -> normalizer is 0, clamped to eps, loss ~ 0.
    # v_proj: pred=1, target=2 -> normalized_l1 = |1-2|/2 = 0.5 for both A and B.
    out = recon_loss(per_module, target_A, target_B)
    assert set(out["components"]) == {"q_proj.A", "q_proj.B", "v_proj.A", "v_proj.B"}
    assert out["components"]["v_proj.A"] == 0.5
    assert out["components"]["v_proj.B"] == 0.5
    assert out["normalizers"]["v_proj.A"] == 2.0
    assert torch.isfinite(out["loss"])
