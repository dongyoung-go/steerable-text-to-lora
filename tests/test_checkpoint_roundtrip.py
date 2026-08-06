"""Checkpoint save/load round-trip and the strict=True handoff contract.

See docs/03_training_validation.md §3.
"""

from __future__ import annotations

import pytest
import torch

from steerable_t2l.checkpoint import load_hypernet, save_checkpoint
from steerable_t2l.testing import tiny_backbone


def test_save_checkpoint_rejects_unknown_stage(tmp_path, hypernet, spec):
    with pytest.raises(ValueError, match="stage"):
        save_checkpoint(tmp_path / "x.pt", hypernet, hypernet.config, spec, stage="bogus", step=0)


def test_checkpoint_roundtrip_strict_load(tmp_path, hypernet, spec, tokenizer):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, hypernet, hypernet.config, spec, stage="recon", step=42)

    loaded_model, payload = load_hypernet(
        path,
        device="cpu",
        dtype=torch.float32,
        backbone=tiny_backbone(tokenizer, hidden=hypernet.d_model),
        tokenizer=tokenizer,
    )

    assert payload["stage"] == "recon"
    assert payload["step"] == 42
    assert payload["target_spec"] == spec.to_dict()
    assert payload["hypernet_config"] == hypernet.config.to_dict()

    original = dict(hypernet.state_dict())
    reloaded = dict(loaded_model.state_dict())
    assert set(original) == set(reloaded)
    for name in original:
        torch.testing.assert_close(reloaded[name], original[name], msg=name)


def test_checkpoint_handoff_target_spec_assertion(tmp_path, hypernet, spec):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, hypernet, hypernet.config, spec, stage="recon", step=0)

    payload = torch.load(path, weights_only=False)
    mismatched_spec = spec.replace(r=spec.r + 1)
    assert payload["target_spec"] != mismatched_spec.to_dict()
