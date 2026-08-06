"""The recon -> SFT handoff: param-group LRs and the strict=True checkpoint load.

See docs/03_training_validation.md §3, handoff gotchas.
"""

from __future__ import annotations

import torch

from steerable_t2l.checkpoint import load_hypernet, save_checkpoint
from steerable_t2l.testing import tiny_backbone
from steerable_t2l.trainers.sft import SFTConfig, build_param_groups


def test_build_param_groups_warm_started_splits_lr(hypernet):
    config = SFTConfig(lr_heads=2e-5, lr_backbone_lora=2e-6, lr_from_scratch=1e-4)
    groups = build_param_groups(hypernet, config, warm_started=True)

    names = {g["name"] for g in groups}
    assert names == {"backbone_lora", "heads"}
    lr_by_name = {g["name"]: g["lr"] for g in groups}
    assert lr_by_name["backbone_lora"] == 2e-6
    assert lr_by_name["heads"] == 2e-5

    total = sum(p.numel() for g in groups for p in g["params"])
    expected = sum(p.numel() for p in hypernet.parameters() if p.requires_grad)
    assert total == expected


def test_build_param_groups_from_scratch_is_a_single_flat_group(hypernet):
    config = SFTConfig(lr_from_scratch=1e-4)
    groups = build_param_groups(hypernet, config, warm_started=False)

    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-4
    total = sum(p.numel() for p in groups[0]["params"])
    expected = sum(p.numel() for p in hypernet.parameters() if p.requires_grad)
    assert total == expected


def test_recon_checkpoint_loads_strict_for_warm_started_sft(tmp_path, hypernet, spec, tokenizer):
    path = tmp_path / "recon.pt"
    save_checkpoint(path, hypernet, hypernet.config, spec, stage="recon", step=100)

    loaded, payload = load_hypernet(
        path,
        device="cpu",
        dtype=torch.float32,
        backbone=tiny_backbone(tokenizer, hidden=hypernet.d_model),
        tokenizer=tokenizer,
    )

    # Handoff gotcha #3: the caller must assert this before trusting the warm start.
    assert payload["target_spec"] == spec.to_dict()
    assert payload["stage"] == "recon"

    # Handoff gotcha #1: a warm start must not have re-zeroed the recon-trained heads --
    # loading strict=True means the loaded state IS the recon checkpoint's state, verbatim.
    original = dict(hypernet.state_dict())
    reloaded = dict(loaded.state_dict())
    for name in original:
        torch.testing.assert_close(reloaded[name], original[name], msg=name)


def test_mismatched_target_spec_is_detectable_before_warm_start(hypernet, spec):
    payload_spec = spec.to_dict()
    mismatched_spec = spec.replace(r=spec.r + 1).to_dict()
    assert payload_spec != mismatched_spec
