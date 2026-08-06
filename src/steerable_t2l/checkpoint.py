"""Shared checkpoint I/O for the recon and SFT trainers.

See ``docs/03_training_validation.md`` §3 ("Checkpoint format (both stages)"). Both stages
write and read the identical schema ``{"state_dict", "hypernet_config", "target_spec",
"stage", "step"}`` so a warm-started SFT run can ``load_state_dict(..., strict=True)`` a
recon checkpoint with zero special-casing -- the handoff this module exists to make safe.
"""

from __future__ import annotations

from pathlib import Path

import torch

from steerable_t2l.hypernet import HyperNetConfig, SteerableHyperLoRA
from steerable_t2l.target_spec import TargetSpec

VALID_STAGES = ("recon", "sft")


def save_checkpoint(
    path: str | Path,
    model: SteerableHyperLoRA,
    config: HyperNetConfig,
    spec: TargetSpec,
    stage: str,
    step: int,
    extra: dict | None = None,
) -> None:
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}, got {stage!r}")
    payload = {
        "state_dict": model.state_dict(),
        "hypernet_config": config.to_dict(),
        "target_spec": spec.to_dict(),
        "stage": stage,
        "step": step,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_hypernet(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    backbone=None,
    tokenizer=None,
) -> tuple[SteerableHyperLoRA, dict]:
    """Reconstruct a ``SteerableHyperLoRA`` from a checkpoint and load it ``strict=True``.

    ``zero_init`` is always ``False`` here regardless of how the checkpoint was produced --
    the subsequent ``strict=True`` load overwrites every parameter anyway, so zero-initializing
    first would only be wasted work, never a correctness difference.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    spec = TargetSpec.from_dict(payload["target_spec"])
    config = HyperNetConfig.from_dict(payload["hypernet_config"])
    model = SteerableHyperLoRA(
        spec,
        config,
        zero_init=False,
        backbone=backbone,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
