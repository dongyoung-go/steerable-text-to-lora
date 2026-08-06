"""Reconstruction and SFT trainers. See ``docs/03_training_validation.md`` §§2c, 3.

``recon.py`` -- reconstruction warm-start
    Batch = (task, description) pairs; no target-model forward at all, so the spec is built
    with ``TargetSpec.from_pretrained`` (AutoConfig only) and target weights are never loaded.
    Loss = per-(module, role) magnitude-normalized L1 against the canonicalized oracles.
    Normalizers are stored in the checkpoint's config dict, NOT as module buffers -- that is
    what keeps the recon and SFT ``state_dict``s structurally identical so the handoff can
    load ``strict=True``. Expect this to be a warm start only: a handful of oracle adapters
    cannot teach instruction generalization. Judge it solely by the from-scratch vs.
    warm-started ablation.

``sft.py`` -- end-to-end task loss
    The core step::

        per_module = hypernet.generate_for_batch(batch["descs"])
        sites = build_sites(spec, per_module)
        with lora_hooks(target, sites, spec.scaling):
            loss = target(**batch).loss

    Shifted CE over response tokens only, per-sequence length normalization, plus
    ``l2_reg_generated_w * (A.pow(2).mean() + B.pow(2).mean())``.
    Backbone gradient checkpointing on, target off.
    Do NOT reproduce the reference's grad-accum bug (it calls ``optimizer.zero_grad()``
    inside the accumulate block right before ``backward()``, silently negating accumulation)
    -- zero *after* ``optimizer.step()``.

Handoff gotchas
    1. ``zero_init = (args.init_from is None)``; re-zeroing discards the entire recon stage.
    2. LR must drop hard: recon ~5e-4 -> SFT 2e-5, warmup 0.03, backbone LoRA ~10x lower than
       the heads. Log L2 drift from the init checkpoint.
    3. ``assert loaded["target_spec"] == spec.to_dict()``.

Checkpoint format (both stages)::

    {"state_dict", "hypernet_config", "target_spec", "stage", "step"}
"""

from steerable_t2l.trainers.recon import (
    ReconConfig,
    build_recon_batches,
    evaluate_recon,
    recon_loss,
    train_recon,
)
from steerable_t2l.trainers.sft import SFTConfig, build_param_groups, sft_step, train_sft

__all__ = [
    "ReconConfig",
    "SFTConfig",
    "build_param_groups",
    "build_recon_batches",
    "evaluate_recon",
    "recon_loss",
    "sft_step",
    "train_recon",
    "train_sft",
]
