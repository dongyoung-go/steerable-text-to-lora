"""End-to-end task loss (SFT) and the recon -> SFT handoff.

See ``docs/03_training_validation.md`` §3. The core step is already expressible with what's
implemented in ``hypernet.py``/``hooks.py``::

    per_module = hypernet.generate_for_batch(batch["descs"])
    sites = build_sites(spec, per_module)
    with lora_hooks(target, sites, spec.scaling):
        loss = target(**batch).loss

Shifted CE over response tokens only, per-sequence length normalization, plus
``l2_reg_generated_w * (A.pow(2).mean() + B.pow(2).mean())``. Backbone gradient checkpointing
on (configured by ``HyperNetConfig``/``SteerableHyperLoRA``); target gradient checkpointing is
also on by default here (``train_sft``'s ``target_gradient_checkpointing=True``) -- a deviation
from docs/03's original assumption, forced by the real ``inp_max_len=2560`` profiling result
combined with a GQA/SDPA memory-efficiency gap in the installed transformers/PyTorch versions
that otherwise OOMs a single B200. See ``train_sft``'s docstring for the measured numbers.

⚠️ Do NOT reproduce the reference's grad-accum bug (``optimizer.zero_grad()`` called inside
the accumulate block right before ``backward()``, silently negating accumulation) -- zero
*after* ``optimizer.step()``.

Handoff gotchas (docs/03 §3)
    1. ``zero_init = (args.init_from is None)`` -- a warm-started run must not re-zero its
       recon-trained heads. This is the CALLER's responsibility (``scripts/train_sft.py``),
       since constructing the hypernetwork happens before ``train_sft`` is invoked -- mirrors
       ``trainers/recon.py::train_recon``'s "caller builds the model" shape.
    2. LR must drop hard: recon ~5e-4 -> SFT 2e-5, warmup 0.03, backbone LoRA ~10x lower than
       the heads. See ``build_param_groups``.
    3. ``assert loaded["target_spec"] == spec.to_dict()`` -- also the caller's responsibility.
    4. Optional: freeze the backbone LoRA for the first ``freeze_backbone_lora_steps`` steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import yaml

from steerable_t2l.checkpoint import save_checkpoint
from steerable_t2l.data.datasets import DataConfig, build_dataloader
from steerable_t2l.data.registry import Task
from steerable_t2l.data.splits import Splits
from steerable_t2l.hooks import build_sites, lora_hooks
from steerable_t2l.hypernet import SteerableHyperLoRA
from steerable_t2l.losses import per_sequence_normalized_ce
from steerable_t2l.target_spec import TargetSpec
from steerable_t2l.validation import run_validation


@dataclass
class SFTConfig:
    lr_heads: float = 2e-5
    lr_backbone_lora: float = 2e-6
    lr_from_scratch: float = 1e-4
    warmup_frac: float = 0.03
    max_steps: int = 20000
    grad_accum: int = 4
    n_tasks_per_batch: int = 4
    n_points_per_task: int = 4
    l2_reg_generated_w: float = 1e-3
    val_freq: int = 200
    max_grad_norm: float = 1.0
    freeze_backbone_lora_steps: int = 0
    init_from: str | None = None
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SFTConfig:
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def sft_step(
    batch: dict,
    hypernet: SteerableHyperLoRA,
    target,
    spec: TargetSpec,
    config: SFTConfig,
    backward_fn=None,
) -> dict:
    """``dedup`` -> ``generate_for_batch`` -> ``build_sites`` -> ``lora_hooks`` -> CE + L2 reg.

    ``backward_fn``, if given, is called with the total loss WHILE the LoRA hooks are still
    attached. This matters whenever ``target`` uses gradient checkpointing: checkpointing
    recomputes each layer's forward during ``backward()``, which happens *after* a bare
    ``with lora_hooks(...): forward()`` block has already removed its hooks -- the recompute
    would then silently run the plain frozen target with no LoRA injection at all, either
    corrupting gradients or (if the recomputed graph also saves a different number of
    tensors than the original forward did) raising ``torch.utils.checkpoint.CheckpointError``.
    Passing ``accelerator.backward`` here keeps the hooks alive across the whole backward call.
    If ``backward_fn`` is ``None``, the caller must call ``.backward()`` itself and must
    guarantee ``target`` is NOT gradient-checkpointed.
    """
    per_module = hypernet.generate_for_batch(batch["descs"])
    sites = build_sites(spec, per_module)
    with lora_hooks(target, sites, spec.scaling):
        logits = target(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        ce_loss = per_sequence_normalized_ce(logits, batch["labels"]).mean()
        reg = sum(A.float().pow(2).mean() + B.float().pow(2).mean() for A, B in per_module.values())
        reg_loss = config.l2_reg_generated_w * (reg / len(per_module))
        loss = ce_loss + reg_loss
        if backward_fn is not None:
            backward_fn(loss)
            return {"loss": loss.detach(), "ce_loss": ce_loss.detach(), "reg_loss": reg_loss.detach()}

    return {"loss": loss, "ce_loss": ce_loss.detach(), "reg_loss": reg_loss.detach()}


def build_param_groups(hypernet: SteerableHyperLoRA, config: SFTConfig, warm_started: bool) -> list[dict]:
    """Backbone LoRA at ``lr_backbone_lora`` (~10x lower than the heads) when warm-started;
    everything at ``lr_from_scratch`` when not -- the ablation's "from scratch" arm has no
    per-group split (docs/03's ablation spec names one flat ``lr=1e-4``)."""
    groups = hypernet.parameter_groups()
    if not warm_started:
        all_params = [p for params in groups.values() for p in params]
        return [{"params": all_params, "lr": config.lr_from_scratch, "name": "all"}]

    head_params = [
        p
        for name in ("queries", "refiner", "shared_decoder", "heads")
        for p in groups[name]
    ]
    return [
        {"params": groups["backbone_lora"], "lr": config.lr_backbone_lora, "name": "backbone_lora"},
        {"params": head_params, "lr": config.lr_heads, "name": "heads"},
    ]


def train_sft(
    config: SFTConfig,
    hypernet: SteerableHyperLoRA,
    target,
    spec: TargetSpec,
    tasks: list[Task],
    splits: Splits,
    tokenizer,
    data_config: DataConfig,
    *,
    warm_started: bool = False,
    accelerator=None,
    out_dir: str | Path | None = None,
    oracle_dir: str | Path | None = None,
    val_batch_size: int = 8,
    target_gradient_checkpointing: bool = True,
) -> dict:
    """``accelerator`` defaults to ``Accelerator(mixed_precision="bf16", gradient_accumulation_
    steps=config.grad_accum)`` -- pass a CPU ``Accelerator(mixed_precision="no", ...)`` for
    tests. Only ``hypernet`` is ``accelerator.prepare``d: ``target`` is frozen with no
    gradient sync needed, so it is just moved to ``accelerator.device`` and left unwrapped.

    ``target_gradient_checkpointing=True`` is a deviation from docs/03's original memory
    budget (which assumed target checkpointing off, at ``L≈1024``). Real length-profiling of
    the GSM8K data forced ``inp_max_len=2560`` (see ``configs/data.yaml``), and at that length
    the installed transformers/PyTorch's SDPA path does not broadcast Qwen2's GQA k/v heads
    through a fused (flash/memory-efficient) kernel and silently falls back to materializing
    full attention score matrices for every layer at once -- ~135 GB forward-only at
    ``bs=16``, enough to OOM even a 178 GB B200 on its own. Checkpointing the target holds
    only one layer's activations at a time, cutting peak fwd+bwd memory to ~44 GB regardless
    of which attention backend is actually dispatched. ``target.train()`` is required for HF's
    checkpointing wrapper to engage (it no-ops when ``self.training`` is False) -- harmless
    here since the target has no dropout and none of its parameters receive gradients.
    """
    if accelerator is None:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=config.grad_accum)

    param_groups = build_param_groups(hypernet, config, warm_started)
    optimizer = torch.optim.AdamW(param_groups)
    warmup_steps = max(1, int(config.max_steps * config.warmup_frac))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    train_data_config = replace(
        data_config, n_tasks_per_batch=config.n_tasks_per_batch, n_points_per_task=config.n_points_per_task
    )
    train_loader = build_dataloader(tasks, tokenizer, train_data_config, split="train", seed=config.seed)

    hypernet, optimizer, train_loader, scheduler = accelerator.prepare(
        hypernet, optimizer, train_loader, scheduler
    )
    base_hypernet = accelerator.unwrap_model(hypernet)
    target = target.to(accelerator.device)
    target.config.use_cache = False
    if target_gradient_checkpointing:
        target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        target.train()  # required for HF's checkpointing wrapper to actually engage
    else:
        target.eval()

    if config.freeze_backbone_lora_steps > 0:
        for p in base_hypernet.parameter_groups()["backbone_lora"]:
            p.requires_grad_(False)

    history: list[dict] = []
    step = 0
    train_iter = iter(train_loader)

    while step < config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(hypernet):
            # backward_fn=accelerator.backward: the hooks must still be attached when
            # backward actually runs, since target's gradient checkpointing recomputes the
            # forward at that point (see sft_step's docstring).
            out = sft_step(batch, base_hypernet, target, spec, config, backward_fn=accelerator.backward)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(hypernet.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                # Zero AFTER step, never before backward -- the reference's bug
                # (sft_trainer.py:245-249) zeros inside the accumulate block right before
                # backward, silently negating everything accumulated so far.
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if config.freeze_backbone_lora_steps and step == config.freeze_backbone_lora_steps:
                    for p in base_hypernet.parameter_groups()["backbone_lora"]:
                        p.requires_grad_(True)

                if step % config.val_freq == 0 or step == config.max_steps:
                    metrics = run_validation(
                        base_hypernet, target, spec, tasks, splits, tokenizer, data_config,
                        oracle_dir=oracle_dir, val_batch_size=val_batch_size, seed=config.seed,
                    )
                    entry = {
                        "step": step,
                        "ce_loss": out["ce_loss"].item(),
                        "reg_loss": out["reg_loss"].item(),
                        "steering_margin": metrics["steering_margin"],
                        "overall": metrics["overall"],
                    }
                    history.append(entry)

                    if out_dir is not None:
                        save_checkpoint(
                            Path(out_dir) / "latest.pt", base_hypernet, base_hypernet.config, spec,
                            stage="sft", step=step, extra={"sft_config": config.to_dict(), "history": history},
                        )

    return {"history": history}
