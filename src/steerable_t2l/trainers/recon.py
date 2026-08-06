"""Stage C: reconstruction warm-start. See ``docs/03_training_validation.md`` §2, Stage C.

Batch = ``(task, description)`` pairs; **no target-model forward at all**, so this stage is
fast and can use a large batch. ``spec`` is built via ``TargetSpec.from_pretrained`` (AutoConfig
only) and target weights are never loaded (saves 3.1 GB).

Loss = per-(module, role) magnitude-normalized L1 against the canonicalized oracles. A and B
magnitudes differ by orders of magnitude across modules; without normalization one module
dominates.

⚠️ The normalizers (``target.abs().mean()`` per module/role) are stored in the checkpoint's
config dict, **not** as ``nn.Module`` buffers -- that is what keeps the recon and SFT
``state_dict``s structurally identical so the SFT handoff can ``load_state_dict(strict=True)``.

**Stated expectation.** A handful of oracle adapters is a very small regression set. This
stage is a *warm start*: it teaches the query/head structure and moves ``B`` off zero. It
cannot teach instruction generalization. Its only success criterion is the from-scratch vs.
warm-started ablation.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from steerable_t2l.checkpoint import save_checkpoint
from steerable_t2l.data.registry import Task
from steerable_t2l.data.splits import Splits, d_axis_available
from steerable_t2l.hooks import delta_weights
from steerable_t2l.hypernet import SteerableHyperLoRA, dedup
from steerable_t2l.oracle.canonicalize import load_and_canonicalize_oracle
from steerable_t2l.target_spec import TargetSpec


@dataclass
class ReconConfig:
    lr: float = 5e-4
    max_steps: int = 2000
    warmup_frac: float = 0.03
    batch_size: int = 64
    val_freq: int = 100
    seed: int = 0
    normalizers: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ReconConfig:
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ReconConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def build_recon_batches(
    tasks: list[Task],
    oracle_dir: str | Path,
    spec: TargetSpec,
    batch_size: int,
    rng: random.Random,
    device: str | torch.device = "cpu",
) -> Iterator[dict]:
    """Batch = ``(task, description)`` pairs, drawn forever (like ``HierarchicalBatchSampler``,
    no epoch boundary). Canonicalized oracle targets are loaded once per task and cached in
    memory -- cheap at any task count actually reached so far, and written as a lookup keyed
    by whatever ``tasks`` contains rather than a fixed-size table, so it scales unchanged as
    more tasks/domains are added later.

    ``device`` must match the hypernet's device: ``load_and_canonicalize_oracle`` otherwise
    defaults to CPU regardless of where the hypernet (and therefore its predicted A/B) live,
    which only happens to work when both sides are CPU.

    Yields ``{"descs": list[str], "target_A": {module: [bs, n_layers, r, in]},
    "target_B": {module: [bs, n_layers, out, r]}}``.
    """
    oracle_cache = {
        task.name: load_and_canonicalize_oracle(str(Path(oracle_dir) / task.name), spec, device=device)
        for task in tasks
    }
    while True:
        chosen_tasks = [rng.choice(tasks) for _ in range(batch_size)]
        descs = [rng.choice(task.metadata.descriptions) for task in chosen_tasks]
        target_A = {
            m: torch.stack([oracle_cache[task.name][m][0] for task in chosen_tasks])
            for m in spec.target_modules
        }
        target_B = {
            m: torch.stack([oracle_cache[task.name][m][1] for task in chosen_tasks])
            for m in spec.target_modules
        }
        yield {"descs": descs, "target_A": target_A, "target_B": target_B}


def normalized_l1(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``L1(pred, target) / target.abs().mean().detach()`` -- the normalizer is recomputed
    from THIS batch's targets (no separate global pre-pass) and detached, so no gradient
    flows through it. Returns ``(loss, normalizer)`` -- the normalizer is what gets logged
    into the checkpoint per the handoff requirement above.
    """
    normalizer = target.abs().mean().detach().clamp_min(1e-8)
    return (pred - target).abs().mean() / normalizer, normalizer


def recon_loss(
    per_module: dict[str, tuple[torch.Tensor, torch.Tensor]],
    target_A: dict[str, torch.Tensor],
    target_B: dict[str, torch.Tensor],
) -> dict:
    """Mean over (module, role) of ``normalized_l1``. Returns the scalar ``loss`` plus
    per-component values and normalizers for logging."""
    components: dict[str, torch.Tensor] = {}
    normalizers: dict[str, float] = {}
    for module, (A_pred, B_pred) in per_module.items():
        loss_A, norm_A = normalized_l1(A_pred, target_A[module])
        loss_B, norm_B = normalized_l1(B_pred, target_B[module])
        components[f"{module}.A"] = loss_A
        components[f"{module}.B"] = loss_B
        normalizers[f"{module}.A"] = float(norm_A)
        normalizers[f"{module}.B"] = float(norm_B)
    loss = torch.stack(list(components.values())).mean()
    return {
        "loss": loss,
        "components": {k: float(v.detach()) for k, v in components.items()},
        "normalizers": normalizers,
    }


def evaluate_recon(
    hypernet: SteerableHyperLoRA,
    tasks: list[Task],
    oracle_dir: str | Path,
    spec: TargetSpec,
    splits: Splits | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Recon-specific scoring: normalized L1 against a "predict the per-module mean target"
    baseline, and cosine similarity of ``ΔW_pred`` to ``ΔW_oracle``. Lives here, not in
    ``validation.py``, since it needs no target model and no tokenized batches -- a different
    input shape than the seven-condition task-loss validation ``validation.py`` is scoped to.

    Uses each task's held-out (D-axis) description when available, else its only/first
    description -- the meaningful "held out" test for reconstruction is generalizing to an
    unseen paraphrase of a known task, not a wholly unseen task (oracle adapters only exist
    for trained tasks).
    """
    if not tasks:
        return {"cosine_similarity": "n/a", "normalized_l1_model": "n/a", "normalized_l1_mean_baseline": "n/a"}

    descs = []
    for task in tasks:
        if splits is not None and d_axis_available(splits, task.name):
            descs.append(task.metadata.descriptions[splits.d_holdout[task.name][0]])
        else:
            descs.append(task.metadata.descriptions[0])

    with torch.no_grad():
        per_module = hypernet.generate_for_batch(descs)

    cos_sims, l1_model, l1_baseline = [], [], []
    for i, task in enumerate(tasks):
        oracle_pm = load_and_canonicalize_oracle(str(Path(oracle_dir) / task.name), spec, device=device)
        for module in spec.target_modules:
            A_pred, B_pred = per_module[module][0][i], per_module[module][1][i]
            A_orc, B_orc = oracle_pm[module]
            dW_pred = delta_weights(A_pred.unsqueeze(0), B_pred.float().unsqueeze(0), spec.scaling)[0]
            dW_orc = delta_weights(A_orc.unsqueeze(0), B_orc.unsqueeze(0), spec.scaling)[0]
            cos_sims.append(float(F.cosine_similarity(dW_pred.flatten(), dW_orc.flatten(), dim=0)))
            l1_model.append(float((dW_pred - dW_orc).abs().mean()))
            l1_baseline.append(float((dW_orc.mean() - dW_orc).abs().mean()))

    return {
        "cosine_similarity": sum(cos_sims) / len(cos_sims),
        "normalized_l1_model": sum(l1_model) / len(l1_model),
        "normalized_l1_mean_baseline": sum(l1_baseline) / len(l1_baseline),
    }


def train_recon(
    config: ReconConfig,
    hypernet: SteerableHyperLoRA,
    tasks: list[Task],
    oracle_dir: str | Path,
    spec: TargetSpec,
    *,
    splits: Splits | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """No target model loaded at all. Checkpoints periodically (if ``out_dir`` given), with
    the trailing average of the last ``val_freq`` steps' normalizers written into the
    checkpoint's config (not as module buffers -- see the module docstring)."""
    rng = random.Random(config.seed)
    device = next(hypernet.parameters()).device
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=config.lr)
    warmup_steps = max(1, int(config.max_steps * config.warmup_frac))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
    )

    batches = build_recon_batches(tasks, oracle_dir, spec, config.batch_size, rng, device=device)
    normalizer_history: dict[str, list[float]] = {}
    history: list[dict] = []

    for step in range(config.max_steps):
        batch = next(batches)
        uniq, inverse = dedup(batch["descs"])
        h = hypernet.encode(uniq)
        per_module = hypernet.heads_forward(h)
        inverse = inverse.to(next(iter(per_module.values()))[0].device)
        per_module_expanded = {m: (A[inverse], B[inverse]) for m, (A, B) in per_module.items()}

        loss_dict = recon_loss(per_module_expanded, batch["target_A"], batch["target_B"])
        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss"].backward()
        optimizer.step()
        scheduler.step()

        for key, value in loss_dict["normalizers"].items():
            normalizer_history.setdefault(key, []).append(value)

        is_last = step + 1 == config.max_steps
        if (step + 1) % config.val_freq == 0 or is_last:
            metrics = evaluate_recon(hypernet, tasks, oracle_dir, spec, splits, device=device)
            entry = {"step": step + 1, "train_loss": loss_dict["loss"].detach().item(), **metrics}
            history.append(entry)

            if out_dir is not None:
                trailing_normalizers = {
                    key: sum(vals[-config.val_freq:]) / len(vals[-config.val_freq:])
                    for key, vals in normalizer_history.items()
                }
                config.normalizers = trailing_normalizers
                save_checkpoint(
                    Path(out_dir) / "latest.pt", hypernet, hypernet.config, spec,
                    stage="recon", step=step + 1,
                    extra={"recon_config": config.to_dict(), "history": history},
                )

    return {"history": history}
