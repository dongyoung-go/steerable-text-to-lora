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

⚠️ **Late-training collapse hazard.** ``SteerableHyperLoRA._apply_zero_init``'s docstring
warns that ``out_B.weight/bias == 0`` is a dead-gradient fixed point at step 0 by design
(only ``out_B`` moves on step 1, unblocking everything else). Nothing stops the optimizer
from being knocked back into that same fixed point *later* in training -- once it happens,
the same dead-gradient argument applies permanently (no step-1-style unblocking mechanism
exists past step 0), and the loss flatlines at the "predict ~0" value with no exception
raised. Observed in practice: a flat (post-warmup) LR of 5e-4 with no gradient clipping
produced exactly this -- ``cosine_similarity`` climbing for several hundred steps, then one
large step, then permanent collapse to ~0 for the rest of the run, in both the v3 and v4
experiments independently. Hence the cosine-decayed LR and ``max_grad_norm`` clipping below,
and ``best.pt`` checkpoint selection as a safety net in case collapse still happens.

⚠️ **Update: a single global-norm clip does not protect ``heads`` from that hazard.**
Re-running v3 against the fix above (2026-08-12) showed collapse merely delayed (step
700 -> 900) and shallower (peak ``cosine_similarity`` 0.025 -> 0.084), not eliminated.
Root cause: ``clip_grad_norm_(hypernet.parameters(), max_grad_norm)`` computes one norm
across the *whole* ~158M-parameter hypernetwork; the ``heads`` group (``bottleneck``/
``out_A``/``out_B`` -- the exact pathway with the dead-gradient fixed point) is a tiny
fraction of that, so a locally huge update to ``heads`` can leave the *global* norm well
under the clip threshold while still knocking ``out_B`` back toward zero. Fixed by clipping
``heads`` separately, at a much tighter norm (``max_grad_norm_heads``), and by giving
``heads``/``backbone_lora`` their own (lower) learning rates via ``lr_heads``/
``lr_backbone_lora`` instead of one flat ``lr`` for every group -- see
``build_param_groups`` below. (A fourth candidate fix, initializing ``out_B`` near-zero
instead of exactly zero to remove the literal fixed point, was **not** applied: it would
give ``backbone_lora``/``queries``/``refiner``/``shared_decoder`` a nonzero gradient at
step 0, breaking ``tests/test_grad_flow.py::test_step_zero_upstream_is_idle``'s
intentionally-guarded "only ``out_B`` moves first" contract -- see that test's docstring.)
"""

from __future__ import annotations

import math
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
    lr: float = 2e-4
    lr_backbone_lora: float = 5e-5
    lr_heads: float = 5e-5
    max_steps: int = 2000
    warmup_frac: float = 0.03
    max_grad_norm: float = 1.0
    max_grad_norm_heads: float = 0.1
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

    The cache is kept on CPU regardless of ``device`` -- v3's "one task dir per instruction"
    architecture can put hundreds of tasks in ``tasks`` (vs. v2's few dozen), and holding every
    task's oracle A/B on GPU simultaneously is enough by itself to OOM a step before it even
    runs (observed: 434 tasks left ~176GB resident before the first ``hypernet.encode`` call).
    Only the ``batch_size``-sized draw actually used by a given step is moved to ``device``.

    Yields ``{"descs": list[str], "target_A": {module: [bs, n_layers, r, in]},
    "target_B": {module: [bs, n_layers, out, r]}}`` already on ``device``.
    """
    oracle_cache = {
        task.name: load_and_canonicalize_oracle(str(Path(oracle_dir) / task.name), spec, device="cpu")
        for task in tasks
    }
    while True:
        chosen_tasks = [rng.choice(tasks) for _ in range(batch_size)]
        descs = [rng.choice(task.metadata.descriptions) for task in chosen_tasks]
        target_A = {
            m: torch.stack([oracle_cache[task.name][m][0] for task in chosen_tasks]).to(device)
            for m in spec.target_modules
        }
        target_B = {
            m: torch.stack([oracle_cache[task.name][m][1] for task in chosen_tasks]).to(device)
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


def build_param_groups(hypernet: SteerableHyperLoRA, config: ReconConfig) -> list[dict]:
    """``backbone_lora`` and ``heads`` each get their own (lower) learning rate; ``queries``/
    ``refiner``/``shared_decoder`` share the base ``lr`` -- see the module docstring's
    "global-norm clip does not protect heads" update for why ``heads`` in particular needs a
    gentler rate, not just a tighter clip.
    """
    groups = hypernet.parameter_groups()
    rest = [p for name in ("queries", "refiner", "shared_decoder") for p in groups[name]]
    return [
        {"params": groups["backbone_lora"], "lr": config.lr_backbone_lora, "name": "backbone_lora"},
        {"params": groups["heads"], "lr": config.lr_heads, "name": "heads"},
        {"params": rest, "lr": config.lr, "name": "rest"},
    ]


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
    groups = hypernet.parameter_groups()
    heads_params = groups["heads"]
    rest_params = [p for name in ("backbone_lora", "queries", "refiner", "shared_decoder") for p in groups[name]]
    optimizer = torch.optim.AdamW(build_param_groups(hypernet, config))
    warmup_steps = max(1, int(config.max_steps * config.warmup_frac))

    def lr_lambda(step: int) -> float:
        # Warmup, then cosine decay to 0 by max_steps -- NOT flat after warmup. A flat 5e-4 LR
        # for the remainder of the run is what let one late large step knock out_B back to ~0,
        # re-triggering the dead-gradient fixed point _apply_zero_init warns about (see module
        # docstring). Decaying keeps late steps small enough not to re-cross that point.
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, config.max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    batches = build_recon_batches(tasks, oracle_dir, spec, config.batch_size, rng, device=device)
    normalizer_history: dict[str, list[float]] = {}
    history: list[dict] = []
    best_cosine_similarity: float | None = None

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
        # heads gets its own, much tighter clip -- a global clip over the whole ~158M-param
        # model can leave a locally huge update to heads (the dead-gradient-fixed-point
        # pathway) well under threshold even while it knocks out_B back toward zero. See the
        # module docstring's "global-norm clip does not protect heads" update.
        torch.nn.utils.clip_grad_norm_(heads_params, config.max_grad_norm_heads)
        torch.nn.utils.clip_grad_norm_(rest_params, config.max_grad_norm)
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

                # Separate best-by-cosine_similarity checkpoint: recon can still collapse late
                # (cosine_similarity -> ~0) despite the stability fixes above, and latest.pt
                # alone would silently hand that collapsed state to SFT warm-start. cosine_
                # similarity (not normalized_l1) is the selection metric because L1 loss can
                # look deceptively low even at ~0 similarity -- it locks onto the "predict the
                # mean" baseline value, which is itself small (see module docstring).
                cos_sim = metrics.get("cosine_similarity")
                if isinstance(cos_sim, (int, float)) and (
                    best_cosine_similarity is None or cos_sim > best_cosine_similarity
                ):
                    best_cosine_similarity = cos_sim
                    save_checkpoint(
                        Path(out_dir) / "best.pt", hypernet, hypernet.config, spec,
                        stage="recon", step=step + 1,
                        extra={
                            "recon_config": config.to_dict(), "history": history,
                            "best_step": step + 1, "best_cosine_similarity": cos_sim,
                        },
                    )

    return {"history": history, "best_cosine_similarity": best_cosine_similarity}
