"""Loss-based validation protocol. See ``docs/03_training_validation.md`` §4.

No generation, no vLLM, no task-accuracy harness. Every metric comes from the same forward
pass as training (``dedup`` -> ``encode``/``heads_forward`` -> ``build_sites`` -> ``lora_hooks``
-> ``target(**batch)``), scored under ``torch.no_grad()``.

Seven description conditions, all scored on the same held-out questions per task:
``base``, ``oracle``, ``train_descs``, ``eval_descs``, ``unseen_task_descs``,
``other_task_descs``, ``gibberish_descs``. A condition that cannot be evaluated (the D axis
has no held-out paraphrase, the T axis is empty, no oracle has been trained yet) reports the
string ``"n/a"`` -- it is never silently skipped or compared against itself.

The primary metric, ``steering_margin``, is reported against both the ``train_descs`` and
``eval_descs`` denominators when available, since docs/03's formula names a generic "correct
desc" and picking a single one would hide the D-axis-unavailable case.

Recon-stage scoring (ΔW cosine similarity, normalized-L1-vs-mean baseline) is NOT here -- it
needs no target model and no tokenized batches, a fundamentally different shape of input than
the task-loss validation this module is scoped to; see ``trainers/recon.py::evaluate_recon``.
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from pathlib import Path

import torch

from steerable_t2l.data.datasets import DataConfig, PerTaskDescDataset, build_dataloader
from steerable_t2l.data.registry import Task, domains
from steerable_t2l.data.splits import Splits, d_axis_available, resolve_q_holdout
from steerable_t2l.hooks import build_sites, lora_hooks
from steerable_t2l.hypernet import SteerableHyperLoRA
from steerable_t2l.losses import per_sequence_normalized_ce
from steerable_t2l.target_spec import TargetSpec

CONDITIONS = (
    "base",
    "oracle",
    "train_descs",
    "eval_descs",
    "unseen_task_descs",
    "other_task_descs",
    "gibberish_descs",
)

# Lifted verbatim from the reference repo's configs/textgrad_repro_gsm8k.yaml
# `additional_eval_descs` -- the critical control: score(gibberish) should land at ~= base.
GIBBERISH_DESCS = [
    "dogs;cats;bananas;",
    "7@9.qwepra#/.sd,s'2OC^039u#rdagjbL",
    "ggggggggggggggggggggg",
]

NA = "n/a"

_ORACLE_KEY_RE = re.compile(r"\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.(lora_A|lora_B)\.weight$")


def build_condition_descs(
    task: Task,
    splits: Splits,
    all_tasks: list[Task],
    condition: str,
    rng: random.Random,
) -> list[str] | None:
    """The pool of candidate descriptions for ``condition``, or ``None`` if unavailable.

    Callers sample one description per row from the returned pool. ``None`` must never be
    papered over -- it means the D axis has no held-out paraphrase for this task, or the T
    axis is empty, or no "other" task exists to draw from.
    """
    if condition in ("base", "oracle"):
        return None
    if condition == "train_descs":
        held_out = set(splits.d_holdout.get(task.name, []))
        pool = [d for i, d in enumerate(task.metadata.descriptions) if i not in held_out]
        return pool or list(task.metadata.descriptions)
    if condition == "eval_descs":
        if not d_axis_available(splits, task.name):
            return None
        return [task.metadata.descriptions[i] for i in splits.d_holdout[task.name]]
    if condition == "unseen_task_descs":
        # `task` itself is the T-held-out task; every one of its descriptions is unseen.
        return list(task.metadata.descriptions)
    if condition == "other_task_descs":
        others = [t for t in all_tasks if t.name != task.name and t.name not in splits.t_holdout]
        if not others:
            return None
        return list(rng.choice(others).metadata.descriptions)
    if condition == "gibberish_descs":
        return list(GIBBERISH_DESCS)
    raise ValueError(f"unknown condition {condition!r}")


def _load_oracle_per_module(
    oracle_dir: str | Path, spec: TargetSpec, device: str | torch.device = "cpu"
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """A task's raw (non-canonicalized) oracle adapter as ``{module: (A [n_layers,r,in],
    B [n_layers,out,r])}`` -- the un-batched layout ``build_sites`` expands with a batch dim.

    ``device`` must be passed explicitly: ``peft.load_peft_weights`` otherwise calls
    ``infer_device()`` and silently lands on ``"cuda"`` whenever one is visible, regardless of
    where ``target`` actually lives -- works by accident on a CPU-only node, breaks the moment
    a GPU is present. Always pass ``target``'s device here.
    """
    import peft

    raw = peft.load_peft_weights(str(oracle_dir), device=str(device))
    per_layer: dict[str, dict[int, dict[str, torch.Tensor]]] = {m: {} for m in spec.target_modules}
    for key, tensor in raw.items():
        match = _ORACLE_KEY_RE.search(key)
        if not match:
            continue
        layer_idx, module_name, role = match.groups()
        if module_name not in per_layer:
            continue
        per_layer[module_name].setdefault(int(layer_idx), {})[role] = tensor

    out = {}
    for module in spec.target_modules:
        layers = per_layer[module]
        A = torch.stack([layers[layer]["lora_A"] for layer in range(spec.n_layers)])
        B = torch.stack([layers[layer]["lora_B"] for layer in range(spec.n_layers)])
        out[module] = (A, B)
    return out


def _score_batches(target, spec: TargetSpec, val_loader, per_module_fn) -> tuple[float, int]:
    """Mean per-sequence-normalized CE over every row in ``val_loader``.

    ``per_module_fn(batch) -> per_module dict | None``; ``None`` means "no LoRA at all" (the
    ``base`` condition), scored with a plain forward and no hooks.
    """
    total_loss = 0.0
    total_n = 0
    device = next(target.parameters()).device
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            per_module = per_module_fn(batch)
            if per_module is None:
                logits = target(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                sites = build_sites(spec, per_module)
                with lora_hooks(target, sites, spec.scaling):
                    logits = target(input_ids=input_ids, attention_mask=attention_mask).logits
            per_seq = per_sequence_normalized_ce(logits, labels)
        total_loss += float(per_seq.sum())
        total_n += per_seq.numel()
    return (total_loss / total_n if total_n else float("nan")), total_n


def score_condition(
    hypernet: SteerableHyperLoRA | None,
    target,
    spec: TargetSpec,
    task: Task,
    splits: Splits,
    all_tasks: list[Task],
    condition: str,
    val_loader,
    oracle_dir: str | Path | None,
    rng: random.Random,
) -> dict | str:
    """Score one (task, condition) pair. Returns ``{"loss": float, "n": int}`` or ``"n/a"``."""
    if condition == "base":
        loss, n = _score_batches(target, spec, val_loader, lambda batch: None)
        return {"loss": loss, "n": n}

    if condition == "oracle":
        if oracle_dir is None:
            return NA
        adapter_dir = Path(oracle_dir) / task.name
        if not adapter_dir.exists():
            return NA
        per_module_fixed = _load_oracle_per_module(adapter_dir, spec, device=next(target.parameters()).device)

        def _fn(batch, per_module_fixed=per_module_fixed):
            bs = batch["input_ids"].shape[0]
            return {
                m: (A.unsqueeze(0).expand(bs, -1, -1, -1), B.unsqueeze(0).expand(bs, -1, -1, -1))
                for m, (A, B) in per_module_fixed.items()
            }

        loss, n = _score_batches(target, spec, val_loader, _fn)
        return {"loss": loss, "n": n}

    if hypernet is None:
        return NA

    pool = build_condition_descs(task, splits, all_tasks, condition, rng)
    if pool is None:
        return NA

    def _fn(batch, pool=pool):
        bs = batch["input_ids"].shape[0]
        descs = [rng.choice(pool) for _ in range(bs)]
        return hypernet.generate_for_batch(descs)

    loss, n = _score_batches(target, spec, val_loader, _fn)
    return {"loss": loss, "n": n}


def steering_margin(conditions: dict[str, dict | str]) -> dict[str, float] | str:
    """``margin = val_loss(other/gibberish) - val_loss(correct)``, against both denominators
    that exist for a task (``train_descs`` always; ``eval_descs`` only when the D axis is
    available). Positive and growing is the entire thesis (docs/03 §4)."""
    other = conditions.get("other_task_descs")
    gibberish = conditions.get("gibberish_descs")
    out: dict[str, float] = {}
    for denom_name in ("train_descs", "eval_descs"):
        denom = conditions.get(denom_name)
        if not isinstance(denom, dict):
            continue
        if isinstance(other, dict):
            out[f"vs_other_task/{denom_name}"] = other["loss"] - denom["loss"]
        if isinstance(gibberish, dict):
            out[f"vs_gibberish/{denom_name}"] = gibberish["loss"] - denom["loss"]
    return out if out else NA


def _aggregate(per_task: dict[str, dict], condition: str) -> float | str:
    total_loss, total_n = 0.0, 0
    for conds in per_task.values():
        entry = conds.get(condition)
        if isinstance(entry, dict):
            total_loss += entry["loss"] * entry["n"]
            total_n += entry["n"]
    return (total_loss / total_n) if total_n else NA


def run_validation(
    hypernet: SteerableHyperLoRA | None,
    target,
    spec: TargetSpec,
    tasks: list[Task],
    splits: Splits,
    tokenizer,
    data_config: DataConfig,
    *,
    oracle_dir: str | Path | None = None,
    val_batch_size: int = 8,
    seed: int = 0,
) -> dict:
    """Iterate tasks x conditions, aggregate per-task/per-domain/overall, compute the
    steering margin. Returns a flat dict suitable for ``json.dump``/logging.

    ``tasks`` should include every discovered task (both trained and T-held-out) -- this
    function partitions them via ``splits.t_holdout`` itself.
    """
    rng = random.Random(seed)
    t_holdout = set(splits.t_holdout)
    trained_tasks = [t for t in tasks if t.name not in t_holdout]
    holdout_tasks = [t for t in tasks if t.name in t_holdout]
    val_data_config = replace(data_config, val_batch_size=val_batch_size)

    per_task: dict[str, dict[str, dict | str]] = {}

    for task in trained_tasks:
        full_ds = PerTaskDescDataset(
            task, tokenizer, data_config.inp_max_len, cache_root=data_config.cache_root, seed=seed
        )
        q_idx = resolve_q_holdout(splits, task.name, len(full_ds))
        if not q_idx:
            continue
        val_loader = build_dataloader(
            [task], tokenizer, val_data_config, split="val", seed=seed,
            row_indices_by_task={task.name: q_idx},
        )
        per_task[task.name] = {
            condition: score_condition(
                hypernet, target, spec, task, splits, tasks, condition, val_loader, oracle_dir, rng
            )
            for condition in ("base", "oracle", "train_descs", "eval_descs", "other_task_descs", "gibberish_descs")
        }

    for task in holdout_tasks:
        full_ds = PerTaskDescDataset(
            task, tokenizer, data_config.inp_max_len, cache_root=data_config.cache_root, seed=seed
        )
        idx = list(range(len(full_ds)))
        entry = per_task.setdefault(task.name, {})
        if not idx:
            entry["unseen_task_descs"] = NA
            continue
        val_loader = build_dataloader(
            [task], tokenizer, val_data_config, split="val", seed=seed,
            row_indices_by_task={task.name: idx},
        )
        entry["unseen_task_descs"] = score_condition(
            hypernet, target, spec, task, splits, tasks, "unseen_task_descs", val_loader, oracle_dir, rng
        )

    margins = {name: steering_margin(conds) for name, conds in per_task.items()}
    overall = {condition: _aggregate(per_task, condition) for condition in CONDITIONS}

    per_domain: dict[str, dict[str, float | str]] = {}
    for domain, domain_tasks in domains(tasks).items():
        subset = {t.name: per_task[t.name] for t in domain_tasks if t.name in per_task}
        per_domain[domain if domain is not None else "<none>"] = {
            condition: _aggregate(subset, condition) for condition in CONDITIONS
        }

    return {
        "per_task": per_task,
        "steering_margin": margins,
        "overall": overall,
        "per_domain": per_domain,
    }
