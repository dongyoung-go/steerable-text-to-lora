"""Stage A: one vanilla PEFT LoRA per task on the target model.

See ``docs/03_training_validation.md`` §2. Config is identical to ``TargetSpec``, asserted --
this is what lets ``canonicalize.py`` skip rescaling by ``scaling``. Uses the same data path
as SFT so oracle and hypernetwork see byte-identical text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import yaml
from peft import get_peft_model
from torch.optim import AdamW

from steerable_t2l.data.datasets import DataConfig, PerTaskDescDataset, build_dataloader
from steerable_t2l.data.registry import Task
from steerable_t2l.data.splits import Splits, resolve_q_holdout
from steerable_t2l.losses import per_sequence_normalized_ce
from steerable_t2l.target_spec import TargetSpec


@dataclass
class OracleConfig:
    r: int = 8
    lora_alpha: int = 16
    use_rslora: bool = False
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 1e-4
    max_steps: int = 500
    patience: int = 3
    val_freq: int = 25
    batch_size: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OracleConfig:
        d = dict(d)
        if "target_modules" in d:
            d["target_modules"] = tuple(d["target_modules"])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> OracleConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


class EarlyStopper:
    """docs/03 doesn't specify a stopping rule beyond "early stopping on the task's own
    validation split"; a patience/min-delta counter on the validation loss is the minimal,
    standard choice."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            return False
        if val_loss > self.best + self.min_delta:
            self.counter += 1
        return self.counter >= self.patience


def build_oracle_peft_model(target_model, spec: TargetSpec, config: OracleConfig):
    """``get_peft_model(target_model, spec.to_lora_config())``, after asserting ``config``'s
    LoRA hyperparameters exactly match ``spec``'s -- a mismatch here would silently break
    ``canonicalize.py``'s "do not rescale by scaling" assumption, and would not raise a shape
    error on the square ``q_proj``/``o_proj`` modules, so it is checked explicitly instead.
    """
    mismatches = [
        f"{field}: oracle={getattr(config, field)!r} spec={getattr(spec, field)!r}"
        for field in ("r", "lora_alpha", "use_rslora", "lora_dropout")
        if getattr(config, field) != getattr(spec, field)
    ]
    if tuple(config.target_modules) != tuple(spec.target_modules):
        mismatches.append(
            f"target_modules: oracle={config.target_modules!r} spec={spec.target_modules!r}"
        )
    if mismatches:
        raise AssertionError(
            "OracleConfig must exactly match TargetSpec: " + "; ".join(mismatches)
        )
    return get_peft_model(target_model, spec.to_lora_config())


def _eval_loss(peft_model, val_loader, device) -> float:
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            logits = peft_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            per_seq = per_sequence_normalized_ce(logits, batch["labels"].to(device))
            total_loss += float(per_seq.sum())
            total_n += per_seq.numel()
    return total_loss / total_n if total_n else float("nan")


def train_one_oracle(
    task: Task,
    target_model,
    spec: TargetSpec,
    config: OracleConfig,
    data_config: DataConfig,
    splits: Splits,
    out_dir: str | Path,
    tokenizer,
) -> dict:
    """Trains one task's oracle LoRA on ``target_model`` (already-loaded, frozen weights --
    callers load the target once and reuse it across every task, discarding the adapter after
    each save). Saves via ``peft_model.save_pretrained`` -- keeps PEFT key names, which
    ``canonicalize.py`` and ``validation.py``'s "oracle" condition both depend on.
    """
    peft_model = build_oracle_peft_model(target_model, spec, config)

    full_ds = PerTaskDescDataset(
        task, tokenizer, data_config.inp_max_len, cache_root=data_config.cache_root
    )
    q_idx = resolve_q_holdout(splits, task.name, len(full_ds))

    train_config = replace(data_config, n_tasks_per_batch=1, n_points_per_task=config.batch_size)
    train_loader = build_dataloader(
        [task], tokenizer, train_config, split="train", seed=0, row_indices_by_task={task.name: q_idx}
    )
    val_loader = None
    if q_idx:
        val_config = replace(data_config, val_batch_size=config.batch_size)
        val_loader = build_dataloader(
            [task], tokenizer, val_config, split="val", seed=0, row_indices_by_task={task.name: q_idx}
        )

    optimizer = AdamW((p for p in peft_model.parameters() if p.requires_grad), lr=config.lr)
    stopper = EarlyStopper(patience=config.patience)
    device = next(peft_model.parameters()).device

    history: list[dict] = []
    best_val = float("inf")
    train_iter = iter(train_loader)

    for step in range(config.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        peft_model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = peft_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        loss = per_sequence_normalized_ce(logits, batch["labels"].to(device)).mean()
        loss.backward()
        optimizer.step()

        if val_loader is not None and (step + 1) % config.val_freq == 0:
            peft_model.eval()
            val_loss = _eval_loss(peft_model, val_loader, device)
            history.append({"step": step + 1, "train_loss": loss.detach().item(), "val_loss": val_loss})
            best_val = min(best_val, val_loss)
            if stopper.step(val_loss):
                break

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir))

    # `get_peft_model` mutates `target_model` in place (its Linear submodules become
    # LoRA-wrapped). Unloading restores the original Linear layers so the same target_model
    # instance can be reused for the next task's oracle -- the caller loads target weights
    # once and fans out across every task, not once per task.
    peft_model.unload()

    return {"history": history, "best_val_loss": best_val if history else None}
