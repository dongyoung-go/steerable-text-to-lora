"""Regression test for the reference's grad-accum bug.

The reference calls ``optimizer.zero_grad()`` inside the accumulate block right before
``backward()``, silently negating accumulation. ``trainers/sft.py::train_sft`` must call
``optimizer.step()`` (once per ``grad_accum`` microbatches) strictly before
``optimizer.zero_grad()``. See docs/03_training_validation.md §3.
"""

from __future__ import annotations

import functools
import json

import torch
import yaml

from steerable_t2l.data.datasets import DataConfig
from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.trainers.sft import SFTConfig, train_sft


def _write_task(root, name="t0", n_rows=8):
    task_dir = root / name
    task_dir.mkdir()
    jsonl_path = task_dir / f"{name}.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"question": f"q{i}", "response": f"r{i}"}) + "\n")
    metadata = {
        "descriptions": ["do the thing"],
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "response",
        "system_message": "",
        "user_prompt_template": "{question}",
    }
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)


def test_optimizer_step_precedes_zero_grad(tmp_path, hypernet, target_model_for_tokenizer, spec, tokenizer, monkeypatch):
    _write_task(tmp_path)
    tasks = discover_tasks(tmp_path, ["t0"])
    splits = Splits(q_frac=0.0, d_holdout={"t0": []}, t_holdout=[], seed=0)
    data_config = DataConfig(
        tasks_root=str(tmp_path), train_tasks=("t0",), inp_max_len=32,
        n_tasks_per_batch=1, n_points_per_task=2, cache_root=str(tmp_path / "cache"),
    )

    call_order: list[str] = []
    orig_step = torch.optim.AdamW.step
    orig_zero_grad = torch.optim.AdamW.zero_grad

    @functools.wraps(orig_step)
    def step_spy(self, *args, **kwargs):
        call_order.append("step")
        return orig_step(self, *args, **kwargs)

    @functools.wraps(orig_zero_grad)
    def zero_grad_spy(self, *args, **kwargs):
        call_order.append("zero_grad")
        return orig_zero_grad(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step_spy)
    monkeypatch.setattr(torch.optim.AdamW, "zero_grad", zero_grad_spy)

    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="no", gradient_accumulation_steps=2)
    config = SFTConfig(
        max_steps=1, grad_accum=2, n_tasks_per_batch=1, n_points_per_task=2,
        val_freq=10_000, l2_reg_generated_w=1e-3,
    )

    train_sft(
        config, hypernet, target_model_for_tokenizer, spec, tasks, splits, tokenizer, data_config,
        accelerator=accelerator, out_dir=None,
    )

    assert call_order.count("step") == 1
    assert call_order.count("zero_grad") == 1
    assert call_order.index("step") < call_order.index("zero_grad")
