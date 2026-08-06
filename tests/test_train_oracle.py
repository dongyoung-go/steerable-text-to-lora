"""Stage A: oracle LoRA training. See docs/03_training_validation.md §2."""

from __future__ import annotations

import json

import pytest
import yaml

from steerable_t2l.data.datasets import DataConfig
from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import make_splits
from steerable_t2l.oracle.train_oracle import OracleConfig, build_oracle_peft_model, train_one_oracle


def _make_task(root, name, n_rows=12):
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


def test_build_oracle_peft_model_asserts_config_matches_spec(spec, target_model_for_tokenizer):
    mismatched = OracleConfig(r=spec.r + 1)
    with pytest.raises(AssertionError, match="TargetSpec"):
        build_oracle_peft_model(target_model_for_tokenizer, spec, mismatched)


def test_build_oracle_peft_model_matching_config_succeeds(spec, target_model_for_tokenizer):
    config = OracleConfig(
        r=spec.r, lora_alpha=spec.lora_alpha, use_rslora=spec.use_rslora,
        lora_dropout=spec.lora_dropout, target_modules=spec.target_modules,
    )
    peft_model = build_oracle_peft_model(target_model_for_tokenizer, spec, config)
    assert peft_model is not None
    peft_model.unload()


def test_train_one_oracle_runs_and_saves(tmp_path, tokenizer, spec, target_model_for_tokenizer):
    _make_task(tmp_path, "t0", n_rows=12)
    task = discover_tasks(tmp_path, ["t0"])[0]
    splits = make_splits([task], t_frac=0.0, q_frac=0.25, seed=0)

    oracle_config = OracleConfig(
        r=spec.r, lora_alpha=spec.lora_alpha, use_rslora=spec.use_rslora,
        lora_dropout=spec.lora_dropout, target_modules=spec.target_modules,
        max_steps=4, val_freq=2, batch_size=2, patience=10,
    )
    data_config = DataConfig(
        tasks_root=str(tmp_path), train_tasks=("t0",), inp_max_len=64,
        cache_root=str(tmp_path / "cache"),
    )
    out_dir = tmp_path / "oracle_out" / "t0"

    result = train_one_oracle(
        task, target_model_for_tokenizer, spec, oracle_config, data_config, splits, out_dir, tokenizer
    )

    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "adapter_model.safetensors").exists()
    assert len(result["history"]) >= 1
    assert result["best_val_loss"] is not None

    # target_model must be usable again -- get_peft_model's mutation was undone by unload().
    import torch

    input_ids = torch.randint(0, 200, (1, 8))
    target_model_for_tokenizer(input_ids=input_ids)  # should not raise
