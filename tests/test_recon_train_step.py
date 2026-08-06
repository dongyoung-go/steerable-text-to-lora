"""Recon training loop wiring: dedup -> encode -> heads_forward -> normalized L1 vs oracles.

See docs/03_training_validation.md §2, Stage C.
"""

from __future__ import annotations

import torch
from peft import get_peft_model

from steerable_t2l.data.metadata import TaskMetadata
from steerable_t2l.data.registry import Task
from steerable_t2l.trainers.recon import ReconConfig, build_recon_batches, evaluate_recon, train_recon


def _task(name, n_desc=1):
    metadata = TaskMetadata(
        descriptions=tuple(f"{name} instruction {i}" for i in range(n_desc)),
        ds_kwargs={"path": "json", "data_files": f"{name}.jsonl", "split": "train"},
        response_field="response",
        user_prompt_template="{question}",
    )
    return Task(name=name, dir=None, metadata=metadata)


def _write_oracle_adapters(tmp_path, tasks, spec, target_model):
    oracle_dir = tmp_path / "oracle"
    for task in tasks:
        peft_model = get_peft_model(target_model, spec.to_lora_config())
        out = oracle_dir / task.name
        out.mkdir(parents=True)
        peft_model.save_pretrained(str(out))
        peft_model.unload()
    return oracle_dir


def test_build_recon_batches_shapes(tmp_path, spec, target_model_for_tokenizer):
    import random

    tasks = [_task("t0", 2), _task("t1", 1)]
    oracle_dir = _write_oracle_adapters(tmp_path, tasks, spec, target_model_for_tokenizer)

    batches = build_recon_batches(tasks, oracle_dir, spec, batch_size=6, rng=random.Random(0))
    batch = next(batches)
    assert len(batch["descs"]) == 6
    for module in spec.target_modules:
        assert batch["target_A"][module].shape == (6, spec.n_layers, spec.r, spec.in_features[module])
        assert batch["target_B"][module].shape == (6, spec.n_layers, spec.out_features[module], spec.r)


def test_evaluate_recon_reports_na_for_no_tasks(hypernet, spec, tmp_path):
    result = evaluate_recon(hypernet, [], tmp_path, spec)
    assert result["cosine_similarity"] == "n/a"


def test_evaluate_recon_computes_finite_metrics(tmp_path, spec, hypernet, target_model_for_tokenizer):
    tasks = [_task("t0", 2)]
    oracle_dir = _write_oracle_adapters(tmp_path, tasks, spec, target_model_for_tokenizer)
    result = evaluate_recon(hypernet, tasks, oracle_dir, spec)
    assert -1.0 <= result["cosine_similarity"] <= 1.0
    assert result["normalized_l1_model"] >= 0
    assert result["normalized_l1_mean_baseline"] >= 0


def test_train_recon_runs_and_produces_finite_loss(tmp_path, spec, hypernet, target_model_for_tokenizer):
    tasks = [_task("t0", 2), _task("t1", 1)]
    oracle_dir = _write_oracle_adapters(tmp_path, tasks, spec, target_model_for_tokenizer)

    config = ReconConfig(lr=1e-2, max_steps=4, warmup_frac=0.0, batch_size=4, val_freq=4, seed=0)
    result = train_recon(config, hypernet, tasks, oracle_dir, spec)

    assert len(result["history"]) == 1
    entry = result["history"][-1]
    assert entry["step"] == 4
    assert torch.isfinite(torch.tensor(entry["train_loss"]))
    assert -1.0 <= entry["cosine_similarity"] <= 1.0


def test_train_recon_writes_checkpoint_with_normalizers(tmp_path, spec, hypernet, target_model_for_tokenizer):
    tasks = [_task("t0", 2)]
    oracle_dir = _write_oracle_adapters(tmp_path, tasks, spec, target_model_for_tokenizer)
    out_dir = tmp_path / "ckpt"

    config = ReconConfig(lr=1e-2, max_steps=3, warmup_frac=0.0, batch_size=4, val_freq=3, seed=0)
    train_recon(config, hypernet, tasks, oracle_dir, spec, out_dir=out_dir)

    payload = torch.load(out_dir / "latest.pt", weights_only=False)
    assert payload["stage"] == "recon"
    assert payload["step"] == 3
    assert "recon_config" in payload
    assert payload["recon_config"]["normalizers"]
    # Normalizers live in the config dict, never as buffers in the state_dict.
    assert not any("normalizer" in k.lower() for k in payload["state_dict"])
