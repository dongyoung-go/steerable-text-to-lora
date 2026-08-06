"""PerTaskDescDataset / collate / build_dataloader wiring. See docs/03_training_validation.md §1."""

from __future__ import annotations

import json

import yaml

from steerable_t2l.data.datasets import DataConfig, PerTaskDescDataset, build_dataloader, collate
from steerable_t2l.data.registry import discover_tasks


def _make_task(tmp_path, name, n_rows=6, n_desc=2):
    task_dir = tmp_path / name
    task_dir.mkdir()
    jsonl_path = task_dir / f"{name}.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"question": f"what is {i}+{i}?", "response": f"{2 * i}"}) + "\n")
    metadata = {
        "descriptions": [f"instruction variant {i}" for i in range(n_desc)],
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "response",
        "system_message": "",
        "user_prompt_template": "{question}",
    }
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)


def test_per_task_desc_dataset_draws_desc_independently(tmp_path, tokenizer):
    _make_task(tmp_path, "t0", n_rows=4, n_desc=3)
    task = discover_tasks(tmp_path, ["t0"])[0]
    ds = PerTaskDescDataset(task, tokenizer, inp_max_len=64, cache_root=tmp_path / "cache", seed=0)

    assert len(ds) == 4
    item = ds[0]
    assert set(item) == {"input_ids", "labels", "descs", "task_name"}
    assert item["task_name"] == "t0"
    assert item["descs"] in task.metadata.descriptions


def test_per_task_desc_dataset_caches_to_disk(tmp_path, tokenizer):
    _make_task(tmp_path, "t0", n_rows=3, n_desc=1)
    task = discover_tasks(tmp_path, ["t0"])[0]
    cache_root = tmp_path / "cache"
    ds1 = PerTaskDescDataset(task, tokenizer, inp_max_len=64, cache_root=cache_root, seed=0)
    assert any(cache_root.rglob("rows.pt"))

    # A second construction should hit the cache and produce identical tokenization.
    ds2 = PerTaskDescDataset(task, tokenizer, inp_max_len=64, cache_root=cache_root, seed=1)
    assert len(ds1) == len(ds2)
    for i in range(len(ds1)):
        assert ds1[i]["input_ids"].tolist() == ds2[i]["input_ids"].tolist()


def test_row_indices_restrict_dataset(tmp_path, tokenizer):
    _make_task(tmp_path, "t0", n_rows=6, n_desc=1)
    task = discover_tasks(tmp_path, ["t0"])[0]
    ds = PerTaskDescDataset(
        task, tokenizer, inp_max_len=64, row_indices=[1, 3], cache_root=tmp_path / "cache", seed=0
    )
    assert len(ds) == 2


def test_collate_pads_and_masks(tokenizer):
    import torch

    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([-100, -100, 3]),
            "descs": "a",
            "task_name": "t0",
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "labels": torch.tensor([-100, 5]),
            "descs": "b",
            "task_name": "t1",
        },
    ]
    out = collate(batch, tokenizer)
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["labels"][1].tolist() == [-100, 5, -100]
    assert out["descs"] == ["a", "b"]
    assert out["task_name"] == ["t0", "t1"]


def test_build_dataloader_train_yields_hierarchical_batches(tmp_path, tokenizer):
    _make_task(tmp_path, "t0", n_rows=8, n_desc=1)
    _make_task(tmp_path, "t1", n_rows=8, n_desc=1)
    tasks = discover_tasks(tmp_path, ["t*"])
    config = DataConfig(
        tasks_root=str(tmp_path),
        train_tasks=("t*",),
        inp_max_len=64,
        n_tasks_per_batch=2,
        n_points_per_task=2,
        cache_root=str(tmp_path / "cache"),
    )
    loader = build_dataloader(tasks, tokenizer, config, split="train", seed=0)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4
    assert len(batch["descs"]) == 4


def test_build_dataloader_val_uses_row_indices(tmp_path, tokenizer):
    _make_task(tmp_path, "t0", n_rows=8, n_desc=1)
    tasks = discover_tasks(tmp_path, ["t0"])
    config = DataConfig(
        tasks_root=str(tmp_path),
        train_tasks=("t0",),
        inp_max_len=64,
        val_batch_size=2,
        cache_root=str(tmp_path / "cache"),
    )
    loader = build_dataloader(
        tasks, tokenizer, config, split="val", seed=0, row_indices_by_task={"t0": [0, 1, 2]}
    )
    total = sum(b["input_ids"].shape[0] for b in loader)
    assert total == 3
