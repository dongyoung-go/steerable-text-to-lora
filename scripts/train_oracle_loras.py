"""Train one oracle LoRA per task. See docs/03_training_validation.md §2, Stage A.

Embarrassingly parallel across tasks, so it fans out cleanly however you schedule GPU jobs --
this script itself just loops sequentially. The target model is loaded ONCE and reused across
every task (``train_one_oracle`` unloads each task's adapter before returning), since loading
real target weights is the expensive part at ~3.1 GB.

    python scripts/train_oracle_loras.py --config configs/oracle.yaml \
        --tasks-root /home/dg793/text-to-lora/tasks --train-tasks 'textgrad_repro_gsm8k_*' \
        --target-dir Qwen/Qwen2.5-1.5B-Instruct --data-config configs/data.yaml \
        --splits data/splits.json --out outputs/oracle_loras

    # Run a subset (e.g. for external job-array fan-out):
    python scripts/train_oracle_loras.py ... --tasks textgrad_repro_gsm8k_00 textgrad_repro_gsm8k_01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from steerable_t2l.data.datasets import DataConfig
from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.oracle.train_oracle import OracleConfig, train_one_oracle
from steerable_t2l.target_spec import TargetSpec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="OracleConfig yaml")
    ap.add_argument("--data-config", required=True, help="DataConfig yaml")
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--train-tasks", nargs="+", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--splits", required=True, help="splits.json from scripts/make_splits.py")
    ap.add_argument("--out", default="outputs/oracle_loras")
    ap.add_argument("--tasks", nargs="+", default=None, help="restrict to these task names")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument(
        "--force", action="store_true",
        help="retrain even if a task already has outputs/<out>/<task>/adapter_model.safetensors",
    )
    ap.add_argument(
        "--attn-implementation",
        default="kernels-community/flash-attn2@c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1",
        help=(
            "Real FlashAttention2 via a prebuilt Hub kernel (pinned revision so it works under "
            "HF_HUB_OFFLINE=1 once cached -- see docs/03_training_validation.md's GPU-bugs "
            "section). Falls back to 'sdpa' if unset; measured ~2-3x faster and far lower "
            "memory than sdpa's fallback path for Qwen2's GQA shapes on this stack."
        ),
    )
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    oracle_config = OracleConfig.from_yaml(args.config)
    data_config = DataConfig.from_yaml(args.data_config)

    with open(args.splits) as f:
        splits = Splits.from_dict(json.load(f))

    tasks = discover_tasks(args.tasks_root, args.train_tasks)
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t.name in wanted]
    tasks = [t for t in tasks if t.name not in splits.t_holdout]
    if not tasks:
        print("no (trained) tasks to run oracles for")
        return 1

    out_root = Path(args.out)
    if not args.force:
        done = [t for t in tasks if (out_root / t.name / "adapter_model.safetensors").exists()]
        if done:
            print(f"skipping {len(done)} already-trained task(s) (--force to retrain): "
                  f"{', '.join(t.name for t in done)}")
        tasks = [t for t in tasks if t not in done]
    if not tasks:
        print("nothing to do -- every task already has an oracle adapter")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    spec = TargetSpec.from_pretrained(
        args.target_dir,
        target_modules=oracle_config.target_modules,
        r=oracle_config.r,
        lora_alpha=oracle_config.lora_alpha,
        use_rslora=oracle_config.use_rslora,
        lora_dropout=oracle_config.lora_dropout,
    )

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_dir, dtype=dtype, attn_implementation=args.attn_implementation
    )
    target_model = target_model.to(device)
    for p in target_model.parameters():
        p.requires_grad_(False)
    spec.verify_against(target_model)

    for task in tasks:
        print(f"=== oracle: {task.name}")
        result = train_one_oracle(
            task, target_model, spec, oracle_config, data_config, splits,
            out_root / task.name, tokenizer,
        )
        n_steps = result["history"][-1]["step"] if result["history"] else 0
        print(f"  stopped after {n_steps} steps, best val loss {result['best_val_loss']}")

    print(f"\n{len(tasks)} oracle LoRAs written under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
