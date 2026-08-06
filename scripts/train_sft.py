"""SFT: end-to-end task loss, optionally warm-started from a recon checkpoint.

See docs/03_training_validation.md §3 and its handoff gotchas.

    accelerate launch scripts/train_sft.py --config configs/sft.yaml \
        --hypernet-config configs/hypernet.yaml \
        --target-dir Qwen/Qwen2.5-1.5B-Instruct --tasks-root /home/dg793/text-to-lora/tasks \
        --train-tasks 'textgrad_repro_gsm8k_*' --splits data/splits.json \
        --out outputs/checkpoints/sft_scratch

    accelerate launch scripts/train_sft.py --config configs/sft_warmstart.yaml \
        --hypernet-config configs/hypernet.yaml \
        --target-dir Qwen/Qwen2.5-1.5B-Instruct --tasks-root /home/dg793/text-to-lora/tasks \
        --train-tasks 'textgrad_repro_gsm8k_*' --splits data/splits.json \
        --init-from outputs/checkpoints/recon/latest.pt --out outputs/checkpoints/sft_warmstart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator

from steerable_t2l.checkpoint import load_hypernet
from steerable_t2l.data.datasets import DataConfig
from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.hypernet import HyperNetConfig, SteerableHyperLoRA
from steerable_t2l.target_spec import TargetSpec
from steerable_t2l.trainers.sft import SFTConfig, train_sft


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="SFTConfig yaml")
    ap.add_argument("--hypernet-config", required=True, help="HyperNetConfig yaml (from-scratch arm only)")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--train-tasks", nargs="+", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--init-from", default=None, help="path to a recon checkpoint; omit for from-scratch")
    ap.add_argument("--data-config", required=True, help="DataConfig yaml")
    ap.add_argument("--out", default="outputs/checkpoints/sft")
    ap.add_argument("--oracle-dir", default=None)
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
    ap.add_argument("--force", action="store_true", help="retrain even if <out>/latest.pt already reached max_steps")
    args = ap.parse_args()

    sft_config = SFTConfig.from_yaml(args.config)
    sft_config.init_from = args.init_from

    ckpt_path = Path(args.out) / "latest.pt"
    if ckpt_path.exists() and not args.force:
        existing_step = torch.load(ckpt_path, map_location="cpu", weights_only=False)["step"]
        if existing_step >= sft_config.max_steps:
            print(f"skipping SFT -- {ckpt_path} already reached step {existing_step} "
                  f"(max_steps={sft_config.max_steps}; --force to retrain)")
            return 0
        print(f"{ckpt_path} exists but only reached step {existing_step}/{sft_config.max_steps} -- "
              "resuming mid-run isn't supported, retraining from the start")

    accelerator = Accelerator(mixed_precision="bf16")
    dtype = torch.bfloat16

    with open(args.splits) as f:
        splits = Splits.from_dict(json.load(f))
    tasks = discover_tasks(args.tasks_root, args.train_tasks)
    data_config = DataConfig.from_yaml(args.data_config)

    spec = TargetSpec.from_pretrained(args.target_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        args.target_dir, dtype=dtype, attn_implementation=args.attn_implementation
    )
    for p in target.parameters():
        p.requires_grad_(False)
    spec.verify_against(target)

    warm_started = args.init_from is not None
    if warm_started:
        # Handoff gotcha #1: a warm-started run must not re-zero its recon-trained heads.
        hypernet, payload = load_hypernet(args.init_from, device=accelerator.device, dtype=dtype)
        # Handoff gotcha #3: an oracle/recon run with a different target spec cannot warm-start.
        assert payload["target_spec"] == spec.to_dict(), (
            "recon checkpoint's target_spec does not match this run's TargetSpec"
        )
    else:
        with open(args.hypernet_config) as f:
            hypernet_config = HyperNetConfig.from_dict(yaml.safe_load(f))
        hypernet = SteerableHyperLoRA(
            spec, hypernet_config, zero_init=True, dtype=dtype, device=accelerator.device
        )

    result = train_sft(
        sft_config, hypernet, target, spec, tasks, splits, tokenizer, data_config,
        warm_started=warm_started, accelerator=accelerator, out_dir=args.out, oracle_dir=args.oracle_dir,
    )

    if result["history"]:
        last = result["history"][-1]
        print(f"final (step {last['step']}): ce_loss={last['ce_loss']:.4f} steering_margin={last['steering_margin']}")
    print(f"checkpoint written to {args.out}/latest.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
