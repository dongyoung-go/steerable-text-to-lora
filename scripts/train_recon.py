"""Stage C: reconstruction warm-start. See docs/03_training_validation.md §2, Stage C.

No target model is loaded -- ``TargetSpec.from_pretrained`` needs ``AutoConfig`` only.

    python scripts/train_recon.py --config configs/recon.yaml --hypernet-config configs/hypernet.yaml \
        --target-dir Qwen/Qwen2.5-1.5B-Instruct --oracle-dir outputs/oracle_loras \
        --tasks-root /home/dg793/text-to-lora/tasks --train-tasks 'textgrad_repro_gsm8k_*' \
        --splits data/splits.json --out outputs/checkpoints/recon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.hypernet import HyperNetConfig, SteerableHyperLoRA
from steerable_t2l.target_spec import TargetSpec
from steerable_t2l.trainers.recon import ReconConfig, train_recon


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="ReconConfig yaml")
    ap.add_argument("--hypernet-config", required=True, help="HyperNetConfig yaml")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--train-tasks", nargs="+", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="outputs/checkpoints/recon")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="retrain even if <out>/latest.pt already exists")
    args = ap.parse_args()

    if (Path(args.out) / "latest.pt").exists() and not args.force:
        print(f"skipping recon -- {args.out}/latest.pt already exists (--force to retrain)")
        return 0

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with open(args.splits) as f:
        splits = Splits.from_dict(json.load(f))

    tasks = discover_tasks(args.tasks_root, args.train_tasks)
    trained_tasks = [t for t in tasks if t.name not in splits.t_holdout]
    if not trained_tasks:
        print("no (trained) tasks with oracle adapters to reconstruct against")
        return 1

    with open(args.hypernet_config) as f:
        hypernet_config = HyperNetConfig.from_dict(yaml.safe_load(f))
    spec = TargetSpec.from_pretrained(args.target_dir)

    hypernet = SteerableHyperLoRA(spec, hypernet_config, zero_init=True, dtype=dtype, device=device, seed=args.seed)
    print(hypernet.parameter_report())

    recon_config = ReconConfig.from_yaml(args.config)
    recon_config.seed = args.seed

    result = train_recon(recon_config, hypernet, trained_tasks, args.oracle_dir, spec, splits=splits, out_dir=args.out)

    last = result["history"][-1] if result["history"] else None
    if last:
        print(
            f"\nfinal (step {last['step']}): train_loss={last['train_loss']:.4f} "
            f"cosine_similarity={last.get('cosine_similarity')}"
        )
    print(f"checkpoint written to {args.out}/latest.pt")
    if result.get("best_cosine_similarity") is not None:
        print(
            f"best checkpoint written to {args.out}/best.pt "
            f"(cosine_similarity={result['best_cosine_similarity']:.4f}) -- "
            "prefer this over latest.pt for SFT warm-start if it collapsed late"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
