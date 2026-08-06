"""Downstream accuracy eval (exact-match GSM8K, real generation). See docs/04_downstream_eval.md.

    python scripts/eval_downstream_accuracy.py --hypernet outputs/checkpoints/sft_warmstart/latest.pt \
        --target-dir Qwen/Qwen2.5-1.5B-Instruct --tasks-root /home/dg793/text-to-lora/tasks \
        --train-tasks 'textgrad_repro_gsm8k_*' --splits data/splits.json \
        --oracle-dir outputs/oracle_loras --out outputs/eval/downstream_accuracy.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from steerable_t2l.checkpoint import load_hypernet
from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.eval_accuracy import GenerationConfig, run_downstream_eval
from steerable_t2l.target_spec import TargetSpec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hypernet", required=True, help="path to a trained hypernet checkpoint (.pt)")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--train-tasks", nargs="+", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--oracle-dir", default=None, help="raw (non-canonicalized) oracle LoRA dir; omit to skip the oracle condition")
    ap.add_argument("--out", default="outputs/eval/downstream_accuracy.json")
    ap.add_argument("--max-new-tokens", type=int, default=2560, help="see docs/04 §4: matches docs/03's 0%%-truncation inp_max_len headroom")
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--attn-implementation",
        default="kernels-community/flash-attn2@c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1",
        help="see docs/03_training_validation.md's GPU-bugs section; falls back to 'sdpa' if unset",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="ignore any existing --out file and re-run every (task, condition) pair from scratch "
        "(default: resume, reusing pairs already recorded in --out -- real generation is expensive)",
    )
    args = ap.parse_args()

    with open(args.splits) as f:
        splits = Splits.from_dict(json.load(f))
    tasks = discover_tasks(args.tasks_root, args.train_tasks)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(
        args.target_dir, dtype=dtype, attn_implementation=args.attn_implementation
    ).to(args.device)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()

    spec = TargetSpec.from_pretrained(args.target_dir)
    spec.verify_against(target)

    hypernet, _ = load_hypernet(args.hypernet, device=args.device, dtype=dtype)
    hypernet.eval()

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if not args.force and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f).get("per_task", {}) or {}
        n_existing = sum(len(conds) for conds in existing.values())
        if n_existing:
            print(f"resuming from {out_path}: {n_existing} (task, condition) pairs already recorded", flush=True)

    # A single (task, condition) pair runs real greedy generation over dozens of rows at up
    # to --max-new-tokens tokens each -- easily minutes. Print progress and flush the JSON
    # after every pair so a long real run is observable and survives an interruption with
    # partial results on disk, rather than going silent until the very end.
    _starts: dict[tuple[str, str], float] = {}
    _partial: dict = {"per_task": {}, "overall": None, "comparisons": None}

    def _on_start(task_name: str, condition: str, n_rows: int) -> None:
        _starts[(task_name, condition)] = time.monotonic()
        print(f"[{time.strftime('%H:%M:%S')}] {task_name} / {condition}: starting ({n_rows} rows)", flush=True)

    def _on_done(task_name: str, condition: str, result_entry) -> None:
        start = _starts.pop((task_name, condition), None)
        if isinstance(result_entry, dict):
            summary = f"accuracy={result_entry['accuracy']:.4f} ({result_entry['n_correct']}/{result_entry['n']})"
        else:
            summary = str(result_entry)
        if start is None:
            print(f"[{time.strftime('%H:%M:%S')}] {task_name} / {condition}: resumed (already done) -- {summary}", flush=True)
        else:
            elapsed = time.monotonic() - start
            print(f"[{time.strftime('%H:%M:%S')}] {task_name} / {condition}: done in {elapsed:.1f}s -- {summary}", flush=True)
        _partial["per_task"].setdefault(task_name, {})[condition] = result_entry
        with open(out_path, "w") as f:
            json.dump(_partial, f, indent=2)

    result = run_downstream_eval(
        hypernet, target, tokenizer, spec, tasks, splits,
        oracle_dir=args.oracle_dir, gen_config=gen_config, seed=args.seed,
        on_condition_start=_on_start,
        on_condition_done=_on_done,
        existing=existing,
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"=== macro-averaged accuracy (over {len(result['per_task'])} tasks)")
    for condition, value in result["overall"].items():
        print(f"  {condition:<24} {value:.4f}" if isinstance(value, float) else f"  {condition:<24} {value}")
    print("=== comparisons (macro-averaged)")
    for key, value in result["comparisons"]["macro"].items():
        print(f"  {key:<32} {value:.4f}" if isinstance(value, float) else f"  {key:<32} {value}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
