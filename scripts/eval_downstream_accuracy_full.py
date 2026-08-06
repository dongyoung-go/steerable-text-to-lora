"""Downstream accuracy eval against each domain's full official test set, not the small
Q-axis holdout. See docs/04_downstream_eval.md §12's "per-task sample sizes are small"
caveat (2-10 rows/task) -- this script scores the same ``base``/``prompted``/
``t2l_train_desc``/``t2l_other_task_desc``/``t2l_gibberish_desc`` conditions
(``eval_downstream_accuracy.py``'s CONDITIONS minus ``oracle``) against gsm8k's real 1,319-row
test set, aqua's real 254-row test set, and each bbh_* task's real ~100-row test split
(``steerable_t2l.data.external_testsets``) -- all disjoint by construction from the data used
to train the hypernetwork/oracle LoRAs (see that module's docstring for why).

``oracle`` is scored only when ``--oracle-dir`` is passed (omit it to skip the condition
entirely, same convention as ``eval_downstream_accuracy.py``). It's a from-scratch LoRA
trained on its own tiny per-task pool (docs/04's oracle condition), so there was no a priori
reason to expect it to generalize past that pool any better than the tiny Q-holdout did --
scoring it here checks that expectation against a much bigger, disjoint test set rather than
assuming it. See docs/04_downstream_eval.md §13's TODO for the reasoning that led to adding
this.

    python scripts/eval_downstream_accuracy_full.py \\
        --hypernet outputs/checkpoints/sft_warmstart_v2/latest.pt \\
        --target-dir Qwen/Qwen2.5-1.5B-Instruct --tasks-root /home/dg793/text-to-lora/tasks \\
        --train-tasks 'textgrad_repro_v2_gsm8k' 'textgrad_repro_v2_aqua' 'textgrad_repro_v2_bbh_*' \\
        --splits data/splits_v2.json --oracle-dir outputs/oracle_loras_v2 \\
        --out outputs/eval/downstream_accuracy_full_v2.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from steerable_t2l.checkpoint import load_hypernet
from steerable_t2l.data.external_testsets import load_external_test_rows
from steerable_t2l.data.registry import Task, discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.eval_accuracy import GenerationConfig, run_downstream_eval
from steerable_t2l.target_spec import TargetSpec

# eval_accuracy.CONDITIONS minus "oracle" -- oracle is appended in main() iff --oracle-dir is
# passed, since (unlike the other four steering/no-steering conditions) it needs a per-task
# trained artifact on disk rather than just a bigger row source. See module docstring.
CONDITIONS = ("base", "prompted", "t2l_train_desc", "t2l_other_task_desc", "t2l_gibberish_desc")


def _external_rows(task: Task) -> list[dict]:
    return load_external_test_rows(task.metadata.domain, task.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypernet", required=True, help="path to a trained hypernet checkpoint (.pt)")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--train-tasks", nargs="+", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument(
        "--oracle-dir", default=None,
        help="raw (non-canonicalized) oracle LoRA dir; omit to skip the oracle condition "
        "(default). Pass e.g. outputs/oracle_loras_v2 to score oracle against the full "
        "official test sets too -- see docs/04 §13's TODO.",
    )
    ap.add_argument("--out", default="outputs/eval/downstream_accuracy_full.json")
    ap.add_argument("--max-new-tokens", type=int, default=2560)
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
        help="ignore any existing --out file and re-run every (task, condition) pair from scratch",
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

    conditions = CONDITIONS + ("oracle",) if args.oracle_dir else CONDITIONS

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if not args.force and out_path.exists():
        with open(out_path) as f:
            existing = json.load(f).get("per_task", {}) or {}
        n_existing = sum(len(conds) for conds in existing.values())
        if n_existing:
            print(f"resuming from {out_path}: {n_existing} (task, condition) pairs already recorded", flush=True)

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
        oracle_dir=args.oracle_dir,
        conditions=conditions,
        gen_config=gen_config, seed=args.seed,
        on_condition_start=_on_start,
        on_condition_done=_on_done,
        existing=existing,
        rows_for_task=_external_rows,
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"=== macro-averaged accuracy over full official test sets ({len(result['per_task'])} tasks)")
    for condition, value in result["overall"].items():
        print(f"  {condition:<24} {value:.4f}" if isinstance(value, float) else f"  {condition:<24} {value}")
    print("=== comparisons (macro-averaged)")
    for key, value in result["comparisons"]["macro"].items():
        print(f"  {key:<32} {value:.4f}" if isinstance(value, float) else f"  {key:<32} {value}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
