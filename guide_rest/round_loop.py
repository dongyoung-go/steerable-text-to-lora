"""Guide-ReST driver: orchestrates T rounds of Grow -> (Feedback) -> Improve -> Eval for
one task, across Condition A (vanilla ReST) and/or Condition B (Guide-ReST).

Deliberately imports no torch/vllm/transformers/peft of its own -- it only does file I/O
and `subprocess.run([sys.executable, "sampling.py", ...])` calls, one per pipeline step, so
it never holds GPU memory itself and can't create a "orchestrator + child both touching the
GPU" conflict. Each child script (`sampling.py`, `feedback.py`, `train.py`,
`eval_heldout.py`) loads its own model and exits, releasing GPU memory before the next
step starts -- see docs/01_train.md for the full rationale.

Directory layout (see docs/01_train.md):
    data/guide_rest/<task>/<condition>/round_{t}/{grow_samples,filtered,local_feedback}.jsonl,
        feedback.txt, checkpoint/, heldout_eval.json
    data/guide_rest/<task>/<condition>/summary.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run_step(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *extra_args]
    print(f"[round_loop] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def round_dir(data_dir: Path, task: str, condition: str, t: int) -> Path:
    return data_dir / task / condition / f"round_{t}"


def feedback_word_count(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text().strip()
    return len(text.split()) if text else 0


def run_condition(args: argparse.Namespace, condition: str) -> None:
    data_dir = Path(args.data_dir)
    cond_dir = data_dir / args.task / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cond_dir / "summary.jsonl"

    prev_checkpoint = args.base_model
    prev_feedback_file: Path | None = None

    for t in range(args.rounds):
        r_dir = round_dir(data_dir, args.task, condition, t)
        r_dir.mkdir(parents=True, exist_ok=True)
        round_seed = args.seed + t

        # -- Step 1+2: Grow + Filter --
        sampling_args = [
            "--task", args.task, "--checkpoint", str(prev_checkpoint), "--out_dir", str(r_dir),
            "--k", str(args.k),
            "--pool_seed", str(args.pool_seed),
            "--dev_pool_size", str(args.dev_pool_size), "--dev_seed", str(args.dev_seed),
            "--dev_k", str(args.dev_k), "--seed", str(round_seed),
            "--temperature", str(args.grow_temperature), "--max_tokens", str(args.max_tokens),
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
            "--max_model_len", str(args.max_model_len),
        ]
        if args.grow_pool_size is not None:
            sampling_args += ["--grow_pool_size", str(args.grow_pool_size)]
        use_feedback = condition == "B" and t >= 1 and prev_feedback_file is not None
        if use_feedback:
            sampling_args += ["--feedback_file", str(prev_feedback_file)]
        run_step("sampling.py", sampling_args)

        # -- Step 3: Feedback (Condition B only, every round -- including round 0, whose
        # critiques seed feedback_1 for round 1's Grow, per the README's worked example) --
        this_round_feedback_file: Path | None = None
        if condition == "B":
            feedback_args = [
                "--checkpoint", str(prev_checkpoint), "--grow_samples", str(r_dir / "grow_samples.jsonl"),
                "--out_dir", str(r_dir), "--n", str(args.n), "--max_words", str(args.max_words),
                "--seed", str(round_seed), "--temperature", str(args.feedback_temperature),
                "--gpu_memory_utilization", str(args.gpu_memory_utilization),
                "--max_model_len", str(args.max_model_len),
            ]
            if prev_feedback_file is not None:
                feedback_args += ["--prev_feedback", str(prev_feedback_file)]
            run_step("feedback.py", feedback_args)
            this_round_feedback_file = r_dir / "feedback.txt"

        # -- Step 4: Improve (always fresh LoRA from base_model, never from prev_checkpoint) --
        checkpoint_out = r_dir / "checkpoint"
        run_step("train.py", [
            "--task", args.task, "--base_model", args.base_model,
            "--filtered", str(r_dir / "filtered.jsonl"), "--dev_filtered", str(r_dir / "dev_filtered.jsonl"),
            "--out_dir", str(checkpoint_out),
            "--r", str(args.lora_r), "--alpha", str(args.lora_alpha), "--dropout", str(args.lora_dropout),
            "--lr", str(args.lr), "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--max_len", str(args.max_tokens), "--seed", str(round_seed),
        ])

        # -- Step 5: Eval held-out pass@1 --
        heldout_out = r_dir / "heldout_eval.json"
        run_step("eval_heldout.py", [
            "--task", args.task, "--checkpoint", str(checkpoint_out), "--out", str(heldout_out),
            "--heldout_size", str(args.heldout_size), "--heldout_seed", str(args.heldout_seed),
            "--seed", str(round_seed), "--max_tokens", str(args.max_tokens),
            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
            "--max_model_len", str(args.max_model_len),
        ])

        grow_stats = json.loads((r_dir / "grow_stats.json").read_text())
        dev_stats_path = r_dir / "dev_stats.json"
        heldout_eval = json.loads(heldout_out.read_text())
        summary_row = {
            "round": t, "condition": condition,
            "filter_pass_rate": grow_stats["filter_pass_rate"],
            "n_filtered_pairs": sum(1 for _ in open(r_dir / "filtered.jsonl")) if (r_dir / "filtered.jsonl").exists() else 0,
            "dev_pass_rate": json.loads(dev_stats_path.read_text())["filter_pass_rate"] if dev_stats_path.exists() else None,
            "heldout_pass_at_1": heldout_eval["pass_at_1"],
            "feedback_word_count": feedback_word_count(r_dir / "feedback.txt") if condition == "B" else None,
            "used_feedback_this_round": use_feedback,
        }
        with open(summary_path, "a") as f:
            f.write(json.dumps(summary_row) + "\n")
        print(f"[round_loop] {args.task}/{condition} round {t}: {summary_row}")

        prev_checkpoint = checkpoint_out
        if this_round_feedback_file is not None:
            prev_feedback_file = this_round_feedback_file


def main(args: argparse.Namespace) -> None:
    conditions = ["A", "B"] if args.condition == "both" else [args.condition]
    for condition in conditions:
        run_condition(args, condition)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["gsm8k", "math"])
    p.add_argument("--condition", default="both", choices=["A", "B", "both"])
    p.add_argument("--base_model", default="Qwen/Qwen3-14B")
    p.add_argument("--data_dir", default=str(Path(__file__).parent.parent / "data" / "guide_rest"))
    p.add_argument("--rounds", type=int, default=5, help="T")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--grow_pool_size", type=int, default=None, help="omit (default) to use the full train split minus the dev pool, matching ReST-EM's own setup")
    p.add_argument("--pool_seed", type=int, default=0, help="fixes the Grow pool identically across rounds/conditions")
    p.add_argument("--dev_pool_size", type=int, default=50, help="fixed dev-set size for train.py's early stopping, disjoint from the Grow pool; 0 disables it")
    p.add_argument("--dev_seed", type=int, default=12345, help="fixes the dev pool identically across rounds/conditions")
    p.add_argument("--dev_k", type=int, default=4, help="completions per dev question, smaller than --k since this is only for early stopping")
    p.add_argument("--heldout_size", type=int, default=200)
    p.add_argument("--heldout_seed", type=int, default=0)
    p.add_argument("--n", type=int, default=8, help="Stage-1 feedback batch size N; sweep {3,8} on gsm8k before fixing")
    p.add_argument("--max_words", type=int, default=150, help="Stage-2 merge word cap")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3, help="cap; train.py early-stops on validation loss before this if --patience triggers first")
    p.add_argument("--patience", type=int, default=1, help="epochs with no val_loss improvement before train.py stops early")
    p.add_argument("--batch_size", type=int, default=32, help="see train.py's --batch_size help: 32 measured as the throughput sweet spot on a 1x B200")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--grow_temperature", type=float, default=0.7)
    p.add_argument("--feedback_temperature", type=float, default=0.7)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0, help="base seed; round t uses seed + t")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
