"""Inference-time iterative TextGrad-style refinement pilot for T2L. See
``docs/07_iterative_t2l_application_v3.md``.

    python scripts/eval_iterative_t2l_v3.py \\
        --hypernet outputs/checkpoints/sft_scratch_v3/latest.pt \\
        --target-dir Qwen/Qwen2.5-1.5B-Instruct \\
        --tasks-root /home/dg793/text-to-lora/tasks \\
        --tasks textgrad_repro_v3_gsm8k_d4 textgrad_repro_v3_aqua_d9 textgrad_repro_v3_strategyqa_d8 \\
        --splits data/splits_v3.json \\
        --feedback-model Qwen/Qwen3-14B \\
        --out outputs/eval/iterative_t2l_v3_pilot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from steerable_t2l.checkpoint import load_hypernet
from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits
from steerable_t2l.eval_accuracy import GenerationConfig
from steerable_t2l.feedback_gen import REWRITE_MODES, load_vllm_engine
from steerable_t2l.iterative_t2l import run_iterative_t2l
from steerable_t2l.target_spec import TargetSpec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypernet", required=True, help="path to a trained hypernet checkpoint (.pt)")
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument(
        "--tasks", nargs="+", required=True,
        help="task-dir names/patterns under --tasks-root, e.g. textgrad_repro_v3_gsm8k_d4",
    )
    ap.add_argument("--splits", required=True)
    ap.add_argument("--feedback-model", default="Qwen/Qwen3-14B")
    ap.add_argument(
        "--feedback-gpu-memory-utilization", type=float, default=0.5,
        help="capped well below vLLM's own default -- the target model and hypernet backbone "
        "also need to fit on the same GPU (docs/07's open item #1, not yet validated)",
    )
    ap.add_argument("--feedback-max-model-len", type=int, default=8192)
    ap.add_argument("--mode", choices=REWRITE_MODES, default="prompt")
    ap.add_argument("--n-rounds", type=int, default=5)
    ap.add_argument("--feedback-n", type=int, default=16)
    ap.add_argument("--score-n", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=2560, help="see docs/04 §4")
    ap.add_argument("--gen-batch-size", type=int, default=16)
    ap.add_argument("--max-words", type=int, default=150, help="comprehensive_feedback mode's word cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--attn-implementation",
        default="kernels-community/flash-attn2@c269cc539ad0c1fc0899abd4b05ecc1303d6c4b1",
        help="see docs/03_training_validation.md's GPU-bugs section; falls back to 'sdpa' if unset",
    )
    ap.add_argument("--out", default="outputs/eval/iterative_t2l_v3_pilot.json")
    args = ap.parse_args()

    with open(args.splits) as f:
        splits = Splits.from_dict(json.load(f))
    tasks = discover_tasks(args.tasks_root, args.tasks)
    if not tasks:
        raise SystemExit(f"no tasks matched {args.tasks!r} under {args.tasks_root}")

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

    print(f"loading feedback model {args.feedback_model} via vLLM ...", flush=True)
    feedback_llm, feedback_tokenizer = load_vllm_engine(
        args.feedback_model, args.feedback_gpu_memory_utilization, args.feedback_max_model_len, args.seed
    )

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reports = []
    for task in tasks:
        print(f"=== {task.name}", flush=True)
        report = run_iterative_t2l(
            hypernet, target, tokenizer, spec, task, splits, tasks,
            feedback_llm, feedback_tokenizer,
            n_rounds=args.n_rounds, feedback_n=args.feedback_n, score_n=args.score_n,
            gen_config=gen_config, mode=args.mode, max_words=args.max_words, seed=args.seed,
        )
        for round_entry in report["rounds"]:
            print(
                f"  round {round_entry['round']}: held_out_accuracy="
                f"{round_entry['held_out_accuracy']:.4f} (used_lora={round_entry['used_lora']})",
                flush=True,
            )
        reports.append(report)
        with open(out_path, "w") as f:
            json.dump({"tasks": reports}, f, indent=2)

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
