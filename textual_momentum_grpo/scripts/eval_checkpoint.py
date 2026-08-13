"""Offline unconditioned-accuracy eval for a tmgrpo/verl checkpoint against MATH-style eval sets.

Per README section 5 ("Eval unconditioned (no directive/critique at test time) for all arms") and
docs/build_and_run_guide.md section 7, this scores a saved checkpoint's boxed-answer accuracy on
data/eval/{math500,aime24,olympiad_slice}.jsonl (or any other {"prompt","label"} jsonl file, e.g.
a custom held-out set). Arm-agnostic by construction: eval is always unconditioned, plain-prompt
generation, regardless of which arm produced the checkpoint (floor / instance-critique /
trajectory-momentum) -- there is no arm-specific eval logic, only --checkpoint and --arm-name (the
latter used purely for labeling the report).

Two stages:
  1. Merge: verl saves FSDP shards (.../global_step_N/actor/model_world_size_*_rank_*.pt), not a
     loadable HF model. `resolve_hf_model_dir` merges those via `verl.model_merger` into
     checkpoints_hf/<arm-name>_<global_step>/ (cached -- reused on repeat runs unless
     --force-remerge). Pointing --checkpoint directly at an already-merged HF dir skips this.
  2. Generate + score: loads the merged model into vLLM, generates with the same chat-template
     settings as training (enable_thinking=false, response_length=2048 by default), extracts the
     boxed answer via tmgrpo.reward.check_answer, and writes a JSON report per eval file plus an
     overall summary.

Must run under the GPU stack (`.venv-verl/bin/python`), not the CPU-only project `.venv` -- this
imports torch/vllm/transformers, which the CPU venv deliberately excludes (see pyproject.toml).
Requires a free GPU (run after the training job has released it, or on a separate allocation).

Usage:
    .venv-verl/bin/python scripts/eval_checkpoint.py \
        --checkpoint checkpoints/tmgrpo/arm1_floor/global_step_300/actor \
        --arm-name arm1_floor \
        --out eval_results/arm1_floor_step300.json

    # defaults to data/eval/{math500,aime24,olympiad_slice}.jsonl; override with --eval:
    .venv-verl/bin/python scripts/eval_checkpoint.py \
        --checkpoint checkpoints/tmgrpo/arm5_trajectory_on/global_step_300/actor \
        --arm-name arm5_trajectory_on \
        --eval data/eval/math500.jsonl \
        --out eval_results/arm5_trajectory_on_step300.json

    # re-score an already-merged HF dir (skips the merge step):
    .venv-verl/bin/python scripts/eval_checkpoint.py \
        --checkpoint checkpoints_hf/arm1_floor_global_step_300 --arm-name arm1_floor \
        --out eval_results/arm1_floor_step300_rerun.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EVAL_FILES = [
    REPO_ROOT / "data" / "eval" / "math500.jsonl",
    REPO_ROOT / "data" / "eval" / "aime24.jsonl",
    REPO_ROOT / "data" / "eval" / "olympiad_slice.jsonl",
]
DEFAULT_HF_CACHE_DIR = REPO_ROOT / "checkpoints_hf"
DEFAULT_MAX_PROMPT_LENGTH = 1024
DEFAULT_MAX_NEW_TOKENS = 2048


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    """Load a {"prompt","label"} jsonl eval file (same schema as data/train.jsonl)."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_merged_hf_dir(path: Path) -> bool:
    """True if `path` already looks like a loadable HF model dir (config + weight files), as
    opposed to a raw verl FSDP shard dir."""
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").exists()


def find_fsdp_actor_dir(path: Path) -> Path | None:
    """Locate the verl FSDP `actor` shard dir given either that dir itself or its parent
    (.../global_step_N). Returns None if no FSDP shards are found either place."""
    if any(path.glob("model_world_size_*_rank_*.pt")):
        return path
    actor_subdir = path / "actor"
    if actor_subdir.is_dir() and any(actor_subdir.glob("model_world_size_*_rank_*.pt")):
        return actor_subdir
    return None


def default_merged_dir_name(checkpoint: Path, arm_name: str) -> str:
    """e.g. checkpoints/tmgrpo/arm1_floor/global_step_300/actor + "arm1_floor"
    -> "arm1_floor_global_step_300"."""
    step_match = re.search(r"global_step_\d+", str(checkpoint))
    step_label = step_match.group(0) if step_match else "unknown_step"
    return f"{arm_name}_{step_label}"


def resolve_hf_model_dir(
    checkpoint: Path,
    arm_name: str,
    hf_cache_dir: Path = DEFAULT_HF_CACHE_DIR,
    force_remerge: bool = False,
    merge_fn: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Return a loadable HF model dir for `checkpoint`, merging FSDP shards if necessary.

    `merge_fn(actor_dir, target_dir)` is injected so this function is unit-testable without
    invoking the real (GPU-stack-only) verl.model_merger subprocess.
    """
    if is_merged_hf_dir(checkpoint):
        return checkpoint

    actor_dir = find_fsdp_actor_dir(checkpoint)
    if actor_dir is None:
        raise FileNotFoundError(
            f"{checkpoint} is neither a merged HF model dir (config.json + weights) nor a verl "
            "FSDP checkpoint dir (model_world_size_*_rank_*.pt, optionally under an 'actor' "
            "subdir). Point --checkpoint at one of those."
        )

    target_dir = hf_cache_dir / default_merged_dir_name(checkpoint, arm_name)
    if target_dir.exists() and not force_remerge and is_merged_hf_dir(target_dir):
        print(f"reusing cached merged model at {target_dir} (pass --force-remerge to rebuild)")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    if merge_fn is None:
        merge_fn = _merge_fsdp_checkpoint
    merge_fn(actor_dir, target_dir)
    return target_dir


def _merge_fsdp_checkpoint(actor_dir: Path, target_dir: Path) -> None:
    """Shell out to verl's own model merger (python -m verl.model_merger merge --backend fsdp)."""
    print(f"merging FSDP checkpoint {actor_dir} -> {target_dir}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor_dir),
            "--target_dir",
            str(target_dir),
        ],
        check=True,
    )


def build_chat_prompt(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool = False) -> str:
    """Render a chat-format prompt the same way training's
    +data.apply_chat_template_kwargs.enable_thinking=false does."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def score_eval_set(
    rows: list[dict[str, Any]],
    completions_per_row: list[list[str]],
    check_answer_fn: Callable[[str, str], bool],
) -> dict[str, Any]:
    """Score one eval set. Supports n>1 samples/row: reports both mean per-sample accuracy
    (comparable to a single-sample run) and pass@n (fraction of questions with >=1 correct
    sample among n)."""
    if len(rows) != len(completions_per_row):
        raise ValueError(f"{len(rows)} rows but {len(completions_per_row)} completion lists")

    per_question = []
    for row, completions in zip(rows, completions_per_row, strict=True):
        label = row["label"]
        corrects = [check_answer_fn(c, label) for c in completions]
        per_question.append(
            {
                "correct_frac": sum(corrects) / len(corrects) if corrects else 0.0,
                "any_correct": any(corrects),
            }
        )

    n = len(per_question)
    accuracy = statistics.mean(q["correct_frac"] for q in per_question) if n else 0.0
    pass_at_n = sum(q["any_correct"] for q in per_question) / n if n else 0.0
    n_samples = len(completions_per_row[0]) if completions_per_row else 0

    result = {"n_questions": n, "n_samples_per_question": n_samples, "accuracy": accuracy}
    if n_samples > 1:
        result[f"pass_at_{n_samples}"] = pass_at_n
    return result


def build_report(
    arm_name: str,
    checkpoint: str,
    model_dir: str,
    eval_set_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    overall_accuracy = (
        statistics.mean(r["accuracy"] for r in eval_set_results.values()) if eval_set_results else 0.0
    )
    return {
        "arm_name": arm_name,
        "checkpoint": checkpoint,
        "model_dir": model_dir,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_sets": eval_set_results,
        # unweighted mean of per-set accuracy (each eval set counts equally regardless of size),
        # matching how the three sets are reported individually in README section 4's success
        # criteria -- read eval_sets[*] directly if a size-weighted pooled number is wanted.
        "overall_accuracy": overall_accuracy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="verl FSDP checkpoint dir (.../global_step_N or .../global_step_N/actor) or an "
        "already-merged HF model dir.",
    )
    parser.add_argument("--arm-name", required=True, help="Label for the report, e.g. arm1_floor.")
    parser.add_argument(
        "--eval",
        type=Path,
        nargs="+",
        default=DEFAULT_EVAL_FILES,
        help="One or more {\"prompt\",\"label\"} jsonl eval files. Defaults to all three of "
        "data/eval/{math500,aime24,olympiad_slice}.jsonl.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Path to write the JSON report.")
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--force-remerge", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Match training's chat-template setting only if it was also enabled there; "
        "arm1_floor/run_arm*.sh train with enable_thinking=false (this flag's default).",
    )
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Completions per question. 1 = greedy accuracy (default). >1 requires "
        "--temperature > 0 and additionally reports pass@n.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.n_samples > 1 and args.temperature <= 0:
        print("--n-samples > 1 needs --temperature > 0 (greedy decoding always gives 1 unique sample)")
        return 1

    model_dir = resolve_hf_model_dir(
        args.checkpoint, args.arm_name, hf_cache_dir=args.hf_cache_dir, force_remerge=args.force_remerge
    )

    # Imported lazily: only the GPU-stack venv (.venv-verl) has these, and everything above this
    # point is meant to stay importable/testable from the CPU-only project venv.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(REPO_ROOT))
    from tmgrpo.reward import check_answer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_new_tokens,
        n=args.n_samples,
        seed=args.seed,
    )

    eval_set_results = {}
    for eval_path in args.eval:
        rows = load_eval_set(eval_path)
        prompts = [build_chat_prompt(tokenizer, r["prompt"], args.enable_thinking) for r in rows]
        outputs = llm.generate(prompts, sampling_params)
        completions_per_row = [[c.text for c in output.outputs] for output in outputs]
        set_result = score_eval_set(rows, completions_per_row, check_answer)
        eval_set_results[eval_path.stem] = set_result
        print(f"{eval_path.stem}: {json.dumps(set_result)}")

    report = build_report(args.arm_name, str(args.checkpoint), str(model_dir), eval_set_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"overall_accuracy={report['overall_accuracy']:.4f} -- wrote report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
