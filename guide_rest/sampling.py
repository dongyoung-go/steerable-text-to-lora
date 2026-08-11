"""Guide-ReST Step 1+2: Grow (vLLM batched sampling) + Filter (verifier scoring).

Loads one checkpoint into vLLM (round 0: the raw base model; round t>0: the previous
round's merged `checkpoint/` directory -- never a live PEFT adapter, since every round's
Improve step already merges its LoRA onto the base before this script would need it, see
`train.py`), samples `k` completions per question for the task's Grow pool, scores every
completion with `tasks.py`'s verifier, and writes two files: `grow_samples.jsonl` (every
sample, labeled, for `feedback.py`'s Stage 1 to draw incorrect/correct triples from) and
`filtered.jsonl` (correct samples only, for `train.py`'s Improve step).

Also samples the task's fixed **dev pool** in the same vLLM session (no extra model load)
-- a small set of questions disjoint from the Grow pool (see `tasks.py::load_*_dev_pool`),
always unconditioned regardless of condition (never given the Condition-B feedback prefix,
since it's meant to validate the bare question -> completion mapping `train.py` actually
trains). Writes `dev_grow_samples.jsonl`/`dev_filtered.jsonl` analogously. `train.py` uses
`dev_filtered.jsonl` as a fixed, independent early-stopping validation set instead of
carving one out of that round's own (small, freshly-generated) training data -- see
docs/01_train.md's "Dev pool" section for why.

Run as its own `uv run --with vllm==... python sampling.py ...` process per round (see
`run.sh` / `round_loop.py`) -- deliberately not a long-lived engine shared across rounds,
so the vLLM engine's GPU memory is always fully released before the next round's `train.py`
(or the next task/condition's `sampling.py`) starts. See docs/01_train.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tasks import TASKS, build_user_prompt


def set_seed(seed: int) -> None:
    random.seed(seed)


def load_vllm_engine(model_dir: str, gpu_memory_utilization: float, max_model_len: int, seed: int):
    from vllm import LLM

    llm = LLM(
        model=model_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    return llm, llm.get_tokenizer()


def build_chat_prompt(tokenizer, user_content: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def generate_and_score(
    llm, tokenizer, spec, task: str, pool, k: int, feedback_prefix: str,
    temperature: float, top_p: float, top_k: int, max_tokens: int, enable_thinking: bool,
    grow_path: Path, filtered_path: Path,
) -> dict:
    from vllm import SamplingParams

    if not pool:
        grow_path.write_text("")
        filtered_path.write_text("")
        return {"n_total": 0, "n_correct": 0, "filter_pass_rate": float("nan")}

    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens, n=k,
    )
    user_contents = []
    for row in pool:
        bare = build_user_prompt(task, row.question)
        content = f"{feedback_prefix}\n\n{bare}" if feedback_prefix else bare
        user_contents.append(content)
    chat_prompts = [build_chat_prompt(tokenizer, c, enable_thinking) for c in user_contents]

    outputs = llm.generate(chat_prompts, sampling_params, use_tqdm=False)

    grow_path.write_text("")
    filtered_path.write_text("")
    n_correct = 0
    n_total = 0
    for row, output in zip(pool, outputs, strict=True):
        grow_rows = []
        filtered_rows = []
        for completion_output in output.outputs:
            completion = completion_output.text
            correct = spec.verify(completion, row.gold)
            n_total += 1
            n_correct += int(correct)
            grow_rows.append({
                "question": row.question,
                "gold": row.gold,
                "completion": completion,
                "correct": correct,
                "feedback_used": bool(feedback_prefix),
            })
            if correct:
                filtered_rows.append({"question": row.question, "completion": completion})
        append_jsonl(grow_path, grow_rows)
        append_jsonl(filtered_path, filtered_rows)

    return {
        "n_total": n_total, "n_correct": n_correct,
        "filter_pass_rate": (n_correct / n_total) if n_total else float("nan"),
    }


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    spec = TASKS[args.task]
    # Dev pool is reserved first (fixed, small) so the Grow pool can safely default to
    # "everything else" (--grow_pool_size omitted -> None) without the two colliding --
    # see tasks.py::load_*_dev_pool / load_*_grow_pool.
    dev_pool = []
    if args.dev_pool_size > 0:
        dev_pool = spec.load_dev_pool(n=args.dev_pool_size, seed=args.dev_seed)
    pool = spec.load_grow_pool(
        n=args.grow_pool_size, seed=args.pool_seed,
        dev_n=args.dev_pool_size, dev_seed=args.dev_seed,
    )

    feedback_prefix = ""
    if args.feedback_file:
        feedback_path = Path(args.feedback_file)
        if feedback_path.exists():
            feedback_prefix = feedback_path.read_text().strip()

    llm, tokenizer = load_vllm_engine(
        args.checkpoint, args.gpu_memory_utilization, args.max_model_len, args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grow_stats = generate_and_score(
        llm, tokenizer, spec, args.task, pool, args.k, feedback_prefix,
        args.temperature, args.top_p, args.top_k, args.max_tokens, args.enable_thinking,
        out_dir / "grow_samples.jsonl", out_dir / "filtered.jsonl",
    )
    (out_dir / "grow_stats.json").write_text(json.dumps(grow_stats, indent=2))
    print(f"[sampling] task={args.task} pool={len(pool)} k={args.k} pass_rate={grow_stats['filter_pass_rate']:.4f}")

    # Dev pool is always unconditioned (no feedback_prefix), even in Condition B: it
    # validates the bare question -> completion mapping train.py actually fits, and using
    # a different, smaller k keeps its cost modest since it's only for early stopping.
    dev_stats = generate_and_score(
        llm, tokenizer, spec, args.task, dev_pool, args.dev_k, "",
        args.temperature, args.top_p, args.top_k, args.max_tokens, args.enable_thinking,
        out_dir / "dev_grow_samples.jsonl", out_dir / "dev_filtered.jsonl",
    )
    (out_dir / "dev_stats.json").write_text(json.dumps(dev_stats, indent=2))
    print(f"[sampling] task={args.task} dev_pool={len(dev_pool)} dev_k={args.dev_k} "
          f"dev_pass_rate={dev_stats['filter_pass_rate']:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--checkpoint", required=True, help="HF model id or local checkpoint dir")
    p.add_argument("--out_dir", required=True, help="round directory to write grow_samples.jsonl/filtered.jsonl into")
    p.add_argument("--feedback_file", default=None, help="path to feedback.txt to prepend; omit/missing = unconditioned")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--grow_pool_size", type=int, default=None, help="omit (default) to use the full train split minus the dev pool, matching ReST-EM's own setup; pass a value to subsample instead (e.g. for a smoke test)")
    p.add_argument("--pool_seed", type=int, default=0, help="fixes which questions are in the Grow pool, held constant across rounds/conditions")
    p.add_argument("--dev_pool_size", type=int, default=50, help="fixed dev-set size for train.py's early stopping; 0 disables the dev pool entirely")
    p.add_argument("--dev_seed", type=int, default=12345, help="fixes which questions are in the dev pool, held constant across rounds/conditions")
    p.add_argument("--dev_k", type=int, default=4, help="completions per dev question -- smaller than --k since this is only for early stopping, not training data")
    p.add_argument("--seed", type=int, default=0, help="vLLM sampling seed, varies per round/condition")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
