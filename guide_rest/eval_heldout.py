"""Guide-ReST Step 5: held-out pass@1 evaluation of a round's checkpoint.

Greedy decoding (temperature 0), question only -- no feedback prefix, in either condition,
per docs/guide_rest_README.md's Metrics section ("evaluated with the question only ... to
measure whether gains compound through fine-tuning rather than just appearing in-sample").
Held-out rows are never part of any round's Grow pool (see `tasks.py`'s separate
`load_*_heldout` split), so this is a genuine out-of-sample check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tasks import TASKS, build_user_prompt


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


def main(args: argparse.Namespace) -> None:
    spec = TASKS[args.task]
    rows = spec.load_heldout(n=args.heldout_size, seed=args.heldout_seed)

    llm, tokenizer = load_vllm_engine(
        args.checkpoint, args.gpu_memory_utilization, args.max_model_len, args.seed
    )
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, n=1)
    prompts = [
        build_chat_prompt(tokenizer, build_user_prompt(args.task, row.question), args.enable_thinking)
        for row in rows
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    per_row = []
    n_correct = 0
    for row, output in zip(rows, outputs, strict=True):
        completion = output.outputs[0].text
        correct = spec.verify(completion, row.gold)
        n_correct += int(correct)
        per_row.append({"question": row.question, "gold": row.gold, "completion": completion, "correct": correct})

    n = len(rows)
    result = {"task": args.task, "checkpoint": args.checkpoint, "n": n, "n_correct": n_correct,
              "pass_at_1": (n_correct / n) if n else float("nan"), "rows": per_row}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[eval_heldout] task={args.task} n={n} pass@1={result['pass_at_1']:.4f}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="path to write heldout_eval.json")
    p.add_argument("--heldout_size", type=int, default=200, help="subsample size; math task's MATH-500 held-out has 500 rows total")
    p.add_argument("--heldout_seed", type=int, default=0, help="fixes which held-out rows are used, held constant across rounds/conditions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
    # See sampling.py's matching os._exit(0) comment -- vLLM's V1 engine teardown can
    # deadlock; output files and flush=True prints above are already durably written.
    os._exit(0)
