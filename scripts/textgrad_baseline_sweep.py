"""Headroom sweep: for each task in textgrad_repro.py's TASKS
registry, 0-shot the seed task description against the val split and print
baseline accuracy.

Renamed from the original textgrad_bbh_baseline_sweep.py now that TASKS
covers more than BBH (MMLU/GPQA/AIME/MultiArith/AQuA/CommonsenseQA/
StrategyQA/TREC too) -- see TEXTGRAD_MULTITASK_PLAN.md section 7. Imports
TASKS/ANSWER_PARSERS directly from textgrad_repro rather than keeping
a second hand-maintained copy of the registry, so adding a task to one
script automatically covers the other.

Full TextGrad training runs (textgrad_repro.py) are only worth
running on tasks that baseline in the 0.3-0.8 band -- the completed GSM8K
run opened at 0.97 and spent nearly every step reverted, because Qwen3 is
far past the gpt-3.5-turbo baseline the paper's headroom was measured
against. This script answers "which tasks have headroom" once, cheaply
(one model load), before spending a full 12-step optimizer run on each.

Needs the same ephemeral overlay as textgrad_repro.py (see
textgrad_repro_run.sh / textgrad_repro_README.md for why):

    uv run --with-editable ./textgrad_repro \\
        --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \\
        --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \\
        --with diskcache --with litellm --with graphviz --with gdown --with tenacity --with python-dotenv \\
        python scripts/textgrad_baseline_sweep.py --model_dir Qwen/Qwen3-14B --no_enable_thinking

GPQA needs HF_TOKEN set (gated dataset, see textgrad_repro.py's
module docstring); AIME needs a much larger max_model_len (handled
automatically per-task via TASKS[...]['min_max_model_len'], same as the
main script).
"""

import argparse

from vllm import LLM, SamplingParams

from textgrad_repro import ANSWER_PARSERS, TASKS, build_chat_prompt, dataset_to_rows


def sweep_task(llm, tokenizer, task_key, enable_thinking):
    spec = TASKS[task_key]
    val_set = spec["loader"]("val")
    rows = dataset_to_rows(val_set)
    parse = ANSWER_PARSERS[spec["parser"]]
    description = spec["task_description"] or val_set.get_task_description()
    max_tokens = spec.get("max_tokens", 2000)

    prompts = [build_chat_prompt(tokenizer, description, r["question_prompt"], enable_thinking) for r in rows]
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens, top_p=0.99, n=1)
    outputs = llm.generate(prompts, sampling_params)
    responses = [o.outputs[0].text for o in outputs]

    n_correct = 0
    for r, response in zip(rows, responses, strict=True):
        predicted = parse(response)
        gold = parse(str(r["answer"]))
        n_correct += int(predicted is not None and predicted == gold)
    accuracy = n_correct / len(rows) if rows else 0.0
    return accuracy, len(rows)


def main(args):
    max_model_len = max(args.max_model_len, max((TASKS[k].get("min_max_model_len") or 0) for k in TASKS))
    llm = LLM(
        args.model_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    tasks = sorted(TASKS) if not args.tasks else args.tasks
    results = []
    for task_key in tasks:
        accuracy, n = sweep_task(llm, tokenizer, task_key, args.enable_thinking)
        in_band = 0.3 <= accuracy <= 0.8
        results.append((task_key, accuracy, n, in_band))
        print(f"{task_key:48s} accuracy={accuracy:.4f} (n={n}){'  [in 0.3-0.8 band]' if in_band else ''}")

    print("\nsummary (tasks in the 0.3-0.8 headroom band):")
    for task_key, accuracy, _n, in_band in results:
        if in_band:
            print(f"  {task_key:48s} {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="Qwen/Qwen3-14B")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384,
        help="bumped automatically to cover any task's min_max_model_len (e.g. aime's 32768)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        choices=sorted(TASKS),
        help="subset of TASKS to run (default: all)",
    )
    parser.add_argument("--enable_thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_enable_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
