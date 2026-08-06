"""Profile response/prompt token-length distributions before choosing ``inp_max_len``.

The GSM8K responses embed Qwen3-32B ``<think>...</think>`` blocks and are long; silently
truncating them would train on cut-off reasoning and look like a model problem, not a data
problem. This script formats and pair-encodes every row through the *same* code path training
uses (``data.formatting.format_example`` / ``tokenize_pair``), reports percentiles, and for a
list of candidate ``inp_max_len`` values reports what fraction of rows would have their
*response* truncated. It only needs a tokenizer (no model weights), so it runs identically on
CPU or a GPU node.

Output is advisory: it prints a table and a recommendation, it does not write any config.

    python scripts/profile_lengths.py --tasks-root /home/dg793/text-to-lora/tasks \
        --train-tasks 'textgrad_repro_gsm8k_*' --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
        --candidate-max-lens 512 1024 1536 2048
"""

from __future__ import annotations

import argparse

from steerable_t2l.data.formatting import format_example, tokenize_pair
from steerable_t2l.data.registry import discover_tasks


def percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {p: float("nan") for p in ("p50", "p90", "p95", "p99", "max")}
    xs = sorted(values)
    n = len(xs)

    def pct(p: float) -> float:
        idx = min(n - 1, int(round(p / 100 * (n - 1))))
        return float(xs[idx])

    return {"p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99), "max": float(xs[-1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--train-tasks", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--candidate-max-lens", type=int, nargs="+", default=[512, 1024, 1536, 2048])
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    tasks = discover_tasks(args.tasks_root, args.train_tasks)
    if not tasks:
        print(f"no tasks found under {args.tasks_root} matching {args.train_tasks}")
        return 1

    import datasets as hf_datasets

    overall_response_lens: list[int] = []
    overall_total_lens: list[int] = []
    per_task_stats = {}

    for task in tasks:
        raw = hf_datasets.load_dataset(**task.metadata.ds_kwargs)
        response_lens, total_lens = [], []
        for row in raw:
            prompt_text, response_text = format_example(row, task.metadata, tokenizer)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
            response_lens.append(len(response_ids))
            total_lens.append(len(prompt_ids) + len(response_ids))
        per_task_stats[task.name] = (percentiles(response_lens), percentiles(total_lens), len(raw))
        overall_response_lens.extend(response_lens)
        overall_total_lens.extend(total_lens)

    print(f"=== {len(tasks)} tasks, {len(overall_response_lens)} rows total\n")
    print(f"{'task':<32} {'rows':>6}  {'resp p50/p90/p95/p99/max':>40}")
    for name, (resp_pct, _, n_rows) in per_task_stats.items():
        line = "/".join(f"{resp_pct[p]:.0f}" for p in ("p50", "p90", "p95", "p99", "max"))
        print(f"{name:<32} {n_rows:>6}  {line:>40}")

    overall_resp_pct = percentiles(overall_response_lens)
    overall_total_pct = percentiles(overall_total_lens)
    print(f"\n{'overall response':<20} {overall_resp_pct}")
    print(f"{'overall prompt+response':<20} {overall_total_pct}")

    print("\n=== truncation under candidate inp_max_len values (response-token fraction cut)")
    for cand in sorted(args.candidate_max_lens):
        n_trunc = 0
        n_total = 0
        for task in tasks:
            raw = hf_datasets.load_dataset(**task.metadata.ds_kwargs)
            for row in raw:
                prompt_text, response_text = format_example(row, task.metadata, tokenizer)
                tok = tokenize_pair(tokenizer, prompt_text, response_text, cand)
                n_total += 1
                n_trunc += int(tok.response_truncated)
        frac = n_trunc / n_total if n_total else float("nan")
        print(f"  inp_max_len={cand:<6} response truncated: {n_trunc}/{n_total} ({frac:.2%})")

    print(
        f"\nrecommendation: inp_max_len >= overall p99 prompt+response length "
        f"({overall_total_pct['p99']:.0f}); round up to a convenient value with headroom."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
