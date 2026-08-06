"""Build task dirs (metadata.yaml + question/response jsonl) from the new, non-<think>,
multi-domain textgrad-repro dataset at data/textgrad_repro/qwen-qwen3-14b_*_textgrad-repro/.

Read-only w.r.t. the source dataset -- writes only to --jsonl-out and --tasks-out (new
locations, never data/textgrad_repro/ itself).

For each source dir:
  - picks the iterations.jsonl round with the highest val_accuracy (ties -> latest round)
  - takes that round's forward_outputs.jsonl rows with split == "val", and (unless
    --no-filter-correct) correct == True. The correctness filter exists because these rows
    double as SFT training targets (response_field: "response", consumed by oracle Stage A and
    hypernetwork SFT) -- keeping incorrect rows would train the model to imitate wrong
    reasoning. The cost is that harder tasks lose most of their rows to the filter (e.g.
    bbh_dyck_languages: 19/100 correct at its best iteration) and the eval-side held-out split
    inherits that same "teacher already got this right" bias. --no-filter-correct keeps every
    val row (whether correct or not) for tasks where a larger, unbiased-by-correctness pool
    matters more than clean SFT targets; --min-samples (only enforced when the correctness
    filter is on) drops a task's output entirely if too few rows survive the filter, rather than
    silently shipping a near-empty task dir.
  - drops any row whose model_response still contains "<think>" (defense in depth --
    docs/03's live training data was ~98% <think>-prefixed; this dataset is supposed to be
    clean, this just makes that a guarantee rather than an observation)
  - descriptions = every distinct "prompt" string seen across iterations.jsonl for that task
    (the different textgrad-optimized instruction phrasings -- this is what makes the D-axis
    (docs/03 splits.d_holdout) non-empty for the first time; previously every task had exactly
    one description and d_holdout was always [])
  - each row also carries forward_outputs.jsonl's own "gold_answer" field verbatim (e.g. "C",
    "(E)", "invalid", "2200" -- whatever bare final-answer form that domain uses). This is what
    lets scripts/eval_downstream_accuracy.py score every domain without an external
    dataset join or a per-domain answer-format table: see
    steerable_t2l.eval_accuracy.classify_answer_parser, which infers integer/mcq_letter/exact
    straight from these values and so keeps working unchanged for any domain added here later.

    python scripts/build_tasks_from_textgrad_repro_v2.py \
        --src-root data/textgrad_repro \
        --jsonl-out /home/dg793/text-to-lora/data/textgrad_repro_v2_t2l \
        --tasks-out /home/dg793/text-to-lora/tasks
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

SRC_DIR_RE = re.compile(r"^qwen-qwen3-14b_(?P<task>.+)_textgrad-repro$")


def domain_for(task_name: str) -> str:
    if task_name == "gsm8k":
        return "gsm8k"
    if task_name == "aqua":
        return "aqua"
    if task_name.startswith("bbh_"):
        return "bbh"
    return "other"


def collect_descriptions(iterations_path: Path) -> list[str]:
    seen: list[str] = []
    for line in open(iterations_path):
        prompt = json.loads(line)["prompt"]
        if prompt not in seen:
            seen.append(prompt)
    return seen


def best_iteration_from_forward_outputs(forward_outputs_path: Path) -> int:
    # iterations.jsonl's val_accuracy can be a carried-forward value from an earlier,
    # "reverted" round rather than the accuracy of forward_outputs actually logged at that
    # iteration index -- count real correctness per iteration directly instead of trusting it.
    correct_by_iter: dict[int, int] = {}
    for line in open(forward_outputs_path):
        row = json.loads(line)
        if row["split"] != "val":
            continue
        correct_by_iter[row["iteration"]] = correct_by_iter.get(row["iteration"], 0) + int(row["correct"])
    return max(correct_by_iter, key=lambda it: (correct_by_iter[it], it))


def best_description_index(iterations_path: Path, best_iter: int, descriptions: list[str]) -> int:
    """Index into ``descriptions`` of the prompt that was actually live at ``best_iter`` --
    i.e. the one whose responses became this task's SFT training targets. NOT necessarily
    (and empirically, usually not) ``descriptions[0]``: that's just whichever prompt textgrad
    tried first (its unoptimized seed instruction), since ``collect_descriptions`` preserves
    first-appearance order, not accuracy order. Downstream eval's ``condition_desc`` (see
    docs/04_downstream_eval.md) uses this to score ``prompted``/``t2l_train_desc`` against the
    instruction that was actually optimized/used, not an arbitrary early one.
    """
    for line in open(iterations_path):
        row = json.loads(line)
        if row["iteration"] == best_iter:
            return descriptions.index(row["prompt"])
    raise ValueError(f"iteration {best_iter} not found in {iterations_path}")


def build_one(
    src_dir: Path,
    task_name: str,
    jsonl_out_dir: Path,
    tasks_out_dir: Path,
    filter_correct: bool,
    min_samples: int,
) -> dict:
    descriptions = collect_descriptions(src_dir / "iterations.jsonl")
    best_iter = best_iteration_from_forward_outputs(src_dir / "forward_outputs.jsonl")
    best_desc_idx = best_description_index(src_dir / "iterations.jsonl", best_iter, descriptions)

    rows: list[dict] = []
    n_think_dropped = 0
    n_incorrect_dropped = 0
    seen_questions: set[str] = set()
    for line in open(src_dir / "forward_outputs.jsonl"):
        row = json.loads(line)
        if row["iteration"] != best_iter or row["split"] != "val":
            continue
        if filter_correct and not row["correct"]:
            n_incorrect_dropped += 1
            continue
        response = row["model_response"]
        if "<think>" in response:
            n_think_dropped += 1
            continue
        q = row["question"]
        if q in seen_questions:
            continue
        seen_questions.add(q)
        rows.append({"question": q, "response": response, "gold_answer": row["gold_answer"]})

    summary = {
        "task": task_name,
        "best_iteration": best_iter,
        "best_description_index": best_desc_idx,
        "n_descriptions": len(descriptions),
        "n_rows": len(rows),
        "n_think_dropped": n_think_dropped,
        "n_incorrect_dropped": n_incorrect_dropped,
        "dropped_min_samples": False,
    }

    if filter_correct and len(rows) < min_samples:
        summary["dropped_min_samples"] = True
        return summary

    jsonl_out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_out_dir / f"{task_name}.jsonl"
    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    task_dir = tasks_out_dir / f"textgrad_repro_v2_{task_name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "descriptions": descriptions,
        "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
        "response_field": "response",
        "system_message": "",
        "user_prompt_template": "{question}",
        "domain": domain_for(task_name),
        "best_description_index": best_desc_idx,
    }
    with open(task_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default="data/textgrad_repro")
    ap.add_argument("--jsonl-out", required=True)
    ap.add_argument("--tasks-out", required=True)
    ap.add_argument(
        "--filter-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows with correct == True (default: on). --no-filter-correct keeps "
        "every val row regardless of correctness -- more rows per task, but SFT/oracle "
        "training would then be imitating some wrong reasoning traces too.",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="When --filter-correct is on, drop a task entirely (write nothing) if fewer than "
        "this many rows survive the filter, rather than shipping a near-empty task dir. "
        "Ignored when --no-filter-correct is passed. Set to 0 to disable.",
    )
    args = ap.parse_args()

    src_root = Path(args.src_root)
    jsonl_out_dir = Path(args.jsonl_out)
    tasks_out_dir = Path(args.tasks_out)

    summaries = []
    for src_dir in sorted(src_root.iterdir()):
        m = SRC_DIR_RE.match(src_dir.name)
        if not m or not (src_dir / "forward_outputs.jsonl").exists():
            continue
        task_name = m.group("task")
        summaries.append(
            build_one(
                src_dir,
                task_name,
                jsonl_out_dir,
                tasks_out_dir,
                filter_correct=args.filter_correct,
                min_samples=args.min_samples,
            )
        )

    written = [s for s in summaries if not s["dropped_min_samples"]]
    dropped = [s for s in summaries if s["dropped_min_samples"]]

    print(f"{'task':<40} {'best_iter':>9} {'n_desc':>7} {'n_rows':>7} {'think_dropped':>14} {'incorrect_dropped':>18}")
    for s in written:
        print(
            f"{s['task']:<40} {s['best_iteration']:>9} {s['n_descriptions']:>7} {s['n_rows']:>7} "
            f"{s['n_think_dropped']:>14} {s['n_incorrect_dropped']:>18}"
        )
    print(f"\nwrote {len(written)} task dirs under {tasks_out_dir} (jsonl data under {jsonl_out_dir})")

    if dropped:
        print(f"\ndropped {len(dropped)} task(s) below --min-samples={args.min_samples} (filter_correct=on):")
        for s in dropped:
            print(f"  {s['task']}: {s['n_rows']} rows after correctness filter")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
