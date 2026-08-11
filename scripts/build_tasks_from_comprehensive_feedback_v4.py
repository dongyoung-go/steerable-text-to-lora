"""v4 experiment, stage B: build task dirs (metadata.yaml + question/response jsonl) from the
comprehensive-feedback chains written by scripts/generate_comprehensive_feedback_v4.py, as an
alternative T2L input to build_tasks_from_textgrad_repro_v3.py's prompt-text-based task dirs.

Read-only w.r.t. both data/textgrad_repro/ and --feedback-root -- writes only to --jsonl-out and
--tasks-out (new locations, disjoint from the v3 builder's namespace). Every LoRA target
(question/response pairs) is identical to what the v3 builder would produce for the same
iteration -- only the T2L input description text differs (comprehensive feedback instead of the
literal optimized prompt).

Structural mirror of build_tasks_from_textgrad_repro_v3.py -- same per-group filters (correctness,
<think> drop, (question, response) dedup, --min-samples), same one-task-dir-per-distinct-
description output shape, same rationale (PerTaskDescDataset samples a description uniformly at
random from a task's whole descriptions pool per row, so genuinely different descriptions must
not be pooled under one task -- see that script's docstring for the full argument). The only
difference is what "distinct description" means:

  - v3's builder groups forward_outputs.jsonl's split == "val" rows by their own literal "prompt"
    field text.
  - this script instead joins each such row's "iteration" field against
    comprehensive_feedback_v4.jsonl's "iteration" -> "comprehensive_feedback" mapping (written by
    generate_comprehensive_feedback_v4.py) and groups by THAT text instead.

Because generate_comprehensive_feedback_v4.py deliberately does NOT fold a reverted round's
gradients into the feedback chain (see its docstring), a reverted round's assigned feedback text
is byte-identical to its parent round's -- so those rows land in the same group as the parent's
own rows here, the feedback-side analogue of how v3's builder pools rows that reused an identical
prompt text. Groups whose feedback text is empty (i.e. no real feedback accumulated yet -- only
possible if iteration 0 itself was reverted) are dropped: an empty description carries no signal
worth training a LoRA against.

    python scripts/build_tasks_from_comprehensive_feedback_v4.py \\
        --src-root data/textgrad_repro \\
        --feedback-root data/textgrad_repro_comprehensive_feedback_v4 \\
        --jsonl-out /home/dg793/text-to-lora/data/comprehensive_feedback_v4_t2l \\
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


def load_feedback_by_iteration(feedback_path: Path) -> dict[int, str]:
    by_iteration: dict[int, str] = {}
    with open(feedback_path) as f:
        for line in f:
            row = json.loads(line)
            by_iteration[row["iteration"]] = row["comprehensive_feedback"]
    return by_iteration


def group_val_rows_by_feedback(
    forward_outputs_path: Path, feedback_by_iteration: dict[int, str]
) -> tuple[dict[str, list[dict]], list[str]]:
    """Groups split == "val" rows by the comprehensive-feedback text assigned to their
    "iteration" (skipping rows whose iteration has no feedback entry, e.g. the iteration == -1
    baseline eval, which predates any gradients). Returns (groups, order) where order lists each
    distinct feedback text in first-appearance order (== group index K used in the emitted task
    dir name)."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for line in open(forward_outputs_path):
        row = json.loads(line)
        if row["split"] != "val":
            continue
        feedback = feedback_by_iteration.get(row["iteration"])
        if not feedback:
            continue
        if feedback not in groups:
            groups[feedback] = []
            order.append(feedback)
        groups[feedback].append(row)
    return groups, order


def build_one(
    src_dir: Path,
    feedback_path: Path,
    task_name: str,
    jsonl_out_dir: Path,
    tasks_out_dir: Path,
    filter_correct: bool,
    min_samples: int,
) -> list[dict]:
    feedback_by_iteration = load_feedback_by_iteration(feedback_path)
    groups, order = group_val_rows_by_feedback(src_dir / "forward_outputs.jsonl", feedback_by_iteration)

    summaries: list[dict] = []
    for group_idx, feedback in enumerate(order):
        group_rows = groups[feedback]
        n_source_iterations = len({r["iteration"] for r in group_rows})
        n_val_total = len(group_rows)
        n_correct_total = sum(int(r["correct"]) for r in group_rows)

        rows: list[dict] = []
        n_think_dropped = 0
        n_incorrect_dropped = 0
        seen_pairs: set[tuple[str, str]] = set()
        for row in group_rows:
            if filter_correct and not row["correct"]:
                n_incorrect_dropped += 1
                continue
            response = row["model_response"]
            if "<think>" in response:
                n_think_dropped += 1
                continue
            q = row["question"]
            pair = (q, response)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            rows.append({"question": q, "response": response, "gold_answer": row["gold_answer"]})

        sub_task_name = f"{task_name}_d{group_idx}"
        summary = {
            "task": sub_task_name,
            "n_source_iterations": n_source_iterations,
            "n_val_total": n_val_total,
            "n_correct_total": n_correct_total,
            "n_rows": len(rows),
            "n_think_dropped": n_think_dropped,
            "n_incorrect_dropped": n_incorrect_dropped,
            "dropped_min_samples": False,
        }

        if filter_correct and len(rows) < min_samples:
            summary["dropped_min_samples"] = True
            summaries.append(summary)
            continue

        jsonl_out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = jsonl_out_dir / f"{sub_task_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        task_dir = tasks_out_dir / f"comprehensive_feedback_v4_{sub_task_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "descriptions": [feedback],
            "ds_kwargs": {"path": "json", "data_files": str(jsonl_path), "split": "train"},
            "response_field": "response",
            "system_message": "",
            "user_prompt_template": "{question}",
            "domain": domain_for(task_name),
            "best_description_index": 0,
        }
        with open(task_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

        summaries.append(summary)

    return summaries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default="data/textgrad_repro")
    ap.add_argument("--feedback-root", default="data/textgrad_repro_comprehensive_feedback_v4")
    ap.add_argument("--jsonl-out", required=True)
    ap.add_argument("--tasks-out", required=True)
    ap.add_argument(
        "--filter-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows with correct == True (default: on). --no-filter-correct keeps "
        "every val row regardless of correctness -- more rows per group, but SFT/oracle "
        "training would then be imitating some wrong reasoning traces too.",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="When --filter-correct is on, drop a group entirely (write nothing) if fewer than "
        "this many rows survive the filter, rather than shipping a near-empty task dir. Ignored "
        "when --no-filter-correct is passed. Set to 0 to disable.",
    )
    args = ap.parse_args()

    src_root = Path(args.src_root)
    feedback_root = Path(args.feedback_root)
    jsonl_out_dir = Path(args.jsonl_out)
    tasks_out_dir = Path(args.tasks_out)

    all_summaries: list[dict] = []
    for src_dir in sorted(src_root.iterdir()):
        m = SRC_DIR_RE.match(src_dir.name)
        if not m or not (src_dir / "forward_outputs.jsonl").exists():
            continue
        task_name = m.group("task")
        feedback_path = feedback_root / task_name / "comprehensive_feedback_v4.jsonl"
        if not feedback_path.exists():
            print(f"  {task_name}: no comprehensive_feedback_v4.jsonl under {feedback_root}, skipping "
                  f"(run scripts/generate_comprehensive_feedback_v4.py first)")
            continue
        all_summaries.extend(
            build_one(
                src_dir,
                feedback_path,
                task_name,
                jsonl_out_dir,
                tasks_out_dir,
                filter_correct=args.filter_correct,
                min_samples=args.min_samples,
            )
        )

    written = [s for s in all_summaries if not s["dropped_min_samples"]]
    dropped = [s for s in all_summaries if s["dropped_min_samples"]]

    print(
        f"{'task':<48} {'n_src_iters':>11} {'n_val_total':>11} {'n_correct':>9} {'n_rows':>7} "
        f"{'think_dropped':>14} {'incorrect_dropped':>18}"
    )
    for s in written:
        print(
            f"{s['task']:<48} {s['n_source_iterations']:>11} {s['n_val_total']:>11} "
            f"{s['n_correct_total']:>9} {s['n_rows']:>7} {s['n_think_dropped']:>14} "
            f"{s['n_incorrect_dropped']:>18}"
        )
    print(f"\nwrote {len(written)} task dirs under {tasks_out_dir} (jsonl data under {jsonl_out_dir})")

    if dropped:
        print(f"\ndropped {len(dropped)} feedback group(s) below --min-samples={args.min_samples} (filter_correct=on):")
        for s in dropped:
            print(f"  {s['task']}: {s['n_rows']} rows after correctness filter")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
