"""Build task dirs (metadata.yaml + question/response jsonl) from the gepa-repro dataset at
data/gepa_repro/qwen-qwen3-14b_*_gepa-repro/. v3 of build_tasks_from_gepa_repro_v2.py: one task
dir (one oracle LoRA) PER DISTINCT CANDIDATE INSTRUCTION seen in a task's search tree, not one
task dir per task with a single Pareto-best candidate plus a pool of alternate-candidate
descriptions. Parallel to build_tasks_from_textgrad_repro_v3.py -- see that file's docstring for
the full rationale (PerTaskDescDataset samples a description at random per training row,
independent of which description produced that row's response; pooling genuinely different
instructions under one task's descriptions list would teach false instruction->response
associations; the fix is "LoRA per description", not "LoRA per task").

Read-only w.r.t. the source dataset -- writes only to --jsonl-out and --tasks-out (new
locations, never data/gepa_repro/ itself).

For each source dir:
  - gepa has no per-row "split" field, so val membership is determined by cross-referencing
    row["question"] against val_set.jsonl's "question_prompt" values (the actual held-out val
    set) -- this is the gepa analog of textgrad's row["split"] == "val" filter.
  - groups those val rows by their own literal "candidate" field text, in first-appearance
    order. Every candidate that was ever forward-passed against the val set (not just the
    Pareto-best one gepa itself selected) becomes its own instruction group -- gepa's tree search
    can re-evaluate a candidate as a parent for multiple mutation attempts, so the same candidate
    text can legitimately have rows from more than one point in the search.
  - for each distinct-instruction group, in first-appearance order (group index 0, 1, 2, ...):
      - keeps rows with (unless --no-filter-correct) correct == True, same rationale as the
        textgrad builder (rows double as SFT/oracle training targets).
      - drops any row whose model_response still contains "<think>".
      - dedupes by (question, response) pair -- NOT question alone -- so a candidate reused
        across multiple search points actually grows the group's row count instead of being
        collapsed back to one-per-question.
      - --min-samples (only enforced when --filter-correct is on) drops that instruction's group
        entirely (writes nothing) if too few rows survive.
  - writes one task dir per surviving group: <tasks-out>/gepa_repro_v3_<task>_d<K>/, K being that
    group's first-appearance index, with metadata.yaml's "descriptions": [that one candidate
    text] and "best_description_index": 0.
  - each row also carries forward_outputs.jsonl's own "gold_answer" field verbatim, same as the
    textgrad builder.

    python scripts/build_tasks_from_gepa_repro_v3.py \
        --src-root data/gepa_repro \
        --jsonl-out /home/dg793/text-to-lora/data/gepa_repro_v3_t2l \
        --tasks-out /home/dg793/text-to-lora/tasks
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

SRC_DIR_RE = re.compile(r"^qwen-qwen3-14b_(?P<task>.+)_gepa-repro$")


def domain_for(task_name: str) -> str:
    if task_name == "gsm8k":
        return "gsm8k"
    if task_name == "aqua":
        return "aqua"
    if task_name.startswith("bbh_"):
        return "bbh"
    return "other"


def load_val_questions(val_set_path: Path) -> set[str]:
    return {json.loads(line)["question_prompt"] for line in open(val_set_path)}


def group_val_rows_by_candidate(
    forward_outputs_path: Path, val_questions: set[str]
) -> tuple[dict[str, list[dict]], list[str]]:
    """Groups rows whose question is in the held-out val set by their own literal "candidate"
    text, in first-appearance order. Returns (groups, order) where order lists each distinct
    candidate text in the sequence it was first seen (== group index K used in the emitted task
    dir name)."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for line in open(forward_outputs_path):
        row = json.loads(line)
        if row["question"] not in val_questions:
            continue
        candidate = row["candidate"]
        if candidate not in groups:
            groups[candidate] = []
            order.append(candidate)
        groups[candidate].append(row)
    return groups, order


def build_one(
    src_dir: Path,
    task_name: str,
    jsonl_out_dir: Path,
    tasks_out_dir: Path,
    filter_correct: bool,
    min_samples: int,
) -> list[dict]:
    val_questions = load_val_questions(src_dir / "val_set.jsonl")
    groups, order = group_val_rows_by_candidate(src_dir / "forward_outputs.jsonl", val_questions)

    summaries: list[dict] = []
    for group_idx, candidate in enumerate(order):
        group_rows = groups[candidate]
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

        task_dir = tasks_out_dir / f"gepa_repro_v3_{sub_task_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "descriptions": [candidate],
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
    ap.add_argument("--src-root", default="data/gepa_repro")
    ap.add_argument("--jsonl-out", required=True)
    ap.add_argument("--tasks-out", required=True)
    ap.add_argument(
        "--filter-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows with correct == True (default: on). --no-filter-correct keeps "
        "every val row regardless of correctness -- more rows per instruction group, but "
        "SFT/oracle training would then be imitating some wrong reasoning traces too.",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="When --filter-correct is on, drop an instruction group entirely (write nothing) "
        "if fewer than this many rows survive the filter, rather than shipping a near-empty "
        "task dir. Ignored when --no-filter-correct is passed. Set to 0 to disable.",
    )
    args = ap.parse_args()

    src_root = Path(args.src_root)
    jsonl_out_dir = Path(args.jsonl_out)
    tasks_out_dir = Path(args.tasks_out)

    all_summaries: list[dict] = []
    for src_dir in sorted(src_root.iterdir()):
        m = SRC_DIR_RE.match(src_dir.name)
        if not m or not (src_dir / "forward_outputs.jsonl").exists():
            continue
        task_name = m.group("task")
        all_summaries.extend(
            build_one(
                src_dir,
                task_name,
                jsonl_out_dir,
                tasks_out_dir,
                filter_correct=args.filter_correct,
                min_samples=args.min_samples,
            )
        )

    written = [s for s in all_summaries if not s["dropped_min_samples"]]
    dropped = [s for s in all_summaries if s["dropped_min_samples"]]

    print(f"{'task':<50} {'n_val_total':>11} {'n_correct':>9} {'n_rows':>7} {'think_dropped':>14} {'incorrect_dropped':>18}")
    for s in written:
        print(
            f"{s['task']:<50} {s['n_val_total']:>11} {s['n_correct_total']:>9} {s['n_rows']:>7} "
            f"{s['n_think_dropped']:>14} {s['n_incorrect_dropped']:>18}"
        )
    print(f"\nwrote {len(written)} task dirs under {tasks_out_dir} (jsonl data under {jsonl_out_dir})")

    if dropped:
        print(f"\ndropped {len(dropped)} instruction group(s) below --min-samples={args.min_samples} (filter_correct=on):")
        for s in dropped:
            print(f"  {s['task']}: {s['n_rows']} rows after correctness filter")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
