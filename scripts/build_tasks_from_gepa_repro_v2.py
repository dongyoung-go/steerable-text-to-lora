"""Build task dirs (metadata.yaml + question/response jsonl) from the gepa-repro dataset at
data/gepa_repro/qwen-qwen3-14b_*_gepa-repro/. Parallel to build_tasks_from_textgrad_repro_v2.py,
but adapted to gepa's schema: forward_outputs.jsonl rows have no "split" field and iterations.jsonl
rows have no "prompt" field (gepa's analogs are "candidate" strings and a separate val_set.jsonl).

Read-only w.r.t. the source dataset -- writes only to --jsonl-out and --tasks-out (new
locations, never data/gepa_repro/ itself).

For each source dir:
  - best_prompt.json's "candidate" field is gepa's already-selected winning prompt text (gepa
    itself tracks the Pareto-best candidate across the run, so there's no need to recompute
    "best iteration" from accuracy the way the textgrad builder does).
  - forward_outputs.jsonl rows carry the literal "candidate" text that was live when that row was
    generated. Filtering row["candidate"] == best_prompt["candidate"] isolates exactly the rows
    generated under the winning prompt, regardless of how many search iterations touched it.
  - gepa has no per-row "split" field, so val membership is determined by cross-referencing
    row["question"] against val_set.jsonl's "question_prompt" values (the actual held-out val
    set) -- this is the gepa analog of textgrad's row["split"] == "val" filter.
  - drops any row whose model_response still contains "<think>", same as the textgrad builder.
  - descriptions = every distinct "candidate" string seen across iterations.jsonl for that task,
    in first-appearance order (the gepa analog of textgrad's distinct "prompt" strings). Some
    tasks converge at the seed candidate with zero accepted iterations logged (empty
    iterations.jsonl); in that case descriptions falls back to just [best_prompt["candidate"]].
  - each row also carries forward_outputs.jsonl's own "gold_answer" field verbatim, same as the
    textgrad builder, so scripts/eval_downstream_accuracy.py's classify_answer_parser keeps
    working unchanged.

    python scripts/build_tasks_from_gepa_repro_v2.py \
        --src-root data/gepa_repro \
        --jsonl-out /home/dg793/text-to-lora/data/gepa_repro_v2_t2l \
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


def collect_descriptions(iterations_path: Path, fallback_candidate: str) -> list[str]:
    seen: list[str] = []
    for line in open(iterations_path):
        candidate = json.loads(line)["candidate"]
        if candidate not in seen:
            seen.append(candidate)
    if not seen:
        # converged_early at the seed with zero accepted iterations logged -- the seed
        # candidate (== best_prompt's candidate) is the only prompt this task ever used.
        seen = [fallback_candidate]
    return seen


def build_one(
    src_dir: Path,
    task_name: str,
    jsonl_out_dir: Path,
    tasks_out_dir: Path,
    filter_correct: bool,
    min_samples: int,
) -> dict:
    best_prompt = json.load(open(src_dir / "best_prompt.json"))
    best_candidate = best_prompt["candidate"]
    val_questions = load_val_questions(src_dir / "val_set.jsonl")
    descriptions = collect_descriptions(src_dir / "iterations.jsonl", best_candidate)
    best_desc_idx = descriptions.index(best_candidate)

    rows: list[dict] = []
    n_think_dropped = 0
    n_incorrect_dropped = 0
    seen_questions: set[str] = set()
    for line in open(src_dir / "forward_outputs.jsonl"):
        row = json.loads(line)
        if row["candidate"] != best_candidate or row["question"] not in val_questions:
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
        "val_accuracy": best_prompt["val_accuracy"],
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

    task_dir = tasks_out_dir / f"gepa_repro_v2_{task_name}"
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
    ap.add_argument("--src-root", default="data/gepa_repro")
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

    print(f"{'task':<45} {'val_acc':>8} {'n_desc':>7} {'n_rows':>7} {'think_dropped':>14} {'incorrect_dropped':>18}")
    for s in written:
        print(
            f"{s['task']:<45} {s['val_accuracy']:>8.3f} {s['n_descriptions']:>7} {s['n_rows']:>7} "
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
