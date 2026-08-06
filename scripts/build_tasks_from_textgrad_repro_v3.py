"""Build task dirs (metadata.yaml + question/response jsonl) from the textgrad-repro dataset at
data/textgrad_repro/qwen-qwen3-14b_*_textgrad-repro/. v3 of build_tasks_from_textgrad_repro_v2.py:
one task dir (one oracle LoRA) PER DISTINCT INSTRUCTION seen in a task's optimization
trajectory, not one task dir per task with a single "winning" instruction plus a pool of
alternate-phrasing descriptions.

Read-only w.r.t. the source dataset -- writes only to --jsonl-out and --tasks-out (new
locations, never data/textgrad_repro/ itself).

Why per-instruction, not per-task (see also build_tasks_from_gepa_repro_v3.py, same rationale):
  - v2 picked a single best-scoring iteration and used only its forward_outputs.jsonl rows,
    with every other instruction phrasing tried during the run listed in metadata.yaml's
    "descriptions" purely as alternates for D-axis (description-holdout) eval -- no responses of
    their own.
  - PerTaskDescDataset.__getitem__ (src/steerable_t2l/data/datasets.py) samples a description
    uniformly at random from a task's WHOLE descriptions pool for every training row,
    independent of which description actually produced that row's response. That's intentional
    for the (not yet built) description-AUGMENTATION feature, where descriptions are meant to be
    semantically-preserving paraphrases of one instruction. It is NOT intentional for the
    textgrad-repro data: consecutive iterations here often try genuinely different instructions
    (different structure, different emphasis -- not just paraphrases), so pooling their
    responses under one task and letting training condition a row on a random OTHER
    instruction would teach false instruction->response associations.
  - The fix is architectural, not a filter: each genuinely distinct instruction gets its own
    task dir (metadata.yaml with exactly one description) and hence its own oracle LoRA --
    "LoRA per description", not "LoRA per task". Rows generated across iterations that reused
    the *same, byte-identical* instruction (textgrad's "reverted" mechanic: a proposed edit is
    tried, scores worse, and the optimizer reverts to the prior prompt for the next round) are
    still pooled together, since that's the literal same description being reused, not a
    different one.
  - Side effect: every emitted task dir now has exactly one description, so the D-axis
    (docs/03 splits.d_holdout) is universally N/A for v3, same degenerate case the codebase
    already warns about for legacy single-description tasks. Accepted tradeoff until
    description-augmentation is built as its own feature.

For each source dir:
  - groups forward_outputs.jsonl's split == "val" rows by their own literal "prompt" field
    text, in first-appearance order. NOTE: this must use forward_outputs.jsonl's own "prompt"
    field, NOT iterations.jsonl's -- the two files log prompt text independently and can diverge
    mid-sentence at the same iteration index (verified empirically on e.g.
    bbh_logical_deduction_seven_objects, iteration 6); grouping by the wrong file's text would
    silently misassign rows to the wrong instruction group.
  - for each distinct-instruction group, in first-appearance order (group index 0, 1, 2, ...):
      - keeps rows with (unless --no-filter-correct) correct == True. The correctness filter
        exists because these rows double as SFT training targets (response_field: "response",
        consumed by oracle Stage A and hypernetwork SFT) -- keeping incorrect rows would train
        the model to imitate wrong reasoning. --no-filter-correct keeps every val row regardless
        of correctness; --min-samples (only enforced when the correctness filter is on) drops
        that instruction's group entirely (writes nothing for it) if too few rows survive,
        rather than shipping a near-empty task dir.
      - drops any row whose model_response still contains "<think>" (defense in depth).
      - dedupes by (question, response) pair -- NOT question alone. Pooling across reverted
        iterations that reused the identical instruction only grows the dataset if a repeated
        question is allowed to keep more than one distinct response text; deduping by question
        alone would collapse pooled rows right back down to one-per-question. Exact duplicate
        (question, response) pairs are still dropped.
  - writes one task dir per surviving group: <tasks-out>/textgrad_repro_v3_<task>_d<K>/, K being
    that group's first-appearance index, with metadata.yaml's "descriptions": [that one
    instruction text] and "best_description_index": 0.
  - each row also carries forward_outputs.jsonl's own "gold_answer" field verbatim, same as v2,
    so scripts/eval_downstream_accuracy.py's classify_answer_parser keeps working unchanged.

    python scripts/build_tasks_from_textgrad_repro_v3.py \
        --src-root data/textgrad_repro \
        --jsonl-out /home/dg793/text-to-lora/data/textgrad_repro_v3_t2l \
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


def group_val_rows_by_prompt(forward_outputs_path: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """Groups split == "val" rows by their own literal "prompt" text, in first-appearance
    order. Returns (groups, order) where order lists each distinct prompt text in the sequence
    it was first seen (== group index K used in the emitted task dir name)."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for line in open(forward_outputs_path):
        row = json.loads(line)
        if row["split"] != "val":
            continue
        prompt = row["prompt"]
        if prompt not in groups:
            groups[prompt] = []
            order.append(prompt)
        groups[prompt].append(row)
    return groups, order


def build_one(
    src_dir: Path,
    task_name: str,
    jsonl_out_dir: Path,
    tasks_out_dir: Path,
    filter_correct: bool,
    min_samples: int,
) -> list[dict]:
    groups, order = group_val_rows_by_prompt(src_dir / "forward_outputs.jsonl")

    summaries: list[dict] = []
    for group_idx, prompt in enumerate(order):
        group_rows = groups[prompt]
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

        task_dir = tasks_out_dir / f"textgrad_repro_v3_{sub_task_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "descriptions": [prompt],
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
        print(f"\ndropped {len(dropped)} instruction group(s) below --min-samples={args.min_samples} (filter_correct=on):")
        for s in dropped:
            print(f"  {s['task']}: {s['n_rows']} rows after correctness filter")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
