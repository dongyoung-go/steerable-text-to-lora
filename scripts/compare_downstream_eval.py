"""Diff two eval_downstream_accuracy_full.py (or eval_downstream_accuracy.py) output JSONs
condition-by-condition and task-by-task. See docs/06_description_augmentation_v5.md's final
"what's not built yet" item: a direct comparison against v3's already-diagnosed collapse.

Each output JSON's own ``"comparisons"`` key (``compute_comparisons`` in
``src/steerable_t2l/eval_accuracy.py``) is a *within-run* macro average across conditions
(t2l_train_desc vs. base/prompted/oracle) -- nothing upstream diffs *across* two separate runs.
This script does that: prints each run's per-condition macro accuracy (``result["overall"]``)
side by side with the delta, then the same for the three derived comparison ratios
(``result["comparisons"]["macro"]``), then a per-task table restricted to tasks present in both
files.

Per-task join key: each task name with --strip-a/--strip-b substrings removed, so
e.g. ``textgrad_repro_v3_aqua_d9`` (file A) and ``textgrad_repro_v5_aqua_d9`` (file B) join as
``textgrad_repro_aqua_d9``. Tasks whose join key doesn't appear in both files are skipped, not
errored -- v3/v5 task-dir names are usually 1:1 but this stays robust if they ever aren't.

    python scripts/compare_downstream_eval.py \
        outputs/eval/downstream_accuracy_full_v3.json outputs/eval/downstream_accuracy_full_v5.json \
        --labels v3 v5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def _delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return b - a
    return "n/a"


def join_key(task_name: str, strip_a: str, strip_b: str) -> str:
    return task_name.replace(strip_a, "").replace(strip_b, "")


def print_condition_table(overall_a: dict, overall_b: dict, labels: tuple[str, str]) -> None:
    conditions = sorted(set(overall_a) | set(overall_b))
    label_a, label_b = labels
    print(f"{'condition':<25} {label_a:>10} {label_b:>10} {'delta':>10}")
    for condition in conditions:
        a = overall_a.get(condition, "n/a")
        b = overall_b.get(condition, "n/a")
        print(f"{condition:<25} {_fmt(a):>10} {_fmt(b):>10} {_fmt(_delta(a, b)):>10}")


def print_comparisons_table(macro_a: dict, macro_b: dict, labels: tuple[str, str]) -> None:
    keys = sorted(set(macro_a) | set(macro_b))
    label_a, label_b = labels
    print(f"{'comparison':<32} {label_a:>10} {label_b:>10} {'delta':>10}")
    for key in keys:
        a = macro_a.get(key, "n/a")
        b = macro_b.get(key, "n/a")
        print(f"{key:<32} {_fmt(a):>10} {_fmt(b):>10} {_fmt(_delta(a, b)):>10}")


def print_per_task_table(
    per_task_a: dict, per_task_b: dict, strip_a: str, strip_b: str, condition: str, labels: tuple[str, str]
) -> list[dict]:
    by_key_a = {join_key(name, strip_a, strip_b): (name, conds) for name, conds in per_task_a.items()}
    by_key_b = {join_key(name, strip_a, strip_b): (name, conds) for name, conds in per_task_b.items()}
    common_keys = sorted(set(by_key_a) & set(by_key_b))

    label_a, label_b = labels
    rows = []
    print(f"\n{'task':<45} {label_a:>10} {label_b:>10} {'delta':>10}")
    for key in common_keys:
        _, conds_a = by_key_a[key]
        _, conds_b = by_key_b[key]
        result_a = conds_a.get(condition)
        result_b = conds_b.get(condition)
        a = result_a["accuracy"] if isinstance(result_a, dict) else "n/a"
        b = result_b["accuracy"] if isinstance(result_b, dict) else "n/a"
        print(f"{key:<45} {_fmt(a):>10} {_fmt(b):>10} {_fmt(_delta(a, b)):>10}")
        rows.append({"task": key, label_a: a, label_b: b})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--labels", nargs=2, default=("a", "b"), metavar=("LABEL_A", "LABEL_B"))
    ap.add_argument("--strip-a", default="v3_", help="substring to strip from file A's task names for joining")
    ap.add_argument("--strip-b", default="v5_", help="substring to strip from file B's task names for joining")
    ap.add_argument("--per-task-condition", default="t2l_train_desc", help="condition to show in the per-task table")
    args = ap.parse_args()

    result_a = json.loads(Path(args.file_a).read_text())
    result_b = json.loads(Path(args.file_b).read_text())
    labels = tuple(args.labels)

    print("=== per-condition macro accuracy")
    print_condition_table(result_a["overall"], result_b["overall"], labels)

    print("\n=== macro-averaged comparisons")
    print_comparisons_table(result_a["comparisons"]["macro"], result_b["comparisons"]["macro"], labels)

    print_per_task_table(
        result_a["per_task"], result_b["per_task"], args.strip_a, args.strip_b,
        args.per_task_condition, labels,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
