"""Build a seeded ``splits.json`` over the three held-out axes (Q, D, T).

Run this right after the data pipeline exists and before any training, per docs/03's
suggested build order -- it is also what lets you record ``base`` reference losses before
training anything.

    python scripts/make_splits.py --tasks-root /home/dg793/text-to-lora/tasks \
        --train-tasks 'textgrad_repro_gsm8k_*' --t-frac 0.15 --q-frac 0.10 --seed 0 \
        --out data/splits.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import d_axis_available, make_splits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-root", required=True)
    ap.add_argument("--train-tasks", nargs="+", required=True)
    ap.add_argument("--t-frac", type=float, default=0.15)
    ap.add_argument("--q-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--force", action="store_true", help="regenerate even if --out already exists")
    args = ap.parse_args()

    if Path(args.out).exists() and not args.force:
        print(f"skipping -- {args.out} already exists (--force to regenerate)")
        return 0

    tasks = discover_tasks(args.tasks_root, args.train_tasks)
    if not tasks:
        print(f"no tasks found under {args.tasks_root} matching {args.train_tasks}")
        return 1

    splits = make_splits(tasks, t_frac=args.t_frac, q_frac=args.q_frac, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits.to_dict(), f, indent=2)

    n_d_na = sum(1 for t in tasks if not d_axis_available(splits, t.name) and t.name not in splits.t_holdout)
    print(f"=== splits.json written to {out_path} ({len(tasks)} tasks total)")
    print(f"  Q: {args.q_frac:.0%} of rows held out per trained task")
    print(f"  D: n/a for {n_d_na}/{len(tasks) - len(splits.t_holdout)} trained tasks (< 2 descriptions)")
    print(f"  T: {', '.join(splits.t_holdout) or '(none)'} ({len(splits.t_holdout)} tasks held out entirely)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
