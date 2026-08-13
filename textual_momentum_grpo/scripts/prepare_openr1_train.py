"""Prepare open-r1/OpenR1-Math-220k as a training pool -- the default training set for all arms.

Per a confirmed data-scope decision (2026-08-12): Critique-GRPO (this project's baseline,
arXiv 2506.03106) does NOT train on Hendrycks MATH -- it samples 4k-32k prompts from a
reorganized 45k subset of `open-r1/OpenR1-Math-220k` (NuminaMath 1.5 problems, DeepSeek-R1
solutions, decontaminated against AIME/MATH-style benchmarks). Training arm1_floor on MATH
instead left Qwen3-8B (already strong on MATH) near-saturated from step 1 (~0.81-0.94
training-batch accuracy, flat over 100+ steps) -- too little reward variance within a rollout
group for GRPO's group-relative advantage to have much to learn from. OpenR1-Math-220k's harder,
broader (olympiad/AoPS-sourced) problem pool is meant to leave more headroom. MATH remains an
explicit opt-in (scripts/prepare_math_train.py -> data/train.parquet) for anyone who wants it.

Uses the `default` HF config (93.7k rows, one row per unique problem) rather than `extended`
(adds easier cn_k12-sourced problems that the Open-R1 team's own ablation found hurt downstream
performance) or `all` (450k rows, multiple generations per problem -- unneeded here since only
one problem+answer pair per row is kept).

Output schema matches this project's other train/eval files (see prepare_math_train.py,
prepare_eval_data.py):
    {"prompt": [{"role": "user", "content": <problem text>}], "label": <final answer str>}

Usage:
    python prepare_openr1_train.py --out data/train_openr1.jsonl
    python prepare_openr1_train.py --out data/train_openr1.jsonl --limit 20000 --seed 0

Run scripts/check_overlap.py against the eval sets afterwards, same as for the MATH pool --
OpenR1-Math-220k's own decontamination is against its own benchmark list, not necessarily
identical to this project's data/eval/*.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def row_to_record(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one open-r1/OpenR1-Math-220k row into this project's {"prompt","label"} schema.

    Returns None if the row is unusable (blank problem or answer).
    """
    problem = (row.get("problem") or "").strip()
    answer = (row.get("answer") or "").strip()
    if not problem or not answer:
        return None
    return {"prompt": [{"role": "user", "content": problem}], "label": answer}


def sample_rows(rows: list[Any], limit: int | None, seed: int) -> list[Any]:
    """Seeded subsample to `limit` rows, or all of `rows` if `limit` is None or already smaller."""
    if limit is None or len(rows) <= limit:
        return rows
    return random.Random(seed).sample(rows, limit)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--hf-dataset", default="open-r1/OpenR1-Math-220k", help="Hugging Face dataset id."
    )
    parser.add_argument(
        "--config",
        default="default",
        choices=["default", "extended", "all"],
        help="HF dataset config. 'default' (93.7k rows) is recommended -- see module docstring.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional seeded subsample size. Default: keep every usable row.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for --limit subsampling.")
    args = parser.parse_args()

    try:
        import datasets
    except ImportError:
        print(
            "error: the `datasets` package is required (run this from textual_momentum_grpo's "
            "own venv: .venv/bin/python scripts/prepare_openr1_train.py ...)",
            file=sys.stderr,
        )
        return 1

    try:
        ds = datasets.load_dataset(args.hf_dataset, args.config, split=args.split)
    except Exception as exc:
        print(
            f"error: failed to load {args.hf_dataset!r} config {args.config!r} ({exc}).\n"
            "If this is a gating/auth error, accept the dataset's terms on the Hub and run "
            "`huggingface-cli login` (or set HF_TOKEN) before retrying.",
            file=sys.stderr,
        )
        return 1

    records = []
    n_skipped = 0
    for row in ds:
        record = row_to_record(row)
        if record is None:
            n_skipped += 1
            continue
        records.append(record)

    records = sample_rows(records, args.limit, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} train rows -> {args.out}")
    if n_skipped:
        print(f"skipped {n_skipped} rows with a blank problem or answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
