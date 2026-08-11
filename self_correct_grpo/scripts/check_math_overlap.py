"""Confirm the DAPO-Math-17k training pool doesn't leak MATH500 eval problems.

Usage:
    python check_math_overlap.py \
        --train vendor/ICRL/data/math/dapo-math-17k.jsonl \
        --eval vendor/ICRL/data/math/math-500.jsonl

Exits non-zero if any exact-string problem overlap is found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _load_problem_texts(path: Path) -> list[str]:
    texts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record["prompt"]
            user_turns = [m["content"] for m in prompt if m.get("role") == "user"]
            texts.append(user_turns[-1] if user_turns else "")
    return texts


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    args = parser.parse_args()

    train_texts = _load_problem_texts(args.train)
    eval_texts = _load_problem_texts(args.eval)

    train_norm = {_normalize(t) for t in train_texts}
    eval_norm = [_normalize(t) for t in eval_texts]

    overlap = [t for t in eval_norm if t in train_norm]

    print(f"train pool: {len(train_texts)} problems ({args.train})")
    print(f"eval set:   {len(eval_texts)} problems ({args.eval})")
    print(f"overlap:    {len(overlap)} problems")

    if overlap:
        print("\nFAIL: exact-string overlap found between train pool and eval set.", file=sys.stderr)
        return 1

    print("\nOK: no exact-string overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
