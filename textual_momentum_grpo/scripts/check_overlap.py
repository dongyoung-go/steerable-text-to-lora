"""Confirm the MATH training split doesn't leak problems into any of the three eval sets.

Mirrors self_correct_grpo/scripts/check_math_overlap.py's exact-string-match approach, extended
to check train.jsonl against all three eval files at once.

Usage:
    python check_overlap.py --train data/train.jsonl \
        --eval data/eval/math500.jsonl data/eval/aime24.jsonl data/eval/olympiad_slice.jsonl

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
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    train_texts = _load_problem_texts(args.train)
    train_norm = {_normalize(t) for t in train_texts}
    print(f"train pool: {len(train_texts)} problems ({args.train})")

    any_overlap = False
    for eval_path in args.eval:
        eval_texts = _load_problem_texts(eval_path)
        eval_norm = [_normalize(t) for t in eval_texts]
        overlap = [t for t in eval_norm if t in train_norm]
        print(f"eval set:   {len(eval_texts)} problems ({eval_path}) -- overlap: {len(overlap)}")
        if overlap:
            any_overlap = True

    if any_overlap:
        print(
            "\nFAIL: exact-string overlap found between train pool and at least one eval set.",
            file=sys.stderr,
        )
        return 1

    print("\nOK: no exact-string overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
