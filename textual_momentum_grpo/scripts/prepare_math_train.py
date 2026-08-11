"""Prepare the MATH training split (README section 5: "MATH (training split...)").

Per a confirmed data-scope decision, this project trains on the actual MATH training split
(`hendrycks/competition_math` on Hugging Face -- the canonical Hendrycks et al. 2021 dataset,
7,500 train / 5,000 test problems), NOT `self_correct_grpo`'s vendored `dapo-math-17k.jsonl`,
which is a different curated training pool used by that (separate) project.

`hendrycks/competition_math` is a gated dataset: you must accept its terms on the Hub and have
`huggingface-cli login` (or `HF_TOKEN` set) before this script can download it.

Output schema matches the vendored eval files this project reuses (see prepare_eval_data.py):
    {"prompt": [{"role": "user", "content": <problem text>}], "label": <boxed answer>}

Usage:
    python prepare_math_train.py --out data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_boxed_answer(solution: str) -> str | None:
    """Same brace-matching logic as tmgrpo/reward.py's extract_boxed_answer (duplicated here to
    keep this script runnable standalone, without importing the tmgrpo package)."""
    match = None
    for m in _BOXED_RE.finditer(solution):
        match = m  # keep the LAST boxed answer, matching MATH-dataset convention
    if match is None:
        return None
    i = match.end()
    depth = 1
    chars = []
    while i < len(solution) and depth > 0:
        c = solution[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(chars)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--hf-dataset",
        default="hendrycks/competition_math",
        help="Hugging Face dataset id for the MATH training split.",
    )
    args = parser.parse_args()

    try:
        import datasets
    except ImportError:
        print(
            "error: the `datasets` package is required (run this from textual_momentum_grpo's "
            "own venv: uv run --no-sync python scripts/prepare_math_train.py ...)",
            file=sys.stderr,
        )
        return 1

    try:
        ds = datasets.load_dataset(args.hf_dataset, split="train")
    except Exception as exc:  # dataset is gated -- surface the auth requirement clearly
        print(
            f"error: failed to load {args.hf_dataset!r} ({exc}).\n"
            "This dataset is gated on the Hugging Face Hub: accept its terms at "
            f"https://huggingface.co/datasets/{args.hf_dataset} and run `huggingface-cli login` "
            "(or set HF_TOKEN) before retrying.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_no_answer = 0
    with args.out.open("w") as f:
        for row in ds:
            problem = row["problem"]
            solution = row["solution"]
            answer = _extract_boxed_answer(solution)
            if answer is None:
                n_skipped_no_answer += 1
                continue
            record = {
                "prompt": [{"role": "user", "content": problem}],
                "label": answer,
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"wrote {n_written} train rows -> {args.out}")
    if n_skipped_no_answer:
        print(f"skipped {n_skipped_no_answer} rows with no \\boxed{{}} answer in solution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
