"""Prepare the MATH training split (README section 5: "MATH (training split...)").

Per a confirmed data-scope decision, this project trains on the actual MATH training split
(the canonical Hendrycks et al. 2021 dataset, 7,500 train / 5,000 test problems across 7 subject
areas), NOT `self_correct_grpo`'s vendored `dapo-math-17k.jsonl`, which is a different curated
training pool used by that (separate) project.

The original `hendrycks/competition_math` repo on the Hugging Face Hub is disabled (403 on every
file, not merely gated -- confirmed 2026-08-11), so this pulls from `EleutherAI/hendrycks_math`
instead: a parquet re-upload of the identical dataset (same `problem`/`solution`/`level`/`type`
schema), split into 7 per-subject configs (algebra, counting_and_probability, geometry,
intermediate_algebra, number_theory, prealgebra, precalculus) each with a `train`/`test` split.
This mirror is widely used elsewhere (e.g. EleutherAI's own lm-evaluation-harness) and needs no
`trust_remote_code` / gating -- it loads as plain parquet.

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


_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--hf-dataset",
        default="EleutherAI/hendrycks_math",
        help="Hugging Face dataset id for the MATH training split (parquet mirror; the original "
        "hendrycks/competition_math repo is disabled on the Hub).",
    )
    parser.add_argument("--split", default="train", help="Split name within each subject config.")
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

    rows = []
    for subject in _SUBJECTS:
        try:
            ds = datasets.load_dataset(args.hf_dataset, subject, split=args.split)
        except Exception as exc:
            print(
                f"error: failed to load {args.hf_dataset!r} config {subject!r} ({exc}).\n"
                "If this is a gating/auth error, accept the dataset's terms on the Hub and run "
                "`huggingface-cli login` (or set HF_TOKEN) before retrying.",
                file=sys.stderr,
            )
            return 1
        rows.extend(ds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_no_answer = 0
    with args.out.open("w") as f:
        for row in rows:
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
