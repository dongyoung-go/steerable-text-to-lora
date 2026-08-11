"""Prepare the three eval sets (README section 5: "eval on MATH500 + a held-out slice of
OlympiadBench/AIME24").

Per a confirmed data-scope decision, these are reused directly from
`self_correct_grpo/vendor/ICRL/data/math/` rather than re-sourced: that directory already has
`math-500.jsonl` (500 rows), `aime-2024.jsonl` (30 rows, "AIME24 full"), and `olympia.jsonl`
(675 rows, OlympiadBench) in this repo, vendored (Apache-2.0) from `brick-pid/ICRL`/`slime` for
the sibling self_correct_grpo project -- see that project's `vendor/README.md` for provenance.
This script:
  - copies math-500.jsonl and aime-2024.jsonl verbatim (no transformation needed -- the
    {"prompt","label"} schema already matches this project's own train.jsonl schema).
  - writes a SEEDED, fixed-size 200-row sample of olympia.jsonl as the "held-out OlympiadBench
    slice" (confirmed data-scope decision: a small slice, not the full 675 rows).

Usage:
    python prepare_eval_data.py --vendor-dir ../self_correct_grpo/vendor/ICRL/data/math \
        --out-dir data/eval
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

OLYMPIAD_SLICE_SIZE = 200
OLYMPIAD_SLICE_SEED = 0


def _sample_jsonl(src: Path, dst: Path, n: int, seed: int) -> int:
    with src.open() as f:
        lines = [line for line in f if line.strip()]
    rng = random.Random(seed)
    sampled = lines if len(lines) <= n else rng.sample(lines, n)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        f.writelines(sampled)
    return len(sampled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "self_correct_grpo" / "vendor" / "ICRL" / "data" / "math",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    math500_src = args.vendor_dir / "math-500.jsonl"
    aime_src = args.vendor_dir / "aime-2024.jsonl"
    olympiad_src = args.vendor_dir / "olympia.jsonl"

    shutil.copyfile(math500_src, args.out_dir / "math500.jsonl")
    print(f"copied {math500_src} -> {args.out_dir / 'math500.jsonl'}")

    shutil.copyfile(aime_src, args.out_dir / "aime24.jsonl")
    print(f"copied {aime_src} -> {args.out_dir / 'aime24.jsonl'}")

    n = _sample_jsonl(
        olympiad_src,
        args.out_dir / "olympiad_slice.jsonl",
        n=OLYMPIAD_SLICE_SIZE,
        seed=OLYMPIAD_SLICE_SEED,
    )
    print(
        f"sampled {n} rows (seed={OLYMPIAD_SLICE_SEED}) from {olympiad_src} "
        f"-> {args.out_dir / 'olympiad_slice.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
