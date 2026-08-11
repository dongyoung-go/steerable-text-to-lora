"""Adapt vendored ICRL math jsonl files to the schema slime's `Dataset` loader expects.

`vendor/ICRL/data/math/{dapo-math-17k,math-500}.jsonl` rows are `{"prompt": [...], "label": ...}`
with no `metadata` field. slime's loader (`vendor/ICRL/slime/utils/data.py`) only auto-derives
`metadata.data_source` for the `criticgrpo` parquet schema (`reward_model.ground_truth` present);
plain math jsonl rows pass through untouched, and `icrl/generate.py` unconditionally reads
`sample.metadata["data_source"]` to decide which environment client to use — so without this
field every rollout would KeyError.

This script adds `"metadata": {"data_source": "math"}` to every row and writes the result to
`self_correct_grpo/data/math_pilot/`, keeping the vendored tree untouched. Run once before the
pilot; both arms (gated and ungated) read from the same prepared files.

Usage:
    python prepare_math_data.py \
        --train vendor/ICRL/data/math/dapo-math-17k.jsonl \
        --eval vendor/ICRL/data/math/math-500.jsonl \
        --out-dir data/math_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _add_data_source(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.setdefault("metadata", {})["data_source"] = "math"
            fout.write(json.dumps(record) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    train_out = args.out_dir / "train.jsonl"
    eval_out = args.out_dir / "eval.jsonl"

    n_train = _add_data_source(args.train, train_out)
    n_eval = _add_data_source(args.eval, eval_out)

    print(f"wrote {n_train} train rows -> {train_out}")
    print(f"wrote {n_eval} eval rows -> {eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
