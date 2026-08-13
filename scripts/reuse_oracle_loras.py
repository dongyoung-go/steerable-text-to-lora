"""Point one task namespace's oracle LoRAs at another namespace's already-trained ones, via
symlinks, instead of retraining. See docs/06_description_augmentation_v5.md.

Built for v5 reusing v3's oracles: v5's task dirs (``textgrad_repro_v5_*``/``gepa_repro_v5_*``)
are byte-for-byte copies of v3's ``(question, response)`` training rows with only ``descriptions``
augmented (see ``scripts/paraphrase_descs.py``), so v3's oracle LoRAs are numerically identical to
what training against v5 would produce. But every downstream consumer (``train_recon.py``,
``train_sft.py``, ``eval_downstream_accuracy*.py``) looks adapters up by ``Path(oracle_dir) /
task.name`` -- strictly keyed by task-dir name -- so v5's differently-named task dirs need their
own oracle-dir tree. This script builds that tree as symlinks, not copies, so nothing is
duplicated on disk and it stays trivially in sync with the source.

Not hardcoded to v3->v5: ``--from-substr``/``--to-substr`` do a plain string substitution on each
discovered task's name to find its source name, so any two namespaces sharing this "same data,
renamed" relationship can reuse this script.

``--splits`` is optional but should be passed the new namespace's ``splits.json`` whenever one
exists: ``train_oracle_loras.py`` never trains an oracle for a ``t_holdout`` task (see its own
``tasks = [t for t in tasks if t.name not in splits.t_holdout]``), so those tasks have no source
oracle to reuse by construction, not by omission. Without ``--splits`` this script has no way to
tell "held out on purpose" apart from "actually missing" and fails loudly on both.

    python scripts/reuse_oracle_loras.py \
        --tasks-root /home/dg793/text-to-lora/tasks \
        --train-tasks textgrad_repro_v5_* gepa_repro_v5_* --splits data/splits_v5.json \
        --source-oracle-dir outputs/oracle_loras_v3 --source-canon-dir outputs/oracle_loras_canon_v3 \
        --out-oracle-dir outputs/oracle_loras_v5 --out-canon-dir outputs/oracle_loras_canon_v5 \
        --from-substr _v5_ --to-substr _v3_
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from steerable_t2l.data.registry import discover_tasks
from steerable_t2l.data.splits import Splits


def _relink(link_path: Path, target: Path, *, force: bool, is_dir: bool) -> None:
    if link_path.exists() or link_path.is_symlink():
        if not link_path.is_symlink():
            raise FileExistsError(
                f"{link_path} already exists and is not a symlink -- refusing to overwrite a "
                "real file/directory (it may be an actually-trained oracle)"
            )
        if not force:
            return
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(target, start=link_path.parent)
    os.symlink(rel_target, link_path, target_is_directory=is_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-root", required=True, help="tasks-root the --train-tasks patterns match against")
    ap.add_argument("--train-tasks", nargs="+", required=True, help="glob pattern(s) for the NEW namespace's task dirs")
    ap.add_argument("--source-oracle-dir", required=True, help="e.g. outputs/oracle_loras_v3")
    ap.add_argument("--source-canon-dir", required=True, help="e.g. outputs/oracle_loras_canon_v3")
    ap.add_argument("--out-oracle-dir", required=True, help="e.g. outputs/oracle_loras_v5")
    ap.add_argument("--out-canon-dir", required=True, help="e.g. outputs/oracle_loras_canon_v5")
    ap.add_argument("--from-substr", required=True, help="substring in the new namespace's task names, e.g. _v5_")
    ap.add_argument("--to-substr", required=True, help="replacement substring giving the source task name, e.g. _v3_")
    ap.add_argument("--force", action="store_true", help="relink even if a symlink already exists at the destination")
    ap.add_argument(
        "--splits",
        help="splits.json from scripts/make_splits.py -- tasks in its t_holdout are skipped, "
        "matching train_oracle_loras.py's own filtering (T-holdout tasks never get an oracle "
        "trained, so they never have a source to reuse)",
    )
    args = ap.parse_args()

    tasks = discover_tasks(args.tasks_root, args.train_tasks)
    if not tasks:
        print("no tasks matched --train-tasks")
        return 1

    if args.splits:
        with open(args.splits) as f:
            splits = Splits.from_dict(json.load(f))
        t_holdout = set(splits.t_holdout)
        tasks = [t for t in tasks if t.name not in t_holdout]

    source_oracle_dir = Path(args.source_oracle_dir)
    source_canon_dir = Path(args.source_canon_dir)
    out_oracle_dir = Path(args.out_oracle_dir)
    out_canon_dir = Path(args.out_canon_dir)

    missing: list[str] = []
    linked = 0
    for task in tasks:
        if args.from_substr not in task.name:
            missing.append(f"{task.name}: does not contain --from-substr {args.from_substr!r}")
            continue
        source_name = task.name.replace(args.from_substr, args.to_substr)
        source_oracle = source_oracle_dir / source_name
        source_canon = source_canon_dir / f"{source_name}.pt"
        if not source_oracle.is_dir():
            missing.append(f"{task.name}: no source oracle dir at {source_oracle}")
            continue
        if not source_canon.is_file():
            missing.append(f"{task.name}: no source canon file at {source_canon}")
            continue

        _relink(out_oracle_dir / task.name, source_oracle, force=args.force, is_dir=True)
        _relink(out_canon_dir / f"{task.name}.pt", source_canon, force=args.force, is_dir=False)
        linked += 1

    print(f"linked {linked}/{len(tasks)} task(s) -> {out_oracle_dir}, {out_canon_dir}")

    if missing:
        print(f"\n{len(missing)} task(s) missing a source oracle/canon:")
        for m in missing:
            print(f"  {m}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
