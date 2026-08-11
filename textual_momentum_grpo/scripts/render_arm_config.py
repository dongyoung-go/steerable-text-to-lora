"""Deep-merge configs/base.yaml with one arm's configs/overrides/*.yaml into a single, fully
resolved config file verl can be pointed at directly.

Kept as a plain-Python deep merge (not Hydra's `defaults:` composition) so this is verifiable
without installing verl or Hydra -- see configs/base.yaml's header comment on why verl's own
config-composition behavior is treated as unverified in this build pass.

Usage:
    python render_arm_config.py --arm arm1_floor --out configs/resolved/arm1_floor.yaml
    python render_arm_config.py --all   # renders every configs/overrides/*.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base`; override wins on scalar/list conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def render(arm_name: str) -> dict:
    base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    override_path = CONFIG_DIR / "overrides" / f"{arm_name}.yaml"
    override = yaml.safe_load(override_path.read_text())
    return deep_merge(base, override)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm", help="Override file stem under configs/overrides/, e.g. arm1_floor")
    group.add_argument("--all", action="store_true", help="Render every configs/overrides/*.yaml")
    parser.add_argument("--out", type=Path, help="Output path (only valid with --arm)")
    args = parser.parse_args()

    if args.all:
        out_dir = CONFIG_DIR / "resolved"
        out_dir.mkdir(parents=True, exist_ok=True)
        for override_path in sorted((CONFIG_DIR / "overrides").glob("*.yaml")):
            arm_name = override_path.stem
            resolved = render(arm_name)
            out_path = out_dir / f"{arm_name}.yaml"
            out_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
            print(f"wrote {out_path}")
    else:
        resolved = render(args.arm)
        out_path = args.out or (CONFIG_DIR / "resolved" / f"{args.arm}.yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
