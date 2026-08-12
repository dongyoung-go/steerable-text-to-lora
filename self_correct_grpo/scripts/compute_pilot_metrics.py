"""Compute Delta[i->c], Delta[c->i], and no-op rate from ICRL rollout trajectory dumps.

self_correct_grpo_README.md §6.3 asks for these three numbers as the entire output of the Phase 1
pilot. Neither vendored `icrl.logging_utils.log_rollout_data` nor `log_eval_rollout_data` computes
them directly -- they log per-role/per-round success *rates* for monitoring, not the paired
per-episode (r(tau1), r(tau2)) deltas this pilot needs. Rather than fork vendored logging code
(which would break the "vendor/ICRL stays byte-for-byte upstream" property -- see
`../vendor/README.md`), this script reconstructs per-episode round sequences directly from the
plaintext trajectory dumps that `_save_rollout_trajectories` already writes, unmodified, for every
rollout batch (train or eval) -- see docs/pilot_eval_design.md for why this pilot's decisive
comparison is two *eval-only* passes (`{exp_dir}/rollouts_eval/eval_<rollout_id>.txt`) against one
gated-trained checkpoint, gated vs. ungated inference, rather than two separately-trained arms'
training-rollout streams (`rollouts_train/train_<rollout_id>.txt`, still supported via --split
train for that earlier, superseded comparison).

Dump format (one entry per executor/critic sample; entries within an episode separated by a
`--------` line, episodes separated by `========`):

    episode_id: 3
    round_id: 1
    role: executor
    reward: 0.0
    task_desc: ...
    <trajectory text>

Definitions (matching self_correct_grpo_README.md §1.1 / §6.3), per episode:
    r1 = round-1 executor reward
    r2 = final-round executor reward (r2 := r1 if no second round happened -- covers both the
         gated arm's oracle-gate skip on already-correct tau1, and a critic that ran but achieved
         nothing)
    Delta[i->c]  = fraction of episodes with r1 == 0 and r2 == 1
    Delta[c->i]  = fraction of episodes with r1 == 1 and r2 == 0
    no-op rate   = fraction of episodes with r2 == r1

Usage:
    python compute_pilot_metrics.py --gated-dir <exp_dir_gated> --ungated-dir <exp_dir_ungated> \
        [--split eval|train]

Each `--*-dir` should be an ICRL `exp_dir` (containing `rollouts_eval/` or `rollouts_train/`), or
that directory itself. `--split` defaults to `eval` (the current pilot design); pass `--split train`
to reproduce the earlier, superseded two-separately-trained-arms comparison.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

_SEPARATOR_RE = re.compile(r"^(?:={8}|-{8})$")
_FIELD_RE = re.compile(r"^(episode_id|round_id|role|reward): (.*)$")


@dataclass(frozen=True)
class RoundEntry:
    episode_id: int
    round_id: int
    role: str
    reward: float | None


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: int
    r1: float
    r2: float
    critic_ran: bool


def parse_trajectory_dump(text: str) -> list[RoundEntry]:
    entries: list[RoundEntry] = []
    fields: dict[str, str] = {}

    def flush() -> None:
        if not fields:
            return
        entries.append(
            RoundEntry(
                episode_id=int(fields["episode_id"]),
                round_id=int(fields["round_id"]),
                role=fields["role"],
                reward=_parse_reward(fields.get("reward")),
            )
        )

    for line in text.splitlines():
        if _SEPARATOR_RE.match(line):
            flush()
            fields = {}
            continue
        if line.startswith("episode_id: ") and "episode_id" in fields:
            # a new entry started without an explicit separator line reaching us (shouldn't
            # normally happen given _save_rollout_trajectories always emits one, but be defensive)
            flush()
            fields = {}
        match = _FIELD_RE.match(line)
        if match and match.group(1) not in fields:
            fields[match.group(1)] = match.group(2)
    flush()
    return entries


def _parse_reward(raw: str | None) -> float | None:
    if raw is None or raw == "None":
        return None
    return float(raw)


def group_into_episodes(entries: list[RoundEntry]) -> list[EpisodeOutcome]:
    by_episode: dict[int, list[RoundEntry]] = {}
    for entry in entries:
        by_episode.setdefault(entry.episode_id, []).append(entry)

    outcomes = []
    for episode_id, episode_entries in by_episode.items():
        executor_entries = sorted(
            (e for e in episode_entries if e.role == "executor"),
            key=lambda e: e.round_id,
        )
        if not executor_entries or executor_entries[0].reward is None:
            continue
        r1 = executor_entries[0].reward
        r2 = executor_entries[-1].reward if executor_entries[-1].reward is not None else r1
        critic_ran = any(e.role == "critic" for e in episode_entries)
        outcomes.append(EpisodeOutcome(episode_id=episode_id, r1=r1, r2=r2, critic_ran=critic_ran))
    return outcomes


@dataclass(frozen=True)
class PilotMetrics:
    num_episodes: int
    fix_rate: float  # Delta[i->c]
    regression_rate: float  # Delta[c->i]
    no_op_rate: float
    critic_invocation_rate: float


def compute_metrics(outcomes: list[EpisodeOutcome]) -> PilotMetrics:
    n = len(outcomes)
    if n == 0:
        return PilotMetrics(0, 0.0, 0.0, 0.0, 0.0)

    fix = sum(1 for o in outcomes if o.r1 == 0 and o.r2 == 1)
    regress = sum(1 for o in outcomes if o.r1 == 1 and o.r2 == 0)
    no_op = sum(1 for o in outcomes if o.r1 == o.r2)
    critic_invoked = sum(1 for o in outcomes if o.critic_ran)

    return PilotMetrics(
        num_episodes=n,
        fix_rate=fix / n,
        regression_rate=regress / n,
        no_op_rate=no_op / n,
        critic_invocation_rate=critic_invoked / n,
    )


def _resolve_rollouts_dir(path: Path, split: str) -> Path:
    dirname = f"rollouts_{split}"
    if path.name == dirname:
        return path
    candidate = path / dirname
    return candidate if candidate.is_dir() else path


def load_metrics_for_dir(exp_dir: Path, split: str = "eval") -> PilotMetrics:
    rollouts_dir = _resolve_rollouts_dir(exp_dir, split)
    if not rollouts_dir.is_dir():
        raise FileNotFoundError(f"no rollouts_{split}/ directory found under {exp_dir}")

    prefix = "train" if split == "train" else "eval"
    outcomes: list[EpisodeOutcome] = []
    for dump_file in sorted(rollouts_dir.glob(f"{prefix}_*.txt")):
        entries = parse_trajectory_dump(dump_file.read_text())
        outcomes.extend(group_into_episodes(entries))
    return compute_metrics(outcomes)


def _format_row(label: str, m: PilotMetrics) -> str:
    return (
        f"{label:<12} episodes={m.num_episodes:<7} "
        f"Delta[i->c]={m.fix_rate:.3f}  Delta[c->i]={m.regression_rate:.3f}  "
        f"no-op={m.no_op_rate:.3f}  critic-invoked={m.critic_invocation_rate:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gated-dir", type=Path, required=True)
    parser.add_argument("--ungated-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["eval", "train"],
        default="eval",
        help="Which rollout dumps to read. 'eval' (default) matches this pilot's current design: "
        "gated-inference vs. ungated-inference eval-only passes against one gated-trained "
        "checkpoint (see docs/pilot_eval_design.md). 'train' reproduces the earlier, superseded "
        "two-separately-trained-arms comparison.",
    )
    args = parser.parse_args()

    gated = load_metrics_for_dir(args.gated_dir, args.split)
    ungated = load_metrics_for_dir(args.ungated_dir, args.split)

    print(_format_row("gated", gated))
    print(_format_row("ungated", ungated))
    print()
    print(f"Delta[c->i] gap (ungated - gated): {ungated.regression_rate - gated.regression_rate:+.3f}")
    print(f"no-op rate gap (ungated - gated):  {ungated.no_op_rate - gated.no_op_rate:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
