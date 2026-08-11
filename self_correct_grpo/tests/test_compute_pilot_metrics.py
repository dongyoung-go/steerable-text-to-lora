from pathlib import Path

import pytest

from scripts.compute_pilot_metrics import (
    compute_metrics,
    group_into_episodes,
    load_metrics_for_dir,
    parse_trajectory_dump,
)

# Mirrors icrl/logging_utils.py::_save_rollout_trajectories's actual on-disk format: entries
# separated by "--------" within an episode and "========" between episodes, each entry a block
# of "key: value" lines followed by free-form trajectory text.
SAMPLE_DUMP = """episode_id: 0
round_id: 1
role: executor
reward: 1.0
task_desc: already correct, gated skip
trajectory for episode 0 round 1
========
episode_id: 1
round_id: 1
role: executor
reward: 0.0
task_desc: wrong then fixed
trajectory for episode 1 round 1
--------
episode_id: 1
round_id: 1
role: critic
reward: 1.0
critique text
--------
episode_id: 1
round_id: 2
role: executor
reward: 1.0
trajectory for episode 1 round 2
========
episode_id: 2
round_id: 1
role: executor
reward: 0.0
trajectory for episode 2 round 1
--------
episode_id: 2
round_id: 1
role: critic
reward: 0.0
critique text
--------
episode_id: 2
round_id: 2
role: executor
reward: 0.0
trajectory for episode 2 round 2 (still wrong)
========
episode_id: 3
round_id: 1
role: executor
reward: 1.0
trajectory for episode 3 round 1
--------
episode_id: 3
round_id: 1
role: critic
reward: -1.0
critique text (ungated: critic ran on an already-correct tau1)
--------
episode_id: 3
round_id: 2
role: executor
reward: 0.0
trajectory for episode 3 round 2 (regressed)
"""


def test_parse_trajectory_dump_counts_entries():
    entries = parse_trajectory_dump(SAMPLE_DUMP)
    # 4 episodes: 1 executor-only + 3x(executor, critic, executor) = 1 + 9 = 10
    assert len(entries) == 10
    assert entries[0].episode_id == 0
    assert entries[0].role == "executor"
    assert entries[0].reward == 1.0


def test_group_into_episodes_and_compute_metrics():
    entries = parse_trajectory_dump(SAMPLE_DUMP)
    outcomes = group_into_episodes(entries)
    assert len(outcomes) == 4

    by_id = {o.episode_id: o for o in outcomes}
    assert by_id[0].r1 == 1.0 and by_id[0].r2 == 1.0 and not by_id[0].critic_ran
    assert by_id[1].r1 == 0.0 and by_id[1].r2 == 1.0 and by_id[1].critic_ran
    assert by_id[2].r1 == 0.0 and by_id[2].r2 == 0.0 and by_id[2].critic_ran
    assert by_id[3].r1 == 1.0 and by_id[3].r2 == 0.0 and by_id[3].critic_ran

    metrics = compute_metrics(outcomes)
    assert metrics.num_episodes == 4
    assert metrics.fix_rate == pytest.approx(0.25)
    assert metrics.regression_rate == pytest.approx(0.25)
    assert metrics.no_op_rate == pytest.approx(0.5)
    assert metrics.critic_invocation_rate == pytest.approx(0.75)


def test_load_metrics_for_dir_reads_multiple_dump_files(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    rollouts_dir = exp_dir / "rollouts_train"
    rollouts_dir.mkdir(parents=True)
    (rollouts_dir / "train_0.txt").write_text(SAMPLE_DUMP)
    (rollouts_dir / "train_1.txt").write_text(SAMPLE_DUMP)

    metrics = load_metrics_for_dir(exp_dir)
    assert metrics.num_episodes == 8
    assert metrics.fix_rate == pytest.approx(0.25)


def test_load_metrics_for_dir_accepts_rollouts_train_directly(tmp_path: Path):
    rollouts_dir = tmp_path / "rollouts_train"
    rollouts_dir.mkdir()
    (rollouts_dir / "train_0.txt").write_text(SAMPLE_DUMP)

    metrics = load_metrics_for_dir(rollouts_dir)
    assert metrics.num_episodes == 4


def test_load_metrics_for_dir_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_metrics_for_dir(tmp_path / "nonexistent")


def test_no_episodes_returns_zeroed_metrics():
    metrics = compute_metrics([])
    assert metrics.num_episodes == 0
    assert metrics.fix_rate == 0.0
    assert metrics.regression_rate == 0.0
    assert metrics.no_op_rate == 0.0
