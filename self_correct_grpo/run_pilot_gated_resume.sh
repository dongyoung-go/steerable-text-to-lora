#!/usr/bin/env bash
# Resume the gated-arm pilot training run from its own last saved checkpoint, rather than
# restarting from the base model. Used to push PILOT_NUM_ROLLOUT as close to completion as
# possible across GPU-fragmentation-OOM crashes, instead of treating whichever checkpoint
# happened to exist at crash time as an acceptable final artifact (the preliminary experiment's
# checkpoint fidelity matters -- see docs/pilot_eval_design.md discussion).
#
# `--load` points at the previous run's checkpoint directory (its `checkpoints/` subdir, which
# holds `latest_checkpointed_iteration.txt` + `iter_XXXXXXX/`) instead of the base HF-converted
# torch-dist checkpoint; `--ref-load` stays pinned to the base model throughout (the KL-reference
# policy in ICRL's reward never moves). `train.py` infers `start_rollout_id` from whatever's
# loaded, so rollout numbering continues rather than restarting at 0.
#
# Caveat: checkpoints are saved with `no_save_optim=true` (see run_pilot_gated.sh's comment on the
# host-RAM checkpoint-save OOM), so there is no optimizer state to load -- pass
# no_load_optim/no_load_rng below (Megatron's load_checkpoint hard-errors on a missing 'optimizer'
# key otherwise) and each resume restarts Adam's momentum/variance from zero. A minor discontinuity
# given lr=1e-6, not a correctness bug, but real and worth keeping in mind when reading loss curves
# across a resume boundary.
#
# Usage:
#   ./run_pilot_gated_resume.sh <path/to/previous/run's/checkpoints/dir> [additional hydra overrides...]
set -euo pipefail

PREV_CHECKPOINTS_DIR="${1:?usage: run_pilot_gated_resume.sh <path/to/prior/training/run/checkpoints/dir> [overrides...]}"
shift

SELF_CORRECT_GRPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [[ ! -f "${PREV_CHECKPOINTS_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "error: ${PREV_CHECKPOINTS_DIR} has no latest_checkpointed_iteration.txt -- not a valid checkpoint dir" >&2
  exit 1
fi
RESUME_ITER="$(cat "${PREV_CHECKPOINTS_DIR}/latest_checkpointed_iteration.txt")"
echo "Resuming from iteration ${RESUME_ITER} in ${PREV_CHECKPOINTS_DIR}"

exec "${SELF_CORRECT_GRPO_DIR}/run_pilot_gated.sh" \
  checkpoint.cli.load="${PREV_CHECKPOINTS_DIR}" \
  +checkpoint.cli.no_load_optim=true \
  +checkpoint.cli.no_load_rng=true \
  "$@"
