#!/bin/bash
# Orchestrator over scripts/gepa_repro_run.sh: runs every task in
# textgrad_repro.py's TASKS registry (or a subset via TASKS=...) back to
# back, on one GPU, sequentially -- each task gets its own
# uv run --with-editable ... subprocess, so each pays its own ~1 model-load
# cost. Mirrors scripts/textgrad_repro_run_all.sh exactly (same skip/resume
# logic, same summary table shape), pointed at gepa_repro.py's
# data/gepa_repro/{model_dir}_{task}_gepa-repro/best_prompt.json outputs
# instead of textgrad_repro's data/textgrad_repro/..._textgrad-repro/ ones.
#
# See TEXTGRAD_MULTITASK_PLAN.md for the TASKS registry this iterates and
# gepa_repro_README.md / textgrad_repro_README.md for the ephemeral-overlay
# mechanics both task families share.
#
# Example (all tasks, defaults otherwise):
#   MODEL_DIR=Qwen/Qwen3-14B ./scripts/gepa_repro_run_all.sh
#
# Example (just gsm8k and one BBH task, force re-run even if already done):
#   TASKS="gsm8k bbh_object_counting" FORCE_RERUN=1 ./scripts/gepa_repro_run_all.sh
#
# Example (preview what would run without running it):
#   DRY_RUN=1 ./scripts/gepa_repro_run_all.sh
set -uo pipefail # deliberately not -e: one task failing shouldn't abort the rest

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3-14B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-3}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-3936}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
SEED="${SEED:-42}"
# Opt-in early-stop diagnostic, unset by default -- see gepa_repro_run.sh /
# gepa_repro.py's --no_improvement_patience. Off means every run here still
# matches the paper's budget-only stopping protocol.
NO_IMPROVEMENT_PATIENCE="${NO_IMPROVEMENT_PATIENCE:-}"

# Always eval_test by default -- the whole point of this wrapper is baseline
# + final *test* performance per task. Override only if you really just
# want val-only smoke runs across the board.
EVAL_TEST="${EVAL_TEST:-1}"

SKIP_DONE="${SKIP_DONE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
if [ "$FORCE_RERUN" = "1" ]; then
  SKIP_DONE="0"
fi
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

LOG_DIR="logs/gepa_repro_run_all"
mkdir -p "$LOG_DIR"

slugify() {
  # Must match textgrad_repro.py's slugify(): re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

SLUG="$(slugify "$MODEL_DIR")"

if [ -n "${TASKS:-}" ]; then
  TASK_LIST="$TASKS"
else
  echo "discovering tasks from textgrad_repro.py's TASKS registry (needs the ephemeral overlay once)..."
  # See textgrad_repro_run_all.sh for why this goes through a sentinel-
  # prefixed print rather than capturing raw stdout: `import vllm` (pulled
  # in transitively by gepa_repro.py -> textgrad_repro -> textgrad.engine.vllm)
  # prints its own INFO lines to stdout, not stderr.
  DISCOVERY_OUTPUT="$(
    uv run --with-editable ./textgrad_repro --with-editable ./gepa_repro \
      --index "https://download.pytorch.org/whl/cu128" --index-strategy unsafe-best-match \
      --with "vllm==0.11.0" --with "transformers==4.57.1" --with "kernels==0.10.0" \
      --with diskcache --with litellm --with cloudpickle --with tqdm \
      --with graphviz --with gdown --with tenacity --with python-dotenv \
      python -c "
import sys
sys.path.insert(0, 'scripts')
from textgrad_repro import TASKS
print('GEPA_TASKS:' + ' '.join(sorted(TASKS)))
"
  )"
  TASK_LIST="$(printf '%s\n' "$DISCOVERY_OUTPUT" | grep '^GEPA_TASKS:' | tail -n1 | sed 's/^GEPA_TASKS://')"
  if [ -z "$TASK_LIST" ]; then
    echo "failed to discover TASKS -- see output below" >&2
    echo "$DISCOVERY_OUTPUT" >&2
    exit 1
  fi
fi

# aime's prompts run far longer than every other task's (competition-math
# CoT, plus a bumped 16000-token forward budget), so it goes last regardless
# of where it falls alphabetically -- a crash/timeout on aime shouldn't
# block the rest of the (cheaper, faster) tasks from getting their turn
# first. Same ordering rule as textgrad_repro_run_all.sh.
if printf '%s\n' "$TASK_LIST" | grep -qw aime; then
  TASK_LIST="$(printf '%s\n' "$TASK_LIST" | tr ' ' '\n' | grep -vw aime | tr '\n' ' ') aime"
  TASK_LIST="$(printf '%s' "$TASK_LIST" | sed -E 's/^ +//; s/ +$//; s/ +/ /g')"
fi

echo "tasks to consider: $TASK_LIST"
echo

is_done() {
  # $1: best_prompt.json path. Done means it exists and, if EVAL_TEST=1,
  # also has test_accuracy AND baseline_test_accuracy fields (both added by
  # gepa_repro.py's main() only when --eval_test ran to completion).
  local best_json="$1"
  [ -f "$best_json" ] || return 1
  if [ "$EVAL_TEST" = "1" ]; then
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if 'test_accuracy' in d and 'baseline_test_accuracy' in d else 1)
" "$best_json"
  else
    return 0
  fi
}

declare -A STATUS
FAILED_ANY=0

for TASK in $TASK_LIST; do
  DATA_DIR="data/gepa_repro/${SLUG}_${TASK}_gepa-repro"
  BEST_JSON="$DATA_DIR/best_prompt.json"

  if [ "$SKIP_DONE" = "1" ] && is_done "$BEST_JSON"; then
    echo "[skip] $TASK -- already done ($BEST_JSON has test_accuracy)"
    STATUS["$TASK"]="skipped"
    continue
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would run $TASK -> $DATA_DIR"
    STATUS["$TASK"]="dry-run"
    continue
  fi

  echo "=== running $TASK ==="
  TASK_LOG="$LOG_DIR/${SLUG}_${TASK}.log"
  if MODEL_DIR="$MODEL_DIR" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    BATCH_SIZE="$BATCH_SIZE" \
    MAX_METRIC_CALLS="$MAX_METRIC_CALLS" \
    ENABLE_THINKING="$ENABLE_THINKING" \
    SEED="$SEED" \
    TASK="$TASK" \
    EVAL_TEST="$EVAL_TEST" \
    NO_IMPROVEMENT_PATIENCE="$NO_IMPROVEMENT_PATIENCE" \
    ./scripts/gepa_repro_run.sh 2>&1 | tee "$TASK_LOG"; then
    STATUS["$TASK"]="ok"
  else
    STATUS["$TASK"]="FAILED"
    FAILED_ANY=1
    echo "[FAILED] $TASK -- see $TASK_LOG" >&2
    if [ "$FAIL_FAST" = "1" ]; then
      echo "FAIL_FAST=1, stopping." >&2
      break
    fi
  fi
  echo
done

read_metrics() {
  # $1: best_prompt.json path (may not exist). Prints
  # "baseline_val final_val baseline_test test converged_early", the first
  # four each either a 4-decimal float or NA, the last "early"/"no"/"NA".
  # converged_early is only ever true when NO_IMPROVEMENT_PATIENCE was set
  # for that run (see gepa_repro.py's --no_improvement_patience) -- absent
  # that flag, every run always matches the paper's budget-only stopping
  # protocol and this column reads "no".
  local best_json="$1"
  if [ -f "$best_json" ]; then
    python3 -c "
import json, sys

def fmt(x):
    return f'{x:.4f}' if isinstance(x, (int, float)) else 'NA'

d = json.load(open(sys.argv[1]))
print(
    fmt(d.get('baseline_val_accuracy')),
    fmt(d.get('val_accuracy')),
    fmt(d.get('baseline_test_accuracy')),
    fmt(d.get('test_accuracy')),
    'early' if d.get('converged_early') else 'no',
)
" "$best_json"
  else
    echo "NA NA NA NA NA"
  fi
}

SUMMARY_FILE="$LOG_DIR/summary_${SLUG}_$(date +%Y%m%d_%H%M%S).txt"
{
  printf "%-48s %-10s %10s %10s %10s %10s %10s\n" "task" "status" "baseline_val" "final_val" "baseline_test" "test" "converged"
  for TASK in $TASK_LIST; do
    DATA_DIR="data/gepa_repro/${SLUG}_${TASK}_gepa-repro"
    read -r BASELINE_VAL FINAL_VAL BASELINE_TEST TEST_ACC CONVERGED_EARLY < <(read_metrics "$DATA_DIR/best_prompt.json")
    printf "%-48s %-10s %10s %10s %10s %10s %10s\n" "$TASK" "${STATUS[$TASK]:-not-run}" "$BASELINE_VAL" "$FINAL_VAL" "$BASELINE_TEST" "$TEST_ACC" "$CONVERGED_EARLY"
  done
} | tee "$SUMMARY_FILE"

echo
echo "summary saved to $SUMMARY_FILE"

if [ "$FAILED_ANY" = "1" ]; then
  exit 1
fi
