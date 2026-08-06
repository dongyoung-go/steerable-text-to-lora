# TextGrad Reproduction Results

Model: `Qwen/Qwen3-14B`, thinking disabled (`ENABLE_THINKING=0`, the
`run_all.sh` default). Optimizer: `TextualGradientDescent` over
`MAX_EPOCHS=3` x `STEPS_PER_EPOCH=4` (12 optimizer steps/task, batch size 3),
`OPTIMIZER_MAX_TOKENS=8000`. See `textgrad_repro_README.md` for the full
environment/library writeup and `TEXTGRAD_MULTITASK_PLAN.md` for the task
registry.

Run via `MODEL_DIR=Qwen/Qwen3-14B XDG_CACHE_HOME=/scratch/dg793/textgrad-cache
TASKS="..." ./scripts/textgrad_repro_run_all.sh` (the `XDG_CACHE_HOME`
redirect avoids an NFS/SQLite-WAL diskcache crash -- see
`textgrad_repro_README.md`). Per-task output dirs:
`data/textgrad_repro/qwen-qwen3-14b_<task>_textgrad-repro/`.

## Summary

28 of 29 registry tasks complete. Verified no `</think>` leakage in any
val/test/baseline_test file across all 28 (see "Verification" below).

| task | baseline val | final val | baseline test | final test | Δ test |
|---|---|---|---|---|---|
| aqua | 0.51 | 0.83 | 0.6181 | 0.7953 | +0.1772 |
| bbh_boolean_expressions | 1.00 | 1.00 | 0.98 | 1.00 | +0.02 |
| bbh_causal_judgement | 0.63 | 0.67 | 0.6757 | 0.7568 | +0.0811 |
| bbh_date_understanding | 0.91 | 0.94 | 0.88 | 0.89 | +0.01 |
| bbh_dyck_languages | 0.14 | 0.19 | 0.26 | 0.25 | -0.01 |
| bbh_formal_fallacies | 0.94 | 0.95 | 0.95 | 0.95 | 0.00 |
| bbh_geometric_shapes | 0.68 | 0.77 | 0.72 | 0.78 | +0.06 |
| bbh_hyperbaton | 0.98 | 0.99 | 0.98 | 0.97 | -0.01 |
| bbh_logical_deduction_seven_objects | 0.76 | 0.92 | 0.63 | 0.92 | +0.29 |
| bbh_movie_recommendation | 0.60 | 0.69 | 0.55 | 0.58 | +0.03 |
| bbh_multistep_arithmetic_two | 1.00 | 1.00 | 0.98 | 0.99 | +0.01 |
| bbh_navigate | 0.94 | 0.99 | 0.96 | 0.99 | +0.03 |
| bbh_object_counting | 0.95 | 1.00 | 0.91 | 0.98 | +0.07 |
| bbh_ruin_names | 0.64 | 0.81 | 0.66 | 0.83 | +0.17 |
| bbh_salient_translation_error_detection | 0.67 | 0.73 | 0.67 | 0.68 | +0.01 |
| bbh_snarks | 0.79 | 0.83 | 0.7857 | 0.8571 | +0.0714 |
| bbh_sports_understanding | 0.81 | 0.84 | 0.76 | 0.84 | +0.08 |
| bbh_temporal_sequences | 0.98 | 1.00 | 0.99 | 1.00 | +0.01 |
| bbh_tracking_shuffled_objects_seven_objects | 0.83 | 1.00 | 0.89 | 0.99 | +0.10 |
| bbh_web_of_lies | 0.91 | 1.00 | 0.98 | 0.96 | -0.02 |
| bbh_word_sorting | 0.00 | 0.40 | 0.00 | 0.38 | +0.38 |
| commonsenseqa | 0.82 | 0.86 | 0.8367 | 0.8533 | +0.0167 |
| gpqa_main | 0.36 | 0.46 | 0.4027 | 0.4933 | +0.0906 |
| gsm8k | 0.97 | 0.9733 | 0.9477 | 0.9515 | +0.0038 |
| mmlu_all | 0.91 | 0.95 | 0.7967 | 0.80 | +0.0033 |
| multiarith | 0.97 | 0.97 | 1.00 | 1.00 | 0.00 |
| strategyqa | 0.74 | 0.77 | 0.70 | 0.71 | +0.01 |
| trec | 0.72 | 0.78 | 0.78 | 0.798 | +0.018 |
| aime | -- | -- | -- | -- | not run |

## Notes

- `gsm8k`'s baseline val accuracy (`0.97`, 291/300) was backfilled after the
  fact via `scripts/textgrad_baseline_sweep.py --tasks gsm8k --no_enable_thinking`
  (that run predates the automated baseline-eval step landing in
  `textgrad_repro.py`'s `main()`, so it was never recorded originally). The
  sweep script 0-shots the same seed task description against the same
  300-row val split, so it's directly comparable to every other task's
  baseline-val column. `gsm8k`'s baseline test accuracy (`0.9477`) lives in
  a separate `baseline_test_accuracy.json` rather than inside
  `best_prompt.json` like every other task -- both are legitimate results,
  just from an older schema version.
- `aime` has not been run in this batch. It shares `TASKS` registry code
  with `gepa_repro.py`, which hit (and fixed) a real crash on this task: a
  degenerate ~13,434-digit numeral in a model response overflowed Python's
  `int()` conversion digit-safety limit. That fix landed in
  `scripts/textgrad_repro.py`'s `_parse_integer` too, so `aime` should be
  runnable, but hasn't been re-attempted since.
- Every other task in the ~29-entry registry (§5 of
  `TEXTGRAD_MULTITASK_PLAN.md`) is accounted for above.

## Verification

- All 28 completed tasks have a `best_prompt.json` (or, for `gsm8k`, the
  equivalent split across `best_prompt.json` + `baseline_test_accuracy.json`)
  with both a final and baseline test accuracy.
- Grepped every `val_eval.jsonl` / `test_eval.jsonl` / `baseline_test_eval.jsonl`
  across all 28 task dirs for `</think>` -- zero occurrences, confirming no
  thinking-mode leakage into scored outputs (see
  `data/textgrad_repro/[DoNotUse]old_thinking-enabled_runs/DEBUG_NOTES.md`
  for the diskcache-contamination bug this guards against, fixed in
  `scripts/textgrad_repro.py`'s `generate()` cache-key).
- Archived (thinking-contaminated or crashed) pre-fix data lives in
  `data/textgrad_repro/[DoNotUse]old_thinking-enabled_runs/`.
