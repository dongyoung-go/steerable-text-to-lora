# GEPA Reproduction Results

Model: `Qwen/Qwen3-14B` (no thinking, stochastic decoding: temperature=0.6, top_p=0.95,
top_k=20, matching the GEPA paper's Appendix E.2). `reflection_minibatch_size=3`,
`max_metric_calls=3936`, `NO_IMPROVEMENT_PATIENCE=50` (opt-in early-stop diagnostic,
see `gepa_repro.py --no_improvement_patience`; does not change what accuracy is
reachable, only stops tasks that have flatlined before the full budget is spent).

Run via `TASKS="..." FORCE_RERUN=1 NO_IMPROVEMENT_PATIENCE=50 ./scripts/gepa_repro_run_all.sh`.
Full per-task logs: `logs/gepa_repro_run_all/qwen-qwen3-14b_<task>.log`.

## Summary

| task | baseline val | final val | baseline test | final test | converged early |
|---|---|---|---|---|---|
| bbh_boolean_expressions | 1.00 | 1.00 | 0.99 | 0.99 | yes |
| bbh_multistep_arithmetic_two | 0.99 | 1.00 | 0.99 | 0.99 | yes |
| bbh_object_counting | 0.94 | 0.99 | 0.90 | 0.99 | yes |
| bbh_word_sorting | 0.00 | 0.57 | 0.00 | 0.49 | yes |
| bbh_navigate | 0.97 | 0.97 | 0.93 | 0.91 | yes |
| bbh_sports_understanding | 0.81 | 0.85 | 0.79 | 0.82 | yes |
| bbh_web_of_lies | 0.96 | 1.00 | 0.95 | 0.99 | yes |
| bbh_temporal_sequences | 0.99 | 0.99 | 0.99 | 0.97 | yes |
| bbh_tracking_shuffled_objects_seven_objects | 0.83 | 0.95 | 0.92 | 0.96 | yes |
| bbh_salient_translation_error_detection | 0.68 | 0.68 | 0.68 | 0.64 | yes |
| bbh_ruin_names | 0.64 | 0.88 | 0.66 | 0.91 | no (hit budget) |
| bbh_snarks | 0.77 | 0.85 | 0.82 | 0.93 | yes |
| mmlu_all | 0.90 | 0.94 | 0.79 | 0.79 | yes |
| gpqa_main | 0.37 | 0.41 | 0.42 | 0.48 | yes |
| multiarith | 0.97 | 0.97 | 1.00 | 1.00 | yes |
| commonsenseqa | 0.81 | 0.84 | 0.84 | 0.84 | yes |
| strategyqa | 0.71 | 0.76 | 0.69 | 0.72 | yes |
| trec | 0.76 | 0.85 | 0.77 | 0.86 | yes |
| aime | — | — | — | — | pending re-run |

## Notes

- `aime` crashed on iteration 28 due to a real bug (not environment/GPU related): the
  task model degenerated into a ~13,434-digit numeral in one response, and Python's
  `int()` conversion hit CPython's 4300-digit safety limit
  (`sys.set_int_max_str_digits`), raising an unhandled `ValueError`. Fixed in
  `scripts/textgrad_repro.py`'s `_parse_integer` to treat absurdly-long digit strings
  as unparseable (wrong answer) instead of crashing. `aime`'s run dir was cleared and
  needs a re-run.
- This bug only ever manifests as a hard crash, never a silent mis-score, so
  `gsm8k` and `multiarith` (both of which ran to completion earlier without
  exceptions) were never affected and do not need to be re-run.
- Only `bbh_ruin_names` ran to the full metric-call budget without converging early —
  it was still improving when the budget ran out.
