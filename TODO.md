# TODO

## Background: v1 vs v2

- **v1**: the original training data, `textgrad_repro_gsm8k_*` — ~98% `<think>`-prefixed
  reasoning traces, one domain (GSM8K).
- **v2**: a cleaner replacement dataset, `textgrad_repro_v2_*` — no `<think>` tokens, 10
  domains so far (`gsm8k`, `aqua`, 8x `bbh_*`). Built and run as a fully parallel pipeline
  (`run_03b_*`, `run_04b_*`, `*_v2`-suffixed outputs) alongside v1, not a replacement in
  place, so v1 stays available for comparison. See `docs/03_training_validation.md` and
  `docs/04_downstream_eval.md` for the v1-vs-v2 result writeups.

## 1. Retire v1, make v2 the only pipeline

Once v2's results are trusted, delete v1's data, scripts, and `run_03_*`/`run_04_*` (non-`b`)
entry points, and rename the `*_v2`/`run_0Nb_*` paths to be the plain, single pipeline —
no more parallel `_v2`-suffixed tracks to keep in sync.

## 2. Train on the full textgrad/gepa task pool, not just today's 10

`data/textgrad_repro/` already has 27 domains reproduced (see `GEPA_REPRO_RESULTS.md`), and
`gepa_repro_*` is a second, separate prompt-optimization method over a similar task set —
both are unused beyond the 10 domains currently trained on. Expand training data to draw
from all tasks either method has produced. New tasks and new prompt-optimization methods
(beyond textgrad/gepa) will keep landing over time — task discovery and dataset-building
should scale to that without code changes each time, the same way `discover_tasks`'s
glob-based design already does for task *count*.

## 3. Filter out tasks a prompt-optimization method didn't actually improve

Some textgrad/gepa runs land on a task where the optimized prompt never beats (or barely
beats) baseline — that's a weak training signal for the oracle/hypernet pipeline and should
be excluded, not trained on uncritically. Add a filtering step keyed off each method's own
baseline-vs-final accuracy (already logged per task, e.g. `GEPA_REPRO_RESULTS.md`'s
`baseline_test`/`test` columns). Expect more filter criteria to be added later (e.g.
minimum row count, description diversity) — design this as a composable set of filters, not
a single hardcoded check.
