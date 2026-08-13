# 06 — Description-paraphrase augmentation (v5 experiment)

**Status: data generation IMPLEMENTED and run; pipeline scripts IMPLEMENTED, not yet run --full.**
`scripts/paraphrase_descs.py` passes `ruff`/`pytest` on CPU (`tests/test_paraphrase_descs.py`); the
real generation run against the `v5` task-dir copies has been executed on the B200 node (see
"Results" below). The rest of the pipeline (`configs/data_v5.yaml`,
`scripts/reuse_oracle_loras.py`, `run_03_training_validation_v5.sh`,
`run_04_downstream_eval_v5.sh`, `run_all_v5.sh`, `scripts/compare_downstream_eval.py`) is now built
and passes `ruff`/`pytest` (CPU-only, no `--full`) — see "What's not built yet" for what's left,
which is now only the actual `--full` GPU run and result logging.

## Motivation

An investigation this session (see the `[inspection] t2l_train_desc is way worse than prompted`
session) found that the v3 hypernetwork's reconstruction warm-start collapsed to predicting a
near-constant, description-independent LoRA — `t2l_train_desc` scored barely above
`t2l_gibberish_desc`/`t2l_other_task_desc` downstream, and the recon stage's own training loss
never beat a "predict the mean" baseline for its entire 2000-step run. A major contributing
factor: every `textgrad_repro_v3_*`/`gepa_repro_v3_*` task dir has **exactly one description**, so
the D-axis (`splits.py`'s `d_holdout`) is universally `n/a` (see `docs/03_training_validation.md`
§4), and neither `build_recon_batches` (recon) nor `PerTaskDescDataset` (SFT) — both of which
already sample uniformly from a task's *whole* `descriptions` list every step — ever see more than
one phrasing per regression/training target to learn genuine phrasing-invariant, content-sensitive
conditioning from.

`docs/03_training_validation.md` §4 had already named this gap and proposed, as an unbuilt step
(b), a `scripts/paraphrase_descs.py` generating ~8 contrastive-sibling-aware paraphrases per task.
A working, previously-validated implementation of that exact idea already existed in the sibling
reference repo, `/home/dg793/text-to-lora/scripts/textgrad_repro_paraphrase_descs.py` (targets the
reference repo's own numbered-task-variant naming and generation/embedding backends this repo
doesn't have). This experiment ports that design in, generalized to work across any experiment
family following this repo's `<family>_<task>_d<K>` task-dir convention (v3, v4, and any future
one) — not hardcoded to v3 — and adds one new filtering rule beyond the reference design (see
"Two-tier filtering" below).

## Why a separate `v5` namespace, not applied to `v3` in place

The very first real run of `scripts/paraphrase_descs.py` was pointed directly at the real
`textgrad_repro_v3_*`/`gepa_repro_v3_*` task dirs under `/home/dg793/text-to-lora/tasks/` (the same
dirs `run_all_v3.sh --full` reads by name) and wrote new descriptions into them in place. That
mutation was later fully rolled back (every touched `metadata.yaml` truncated back to its original
single description — the script only ever *appends*, so this is exact and lossless) once it became
clear this doesn't actually give `v3`/the new experiment real separation: `run_all_v3.sh --full`
happening to skip the mutated data today is an accident of its own skip-if-exists/
already-at-`max_steps` checks (task-building skips because the dirs already exist; splits skip
without `--force`; recon/SFT skip because they're already at `step=2000`), **not** because the data
itself is isolated. If `v3` training were ever force-retrained later, it would have silently picked
up the new descriptions.

So the augmented data instead lives under a **new, fully independent namespace**:
`textgrad_repro_v5_*`/`gepa_repro_v5_*`, created as direct copies of the (pristine, rolled-back)
`v3` task dirs before generation ever touched them:

```
for each textgrad_repro_v3_<task>_d<K> / gepa_repro_v3_<task>_d<K> dir:
    copy to textgrad_repro_v5_<task>_d<K> / gepa_repro_v5_<task>_d<K>
```

`v3`'s task dirs are confirmed byte-for-byte back to their pre-session state (576/576 at exactly 1
description). `v5`'s copies share the *same* underlying `(question, response)` training data
(`ds_kwargs.data_files` in each `metadata.yaml` still points at the original
`data/textgrad_repro_v3_t2l/*.jsonl` files — only the task-dir *name* and its `descriptions` list
differ across versions; the actual training rows are identical and not duplicated) — `v5` is
purely "v3's data, with augmented descriptions," not a new data-collection run.

## `scripts/paraphrase_descs.py`

### Generalization mechanism

Takes `--tasks-root` and one or more `--train-tasks` glob patterns — the same interface
`discover_tasks()` and every other v3/v4 script already uses. No family name is hardcoded:

1. For each `--train-tasks` pattern, its literal prefix (text before the first wildcard) is
   derived automatically, e.g. `"textgrad_repro_v5_*"` → `"textgrad_repro_v5_"`.
2. A task's underlying task key = its dir name with that prefix stripped, then a trailing
   `_d<K>` stripped, e.g. `textgrad_repro_v5_bbh_causal_judgement_d9` → `bbh_causal_judgement`.
   (Not `task.metadata.domain`: that's a coarse category — `"bbh"` for every `bbh_*` task, see
   `domain_for()` in the builders — not a per-task key.)
3. Tasks are grouped by this key **across every family/algorithm pattern passed in one
   invocation** — e.g. passing both `textgrad_repro_v5_*` and `gepa_repro_v5_*` together (as the
   real run did) treats `aqua`'s textgrad and GEPA variants as one sibling group, since they
   really are the same underlying task.

### Generation and embedding backends

- **Generation**: this repo's own `scripts/gepa_repro_common.py` (`load_vllm_engine`,
  `batched_generate`) plus hand-rolled JSON parsing (`safe_parse_json`, ported from the reference
  repo's `textgrad_gen_backend.py`) with retry on unparseable output — nothing in this repo does
  guided/structured decoding. Run via the same ephemeral `uv run --with "vllm==0.11.0" --with
  "transformers==4.57.1" --with "kernels==0.10.0" ...` overlay every other vLLM-dependent script in
  this repo uses (`scripts/textgrad_repro.py`/`scripts/gepa_repro.py`'s own docstrings explain why:
  newer vLLM ships a CUDA-13-linked extension that fails to import against this machine's CUDA
  12.8 driver).
- **Embedding**: `Alibaba-NLP/gte-modernbert-base`, not the reference's `gte-large-en-v1.5` — same
  GTE lineage and CLS-pooling method, ~2.7x fewer params (149M vs ~409M) and no
  `trust_remote_code` requirement, for only ~1 point lower MTEB average (64.38 vs 65.39). A good
  trade for embedding a few thousand short instruction texts as a cheap side-computation next to
  the much more expensive vLLM generation step.

### Two-tier filtering

For task *T* in sibling group *G* (siblings = other tasks sharing *T*'s underlying task key;
outsiders = every task outside *G*, across every family/algorithm in scope):

```
sim_own             = cos(candidate, T's own original description)
sim_sibling_max     = max( cos(candidate, s's original) for s in siblings, EXCLUDING any
                            sibling whose own text is byte-identical to T's )
sim_same_task_mean  = mean( cos(candidate, x's original) for x in {T} + siblings, same exclusion )
sim_other_task_max  = max( cos(candidate, o's original) for o in outsiders, same exclusion )

keep if  sim_own >= sim_threshold
     AND sim_own - sim_sibling_max        >= contrast_margin      # rule 1: don't blur into a
                                                                   # sibling _dK's DIFFERENT
                                                                   # instruction for the same task
     AND sim_same_task_mean - sim_other_task_max >= cross_task_margin  # rule 2 (new vs. the
                                                                        # reference design): don't
                                                                        # blur into a different
                                                                        # task entirely
```

The byte-identical-sibling exclusion matters in practice: `textgrad_repro_v3_aqua_d0` and
`gepa_repro_v3_aqua_d0` turned out to share the exact same literal seed instruction (both
algorithms started `aqua` from the same unoptimized prompt) — since each algorithm's builder only
dedupes reverted/repeated instructions *within its own* iteration history, this exact-duplicate
pair slips through as two distinct task dirs. Without excluding it, `T`'s own contrast margin
against that "sibling" is mathematically unsatisfiable (a paraphrase can never score higher against
its own original than against a byte-identical copy of it). This is a real but *minor* effect
(exactly one such pair per task family in the real data) — the dominant reason many tasks still get
few or no new paraphrases is genuinely close (but non-identical) siblings from small, incremental
textgrad edits round-over-round (median closest-non-identical-sibling similarity ~0.95-0.97 across
every family checked).

### Margin tuning (empirical, done against real data)

The reference script's own default margins (`sim_threshold=0.80`, `contrast_margin=0.05`) turned
out to be **far too strict** for this repo's data: a first real run at those defaults kept **0 of
36** candidates for the `aqua` family alone, entirely rejected by the within-task contrast margin,
because many `_dK` siblings for the same task are themselves 0.90-1.00 similar (textgrad's
optimizer often only tweaks a few words per round). `contrast_margin`/`cross_task_margin` were both
loosened to **0.01** for the real run — this let real paraphrases through for tasks with
moderately-separated siblings, while tasks whose closest sibling is still ≥~0.99 similar even after
loosening still correctly get 0 new descriptions (there's no headroom a paraphrase could occupy
without looking more like a *different* real instruction than its own).

## Results

Real run (2026-08-10): `--train-tasks textgrad_repro_v5_* gepa_repro_v5_* --target-n-descs 8
--contrast-margin 0.01 --cross-task-margin 0.01`, logs under
`data/paraphrase_descs_logs_v5_full/` (one JSON per task: original/kept/dropped/raw_candidates,
plus raw model output on total generation failure).

**1461 new descriptions written across 576 task dirs.** Final description-count distribution:

| descriptions | task dirs |
|---|---|
| 1 (no paraphrases kept) | 295 |
| 2 | 30 |
| 3 | 22 |
| 4 | 19 |
| 5 | 22 |
| 6 | 22 |
| 7 | 30 |
| 8 (full target reached) | 136 |

Nearly identical to the (rolled-back) `v3`-in-place run's numbers (1457 new descriptions; 296/1,
136/8) — expected, since `v5` started from a byte-identical pristine copy of the same data and used
the same configuration; the 4-description difference is generation-temperature noise (`temperature
0.7`) between the two runs, not a meaningful change. The 295 zero-gain tasks are the same pattern
found earlier: their closest non-identical sibling is still similar enough (often ≥0.99) that even
a 0.01 margin can't be cleared — see "Margin tuning" above.

`v3`'s own task dirs were verified unaffected: all 576 confirmed still at exactly 1 description
each after this run (the augmentation only ever touched the `v5`-named copies).

## Pipeline scripts (implemented, not yet run --full)

Everything below this line was unbuilt as of the "IMPLEMENTED and run" status above; it is now
built (CPU-tested only, no GPU run yet):

- `configs/data_v5.yaml` — mirrors `configs/data_v3.yaml`, pointed at
  `[textgrad_repro_v5_*, gepa_repro_v5_*]`, own `cache_root: data/.cache_v5`.
- `scripts/reuse_oracle_loras.py` (+ `tests/test_reuse_oracle_loras.py`) — since `v5`'s task dirs
  share `v3`'s exact `(question, response)` rows (only `descriptions` differs), `v3`'s
  already-trained oracle LoRAs are numerically identical to what training against `v5` would
  produce. This script symlinks `outputs/oracle_loras_v5/<v5_name>` →
  `outputs/oracle_loras_v3/<v3_name>` (and the canonicalized `.pt` equivalents) instead of
  retraining — downstream consumers key oracle lookups strictly by task-dir name
  (`Path(oracle_dir) / task.name`), so this rename step is required, not optional.
- `run_03_training_validation_v5.sh` — mirrors `run_03_training_validation_v4.sh`'s shape (no
  task-build stage; `v5`'s task dirs already exist). `--full`: `make_splits.py` fresh against `v5`
  (the step that finally makes the D-axis non-degenerate, now that these tasks have >1 description
  each) → `reuse_oracle_loras.py` → `train_recon.py` → `train_sft.py` ×2 (scratch + warmstart) →
  `run_ablation.py`. Requires `outputs/oracle_loras_v3`/`outputs/oracle_loras_canon_v3` to already
  exist.
- `run_04_downstream_eval_v5.sh` — mirrors `run_04c_downstream_eval_v3.sh`'s winning-instruction
  task scope (not `v4`'s full-scope eval), since `outputs/eval/downstream_accuracy_full_v3.json` —
  the intended comparison target — was produced under that same restricted scope.
  `scripts/select_best_prompt_tasks_v3.py` hardcodes the `_v3_` prefix, so this script instead
  regenerates `data/best_prompt_tasks_v3.txt` fresh and derives `data/best_prompt_tasks_v5.txt` by
  substituting `_v3_` → `_v5_` (verified 1:1 against `v5`'s task dirs), then runs both
  `eval_downstream_accuracy.py` and `eval_downstream_accuracy_full.py` against the `v5` checkpoint/
  splits/oracle dir.
- `run_all_v5.sh` — mirrors `run_all_v4.sh`, orchestrating the two scripts above end to end.
- `scripts/compare_downstream_eval.py` (+ `tests/test_compare_downstream_eval.py`) — diffs two
  `downstream_accuracy_full_*.json` files (per-condition macro accuracy, macro comparisons, and a
  per-task table joined by stripping the `v3_`/`v5_` infix from task names). Run as:
  `python scripts/compare_downstream_eval.py outputs/eval/downstream_accuracy_full_v3.json outputs/eval/downstream_accuracy_full_v5.json --labels v3 v5`.

## Recon `max_steps` raised 2000 → 4000 (2026-08-12, ahead of the `v5` recon run)

The 2026-08-12 "second round" recon fix (per-group clip/LR, see
`docs/03_training_validation.md`) was re-verified with a standalone recon-only re-run against
`recon_v3` before committing to this experiment: collapse is eliminated (`cosine_similarity` rose
monotonically for the full 2000 steps, ending at 0.128, `best.pt == latest.pt`), but the curve was
still rising at step 2000, and the cosine LR schedule decays to ~0 by `max_steps`, so that
flattening tail can't be distinguished from genuine convergence. `configs/recon.yaml`'s
`max_steps` was bumped 2000 → 4000 (same `warmup_frac`, so warmup scales proportionally) so `v5`'s
recon run gives the schedule more room to tell the two apart — this affects `v5`'s recon run (and
any future `v3` re-run) since both read the same shared `configs/recon.yaml`.

## What's not built yet

Only the actual GPU run and its result logging remain:

- Run `bash run_all_v5.sh --full` on the B200 node (needs `v3`'s oracle LoRAs already trained).
- Run `scripts/compare_downstream_eval.py` against the resulting
  `outputs/eval/downstream_accuracy_full_v5.json` and `v3`'s equivalent, and record a "Results"
  section here (mirroring the generation-stage "Results" above) confirming whether the added
  description diversity actually fixes the steering collapse, per this experiment's original
  motivation.
