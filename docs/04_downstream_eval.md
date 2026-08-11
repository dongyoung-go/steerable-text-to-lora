# 04 — Downstream Accuracy Evaluation

**Status: IMPLEMENTED, unit-tested on CPU.** The v1 (GSM8K-only) real run was never finished
(started once, 2/13 tasks, `overall`/`comparisons` left `null` — see §10). **The v2 (10-domain)
real run completed on the B200 node on 2026-08-05 — see §11** for the result table and for the
multi-domain answer-parsing fix that run required. §12 is that run's conclusion (steering beats
doing nothing, beats prompting, and discriminates from controls) — **but §12's per-task sample
sizes were 2-10 rows, and §13 (same day, same checkpoint, official full test sets instead of the
tiny holdout) shows that conclusion does not fully survive at scale: the "beats doing nothing"
and "beats prompting" comparisons hold, macro-averaged, but the "discriminates real steering
from gibberish" claim does not — `t2l_gibberish_desc` macro-averages *higher* than
`t2l_train_desc` on the full test sets. Read §13 for the corrected picture; §12 is kept for
history but its point 3 is superseded.**

---

## 1. Why this document exists

`docs/03_training_validation.md` §4 is explicit and deliberate: **loss-based validation only**,
no generation, no accuracy harness. That was the right scope for training-time checkpoint
selection (cheap enough to run every `val_freq` steps) — but it means the real run's numbers
(`docs/03_training_validation.md`'s "Real-run result" table) only ever show **validation loss**,
never **is the final answer correct**. Those are related but not the same claim: lower loss on a
held-out response is suggestive of better generation, not proof of it.

This document specs the eval that actually answers:

> What is the downstream task accuracy (exact-match on GSM8K) of the T2L-generated LoRA using its
> best available steering description — and is that accuracy higher than (a) the frozen base
> model with no steering at all, or (b) the frozen base model given the *same* steering
> instruction directly as a prompt (no LoRA at all)?

Condition (b) — "just prompt it" — does not exist anywhere in the current codebase.
`data.formatting.format_example` / `TaskMetadata` hard-enforce `system_message == ""` specifically
*because* the steering instruction must be absent from the target's context for the LoRA-steering
experiment to mean anything (docs/02 and docs/03 both call this out). Measuring the prompted
baseline requires a deliberately separate code path that is *not* allowed to reuse or relax that
invariant.

---

## 2. Conditions

All six scored via real generation (not teacher forcing) over the same held-out rows, then
exact-integer-match against the gold answer:

| condition | LoRA | steering instruction in target's own prompt? | tests |
|---|---|---|---|
| `base` | none | absent | reference floor — no steering at all |
| `prompted` | none | **present** (system or user turn) | the "just prompt it" baseline — the natural alternative to LoRA-based steering |
| `oracle` | task's own trained LoRA (Stage A) | absent | reference ceiling |
| `t2l_train_desc` | hypernetwork-generated from a training description | absent | **the answer to the user's question** |
| `t2l_other_task_desc` | hypernetwork-generated from another task's description | absent | control: should be worse than `t2l_train_desc` |
| `t2l_gibberish_desc` | hypernetwork-generated from gibberish | absent | control: should be `≈ base` |

Mapping back to the original question:
- "final downstream accuracy of T2L with best prompt" = `t2l_train_desc` accuracy. ("Best" here
  means whichever training description was actually used to generate the LoRA; every task has
  exactly one description today per docs/03's D-axis limitation, so there is no selection to make
  yet — this becomes a real choice once multi-description tasks exist.)
- "higher than base prompting model" = `t2l_train_desc` vs. `base`.
- "or model equipped with best prompt" = `t2l_train_desc` vs. `prompted` — this is the
  comparison that actually justifies (or doesn't) the whole LoRA-generation machinery over just
  writing the instruction into the system prompt.

`t2l_other_task_desc` / `t2l_gibberish_desc` mirror `docs/03`'s controls, translated from loss to
accuracy: if `t2l_train_desc` doesn't clearly beat them here too, the hypernetwork is generating a
LoRA that isn't actually conditioned on the instruction's content, in accuracy terms — the same
collapse failure mode docs/03 §4 warns about, just measured a different way.

`eval_descs` (held-out paraphrase) is omitted from this table for the same reason it is `n/a`
throughout docs/03: no task has ≥2 descriptions yet. Add it here the moment the D axis exists.

---

## 3. Held-out set

Reuse the exact same Q-axis held-out rows docs/03 validation scored, via
`data.splits.resolve_q_holdout` against the same `data/splits.json`. This is what makes this
document's accuracy numbers and docs/03's per-task validation-loss numbers directly comparable,
row for row, rather than a separately-sampled and therefore not-quite-comparable eval set.

---

## 4. Generation

- **Greedy decoding** (`temperature=0` / `do_sample=False`), for reproducibility — matches the
  reference repo's `batched_generate(..., temperature=0)` default
  (`scripts/textgrad_repro.py`).
- **`max_new_tokens`**: generous. `docs/03`'s real length-profiling found responses hitting a hard
  ~2001-token cap from how they were originally generated, and set `inp_max_len=2560` to get 0%
  *training-time* truncation. Generation needs the same headroom — recommend `max_new_tokens` in
  the `2048`–`2560` range so a correct final answer is never cut off after a long `<think>` block.
- **LoRA injection** for the `oracle`/`t2l_*` conditions reuses `hooks.build_sites`/
  `hooks.lora_hooks` and (for `t2l_*`) `hypernet.generate_for_batch` exactly as
  `validation.score_condition` does — no new injection mechanism needed.
- ⚠️ **`lora_hooks` must stay attached for the *entire* generation loop, not one forward call.**
  `model.generate` calls the model repeatedly (incremental KV-cached decoding, one new token per
  step); a `with lora_hooks(...): model.generate(...)` block is almost certainly fine since the
  hooks are attached for the duration of that whole call — but this needs an explicit test, not
  an assumption, given `docs/03`'s bug #6 was exactly this class of "hooks silently not attached
  when a later computation actually needs them" mistake (there: gradient-checkpointing recompute;
  here: incremental decode steps). Verify hook coverage across every generated token before
  trusting any number this produces.
- `use_cache=True` for generation (training/validation always force `use_cache=False`, since KV
  caching is meaningless without incremental decoding).

## 5. The `prompted` condition — a genuinely separate code path

`data.formatting.format_example` cannot be reused with a non-empty `system_message`:
`TaskMetadata.__post_init__` raises if `system_message != ""`, by design (see docs/03 §1's
"system_message stays empty" invariant and its rationale). Do not weaken that invariant to make
this condition easier to build. Instead, `scripts/eval_downstream_accuracy.py` (or wherever this
lands) builds the prompted input itself: apply the tokenizer's chat template with the task's
steering description as the system turn (or prepended to the user turn, if the tokenizer's
template has no system role) and the question as the user turn, with **no LoRA hooks at all**.
This is the one place in the entire codebase where the instruction is deliberately allowed into
the target model's own context — keep it walled off, not threaded back into the shared
training/validation formatting path.

## 6. Answer parsing and the gold-answer question

`textgrad.tasks.big_bench_hard.parse_integer_answer` (vendored via `textgrad_repro/`, already
used by `scripts/textgrad_repro.py`) extracts the last numeric token from a response,
strips it to digits, and parses it as an int (returns `0` on any parse failure — check whether
that default is a hazard for a genuinely-answer-0 question before reusing it verbatim). Apply it
to both the generated response and the gold answer.

**Verified: a real gold answer is recoverable, by exact question-text join.** The
`textgrad_repro_gsm8k_NN.jsonl` rows used for training are `{"question", "response"}` pairs where
`response` was itself generated by Qwen3-32B — parsing *that* as "gold" would be circular (it
would just measure agreement with another model's guess, not correctness). Checked directly: the
canonical dataset (`datasets.load_dataset("gsm8k", "main", split="train")`, also what
`tasks/gsm8k/metadata.yaml` points at) contains the real gold answer with GSM8K's `"#### N"`
marker, and joins cleanly against the reformatted rows by exact question-text match after
stripping the `"Question: "` prefix the reformatted rows add (confirmed 20/20 on
`textgrad_repro_gsm8k_00`):

```python
ds = datasets.load_dataset("gsm8k", "main", split="train")
gold_by_question = {q.strip(): a for q, a in zip(ds["question"], ds["answer"])}

def gold_answer(row: dict) -> str:
    q = row["question"]
    q = q[len("Question: "):].strip() if q.startswith("Question: ") else q.strip()
    return gold_by_question[q]  # KeyError -> a row the join didn't cover; do not silently skip it
```

Build this join once per task at eval-set-construction time (not per-row at scoring time), and
raise loudly on any `KeyError` rather than dropping unmatched rows silently -- a silent drop would
quietly shrink the "held-out" set to whatever happens to join, which is exactly the kind of
mismatch docs/03's D-axis handling was careful to make loud (`n/a` + a warning) rather than quiet.

## 7. Metrics

- Per-task, per-condition accuracy = (exact-integer-matches) / (scored rows).
- Aggregate by macro-averaging across tasks (not pooling rows across tasks), mirroring
  `scripts/run_ablation.py`'s per-task steering-margin averaging — so one large task
  (`textgrad_repro_gsm8k_12` has ~5x the rows of most others) doesn't dominate the headline number.
- The three comparisons that answer the original question, computed per-task and macro-averaged:
  - `t2l_train_desc` − `base`: does steering help at all over doing nothing?
  - `t2l_train_desc` − `prompted`: does LoRA-based steering beat just prompting the frozen model
    with the same instruction? (The load-bearing comparison — if `prompted` wins, the LoRA
    machinery is not pulling its weight relative to a system prompt.)
  - `t2l_train_desc` / `oracle`: fraction of oracle headroom recovered, in accuracy terms (the
    exact-match analogue of docs/03's `val_loss(hypernet) − val_loss(oracle)`).

## 8. Non-goals / scope boundary

- No vLLM. Consistent with `docs/01_env.md`'s "`vllm` only if generation-based eval is added
  later" note — plain `model.generate` is enough at this data/model scale (13 tasks, ~10%
  held-out rows each).
- No re-litigation of any loss-based metric — this document is additive to, not a replacement
  for, `docs/03_training_validation.md` §4.
- No `scripts/paraphrase_descs.py` / D-axis work — `eval_descs`-equivalent stays out of scope
  here for the same reason it's `n/a` in docs/03.

## 9. Build order (as implemented)

1. `src/steerable_t2l/data/gold_answers.py` — the gold-answer join (§6). `load_gold_index()`
   builds `{stripped question -> "#### N"}` **once globally** (the join target is the single
   shared canonical `datasets.load_dataset("gsm8k", "main", split="train")`, not a per-task
   dataset — every `textgrad_repro_gsm8k_NN` task's questions are a subset of it), raising
   `KeyError` loudly on any miss. `strip_question_prefix`/`gold_answer` are pure and unit-tested
   without network; `load_gold_index` itself needs one live GSM8K fetch (or an HF cache hit).
2. `src/steerable_t2l/eval_accuracy.py` — generation + scoring. Reuses `hooks.build_sites`/
   `lora_hooks`, `hypernet.generate_for_batch`, `oracle.canonicalize.load_and_canonicalize_oracle`
   (per §9 item 2's original plan), and a from-scratch `parse_integer_answer` — **not** the
   vendored `textgrad.tasks.big_bench_hard.parse_integer_answer`, which defaults to `0` on parse
   failure (§6's flagged hazard); this one returns `None` and a `None` on either side never
   counts as a match, matching `scripts/textgrad_repro.py`'s own `_parse_integer` convention.
3. `scripts/eval_downstream_accuracy.py` — CLI wiring: loads the target model, a trained
   hypernet checkpoint (`checkpoint.load_hypernet`), `data/splits.json`, iterates every
   condition over every non-T-held-out task's `data.splits.resolve_q_holdout` rows, writes a
   JSON report.
4. `tests/test_eval_accuracy.py::test_lora_hooks_stay_attached_across_multi_token_generate` —
   the §4 ⚠️ verification: wraps `hooks._make_hook` to count forward-hook firings across a
   real multi-token `model.generate()` call and asserts more firings occur than one prefill
   pass alone would produce, confirming the hooks stay live through incremental decode.
5. `run_04_downstream_eval.sh` — the single-command wrapper (`bash run_04_downstream_eval.sh`
   for lint+tests; `bash run_04_downstream_eval.sh --full` for the real B200 run), mirroring
   `run_03_training_validation.sh`'s shape.
6. **Not yet done: the real run.** Needs a B200 node — see §10.

One deliberate design choice beyond what §1–§8 specified: every condition generates **one LoRA
per `(task, condition)`** from one fixed description and applies it to every held-out row of
that task, rather than resampling a description per row the way `validation.py` does for its
loss-based margin. This document's accuracy numbers describe a single concrete LoRA artifact
per task/condition ("the T2L-generated LoRA using its best available steering description"),
not an average over many resampled ones. `prompted` and `t2l_train_desc` are resolved to the
*same* fixed description per task (`eval_accuracy.condition_desc`), since §2's `prompted` row is
explicitly "the same instruction" as the training description used for `t2l_train_desc`.

## 10. Running the real eval

Not run yet — this was implemented and verified from a CPU-only node. To run for real on the
B200 node once a trained hypernet checkpoint exists (`docs/03`'s `sft_scratch`/`sft_warmstart`):

```bash
HYPERNET_CKPT=outputs/checkpoints/sft_warmstart/latest.pt \
ORACLE_DIR=outputs/oracle_loras \
bash run_04_downstream_eval.sh --full
```

This calls `scripts/eval_downstream_accuracy.py` against the real 13-task GSM8K data
(`TASKS_ROOT`, default `/home/dg793/text-to-lora/tasks`) and real Qwen2.5-1.5B-Instruct
weights, writing `outputs/eval/downstream_accuracy.json` plus the §7 macro-averaged
comparisons to stdout. Confirmed on the CPU node that the script correctly reaches real
generation (loads the real weights, builds real prompts, calls `model.generate`) and only
fails there for the expected CPU-incompatibility reason: `--attn-implementation` defaults to
the same pinned `kernels-community/flash-attn2` CUDA kernel `scripts/train_sft.py` uses
(docs/03's GPU-bugs §5), which has no CPU backend — exactly the failure mode expected on a
non-GPU node, not a bug. The v1 real run against `outputs/eval/downstream_accuracy.json` was
started once on the B200 node but never finished or resumed: only 2 of 13 `textgrad_repro_gsm8k_*`
tasks have recorded results, and top-level `"overall"`/`"comparisons"` are both `null`.

## 11. v2 dataset run (2026-08-05) — non-`<think>`, 10-domain data, and the multi-domain answer-parsing fix

**What "v1" and "v2" mean here** (same distinction as `docs/03_training_validation.md`'s
2026-08-04 changelog entry): **v1** is the original real dataset behind
`textgrad_repro_gsm8k_NN` — ~98% `<think>`-prefixed reasoning traces, a single domain (GSM8K).
**v2** is a new raw dump at `data/textgrad_repro/` (10 domains used so far: `gsm8k`, `aqua`,
8x `bbh_*`; no `<think>` tokens at all), built to test whether v1's `<think>`-heavy,
single-domain data was causing excess distribution shift between the base model and the
oracle/SFT training targets. `scripts/build_tasks_from_textgrad_repro_v2.py` converts it into
this pipeline's task format, and the whole pipeline (training in `run_03b_training_validation_v2.sh`,
downstream eval in `run_04b_downstream_eval_v2.sh`) is mirrored stage-for-stage into
`*_v2`-suffixed outputs so v1's outputs and the raw v1/v2 source datasets are never touched.

**This is the first downstream-accuracy real run to actually finish** (v1's own attempt above
never got past 2/13 tasks). Getting it running past the very first row surfaced a real gap,
not present in v1: `eval_downstream_accuracy.py` was built entirely around GSM8K's shape —
§6's gold-answer join hits the canonical HF `gsm8k` dataset specifically, and
`parse_integer_answer` only extracts a trailing integer. Neither works for v2's other 9
domains, whose gold answers are letters (`aqua`, most `bbh_*`), yes/no (`bbh_causal_judgement`),
valid/invalid (`bbh_formal_fallacies`), or bracket sequences (`bbh_dyck_languages`) — running
`--full` crashed immediately with a `KeyError` on the very first non-GSM8K row.

Fixed in two parts:
- `scripts/build_tasks_from_textgrad_repro_v2.py` now threads `forward_outputs.jsonl`'s own
  `gold_answer` field straight into each built task row (e.g. `"C"`, `"(E)"`, `"invalid"`,
  `"2200"` — whatever bare final-answer form that domain uses), removing the need for any
  external per-domain dataset join.
- `src/steerable_t2l/eval_accuracy.py` gained `parse_mcq_letter_answer`/`parse_exact_answer`
  (ported from `scripts/textgrad_repro.py`'s own `_parse_mcq_letter`/`_parse_exact` — the
  sibling text-to-lora repo's textgrad reproduction, which is where this dataset's own
  `correct` labels were originally computed) and `classify_answer_parser`, which infers
  integer/mcq_letter/exact straight from a task's own embedded gold-answer values rather than
  a hardcoded per-task table — so this keeps working unchanged for any future domain added to
  `data/textgrad_repro/` (e.g. the wider set already reproduced in `GEPA_REPRO_RESULTS.md`:
  `mmlu_all`, `gpqa_main`, `commonsenseqa`, `strategyqa`, `trec`, `aime`, more `bbh_*`
  subtasks), with no code changes. Legacy GSM8K tasks built without the embedded field
  (v1) fall back unchanged to the original external-join + integer-parser path — verified by
  `tests/test_eval_accuracy.py`'s existing suite passing unmodified, plus new tests for the
  three parsers and the classifier.

**Real-run result** (`outputs/checkpoints/sft_warmstart_v2`, `data/splits_v2.json`,
`outputs/oracle_loras_v2`, `outputs/eval/downstream_accuracy_v2.json`), macro-averaged over the
8 trained tasks (`gsm8k` and `bbh_hyperbaton` are v2's T-axis holdout tasks and are correctly
excluded — this eval's scope is trained tasks only, §8):

| condition | accuracy |
|---|---|
| `base` | 0.261 |
| `prompted` | 0.502 |
| `oracle` | 0.353 |
| `t2l_train_desc` | **0.687** |
| `t2l_other_task_desc` (control) | 0.410 |
| `t2l_gibberish_desc` (control) | 0.455 |

Comparisons: `t2l_train_desc − base` = **+0.426**, `t2l_train_desc − prompted` = **+0.185**
(the LoRA beats even handing the model the same instruction directly), `t2l_train_desc / oracle`
= **2.02x** (the steered LoRA roughly doubles the individually-trained oracle LoRA's accuracy).
`t2l_train_desc` wins or ties on 6 of 8 tasks and beats both controls on the macro average — real
task-specific steering signal, not just "any LoRA helps."

Two per-task exceptions worth flagging rather than averaging away: `bbh_causal_judgement` scores
low across every condition (0.0-0.286) — confirmed by inspecting raw generations directly that
this is genuine model weakness (garbled/incoherent chain-of-thought reasoning on this specific
task) and not a scoring-parser bug, since `parse_exact_answer` correctly isolated the model's
stated Yes/No answer in every sample checked. `bbh_movie_recommendation` is the one task where
`oracle` (0.857) clearly beats `t2l_train_desc` (0.571) — the only reversal of the usual
`t2l_train_desc > oracle` pattern, plausibly noise given only 7 held-out rows, not yet
investigated further.

Per-task breakdown, `outputs/eval/downstream_accuracy_v2.json`:

| task | base | prompted | oracle | t2l_train_desc | t2l_other_task | t2l_gibberish |
|---|---|---|---|---|---|---|
| aqua | 0.500 (4/8) | 0.750 (6/8) | 0.500 (4/8) | 0.875 (7/8) | 0.625 (5/8) | 0.625 (5/8) |
| bbh_causal_judgement | 0.000 (0/7) | 0.286 (2/7) | 0.000 (0/7) | 0.143 (1/7) | 0.000 (0/7) | 0.286 (2/7) |
| bbh_date_understanding | 0.444 (4/9) | 0.778 (7/9) | 0.556 (5/9) | 0.889 (8/9) | 0.889 (8/9) | 0.667 (6/9) |
| bbh_dyck_languages | 0.000 (0/2) | 0.000 (0/2) | 0.000 (0/2) | 1.000 (2/2) | 0.000 (0/2) | 0.000 (0/2) |
| bbh_formal_fallacies | 0.100 (1/10) | 0.700 (7/10) | 0.300 (3/10) | 0.700 (7/10) | 0.400 (4/10) | 0.500 (5/10) |
| bbh_geometric_shapes | 0.250 (2/8) | 0.375 (3/8) | 0.500 (4/8) | 0.875 (7/8) | 0.750 (6/8) | 0.375 (3/8) |
| bbh_logical_deduction_7obj | 0.222 (2/9) | 0.556 (5/9) | 0.111 (1/9) | 0.444 (4/9) | 0.333 (3/9) | 0.333 (3/9) |
| bbh_movie_recommendation | 0.571 (4/7) | 0.571 (4/7) | 0.857 (6/7) | 0.571 (4/7) | 0.286 (2/7) | 0.857 (6/7) |

**Note on the raw source data's actual size**: `data/textgrad_repro/` contains 27 domains total
(not just the 10 v2 currently trains on — the wider set matches `GEPA_REPRO_RESULTS.md`'s task
list: `mmlu_all`, `gpqa_main`, `commonsenseqa`, `strategyqa`, `trec`, `aime`, and more `bbh_*`
subtasks). `build_tasks_from_textgrad_repro_v2.py` builds task dirs for every domain present
under `data/textgrad_repro/` unconditionally, so re-running it (as this fix required) also
regenerated task dirs for all 27, not just the 10 already trained on — this did **not** touch
`data/splits_v2.json` or any checkpoint, and this eval run was explicitly scoped via
`--train-tasks` to only the 10 domains `sft_warmstart_v2` actually trained on. Expanding v2 to
more of the available 27 domains would need a fresh `make_splits.py` + oracle/recon/SFT training
run, not just an eval-side change — not yet done.

## 12. Overall conclusion — does the hypernetwork steering actually work?

> **⚠️ Superseded in part by §13.** This section's conclusion was computed on §11's 2-10-row
> Q-holdout per task. §13 (2026-08-05, same checkpoint, official full test sets — 37-1319
> rows/task) confirms points 1 and 2 below macro-averaged, but **not** point 3: on the full test
> sets, `t2l_gibberish_desc` macro-averages *higher* than `t2l_train_desc`. Kept below for
> history; treat §13 as the current bottom line.

**Yes.** Both the loss-based validation (`docs/03_training_validation.md`'s v2 real-run result)
and this document's real generation-accuracy run (§11) agree, on independent metrics computed
from independent code paths, that the recon-warm-started hypernetwork learns genuine
task-conditional steering — not just "any LoRA helps," and not just memorized outputs.

**The load-bearing evidence, in order of how directly it answers "is this worth the complexity":**

1. **It beats doing nothing.** `t2l_train_desc` macro accuracy 0.687 vs. `base` 0.261
   (+0.426) — matches the loss side, where `steering_margin` vs. `base` is large and positive
   for every trained task.
2. **It beats just prompting the frozen model with the same instruction.** `t2l_train_desc`
   (0.687) vs. `prompted` (0.502), +0.185. This is the single most important number in this
   whole evaluation: if `prompted` had won, the entire LoRA-generation machinery (hypernetwork,
   recon warm-start, SFT) would be adding complexity without adding capability over a plain
   system prompt. It didn't win — steering via a generated LoRA does something a system prompt
   alone can't.
3. **It discriminates real steering from fake steering.** `t2l_train_desc` (0.687) clearly beats
   both negative controls — `t2l_other_task_desc` (0.410, a real but *wrong* task's description)
   and `t2l_gibberish_desc` (0.455, a nonsense description) — and the loss-side
   `steering_margin` (vs. gibberish 0.641, vs. other-task 0.858) is the same finding in a
   different metric. A hypernetwork that just learned "always emit a generically-helpful LoRA"
   would score similarly across all three; it doesn't.
4. **The recon warm-start is doing real work, not just providing a good initialization that
   training would reach anyway.** From-scratch SFT collapses to a near-constant,
   instruction-ignoring LoRA (loss-side steering margin 0.017-0.063); the warm-started arm's
   margin is 38-50x larger. Skip the recon stage and the method does not work at this data
   scale/step budget.
5. **It beats the per-task oracle LoRA on average** (`t2l_train_desc` / `oracle` = 2.02x
   accuracy) — a real result, not a bug: the oracle is a from-scratch, single-task LoRA with no
   access to the task description at all, so this measures the hypernetwork's ability to use
   (a) the steering instruction itself and (b) cross-task shared structure from recon
   pretraining, neither of which the oracle baseline has. It is not a claim that generated LoRAs
   are better than fine-tuning could ever do with more budget/tuning per task.

**What this does *not* yet establish (open gaps, not contradicting evidence):**

- **True zero-shot task generalization is only measured in loss, not accuracy.** The two v2
  T-axis holdout tasks (`gsm8k`, `bbh_hyperbaton`) show near-in-distribution `val_loss` (§per
  `docs/03`: 0.022-0.028 vs. 0.018-0.021 for trained tasks) — encouraging, but this document's
  real-generation accuracy eval explicitly excludes T-holdout tasks (§8/§9), so "does the
  generated LoRA for a never-trained-on task actually solve real problems, not just predict
  plausible next tokens" is untested. Running `eval_downstream_accuracy.py` against
  `gsm8k`/`bbh_hyperbaton` (dropping the current trained-tasks-only restriction) would close
  this gap.
- **Per-task sample sizes are small** (7-10 held-out rows/task, `bbh_dyck_languages` only 2) —
  individual task percentages swing by ~10-15 points per flipped answer. Trust the
  macro-averaged numbers and the direction of the four comparisons above; don't over-read any
  single task's exact percentage.
- **`bbh_causal_judgement` is weak in every condition** (0.0-0.286) — confirmed by inspecting
  raw generations directly (§11) that this is the base model's own reasoning quality on a
  genuinely hard task (garbled, self-contradictory chains of thought), not a scoring bug or a
  steering failure specific to this pipeline.
- **`bbh_movie_recommendation` is the one task where `oracle` beats `t2l_train_desc`** (0.857 vs.
  0.571) — the sole reversal of the pattern in point 5 above. Plausibly noise given n=7, but
  not yet investigated further; worth a look before treating point 5 as true for every domain.
- **Only 10 of the 27 domains available in `data/textgrad_repro/` have been trained on** — these
  conclusions are established for this 10-domain mix, not yet verified to hold as more/different
  domains are added (see §11's closing note).

**Bottom line for anyone deciding whether to trust this method:** on the evidence gathered so
far, task-conditional LoRA generation via this hypernetwork does what it's supposed to
do — it steers, the steering is instruction-specific rather than generic, and it beats the
simplest alternative (just prompting the model). The open items above are about *how far* this
generalizes (unseen tasks, more domains, per-task robustness), not about *whether the core
mechanism works*.

---

## 13. Full official test set follow-up (2026-08-05) — a less noisy but less flattering picture

§12's conclusion was built on §11's Q-holdout, which is only 2-10 rows/task (`docs/04`'s own
"per-task sample sizes are small" caveat) — a ~10% holdout of textgrad_repro's own small *val*
subsample, not each domain's real, much larger, official test split. This section re-scores the
same `sft_warmstart_v2` checkpoint's `base`/`prompted`/`oracle`/`t2l_train_desc`/
`t2l_other_task_desc`/`t2l_gibberish_desc` conditions (`oracle` added later the same day — see
the end of this section for how)
against each domain's actual official test set: **1,319 rows for gsm8k, 254 for aqua, 37-100 for
each BBH subtask** — up to ~45x the row count of §11's holdout, and genuinely disjoint from it
by construction (official train/val/test split boundaries, not ad hoc slicing — see
`src/steerable_t2l/data/external_testsets.py`'s module docstring for the exact disjointness
argument per domain).

New infrastructure this required, all merged and tested (`tests/test_external_testsets.py`,
`tests/test_eval_accuracy.py`):
- `src/steerable_t2l/data/external_testsets.py` — per-domain full-test-set loaders (gsm8k, aqua,
  every BBH subtask by name, plus mmlu_all/gpqa_main/commonsenseqa/strategyqa/trec/multiarith
  for the not-yet-trained-domain follow-up, still TODO — see below).
- `eval_accuracy.run_downstream_eval` gained a `rows_for_task` override (defaults to the
  existing Q-holdout `eval_rows_for_task`) so the same scoring engine
  (`score_condition`/`generate_texts`/`build_sites`/`lora_hooks`) runs unchanged against either
  row source — no new scoring logic, only a swapped row source.
- `scripts/eval_downstream_accuracy_full.py` — the CLI wrapper, mirroring
  `eval_downstream_accuracy.py`'s shape.
- **`--gen-batch-size` matters a lot.** The default of 8 leaves a B200 at ~26-30% utilization;
  a sweep found `--gen-batch-size 64` gives ~9x throughput (4.63s/row → 0.52s/row) at only 5.6GB
  peak memory, with batch=128 actually *slower* than 64 (padding-to-longest-in-batch stragglers)
  — 64 is the sweet spot, not "as large as memory allows."

### Two real bugs found and fixed while building this

**Bug 1 — `condition_desc` picked the wrong description for `prompted`/`t2l_train_desc`.**
`task.metadata.descriptions` is collected in *first-appearance order* across textgrad's
optimization iterations, not accuracy order — index 0 is textgrad's unoptimized *seed* prompt,
not the best-converged one. `condition_desc` unconditionally returned `pool[0]`, so both
`prompted` and `t2l_train_desc` were being scored against the seed instruction, not "the task's
own training description" the top of this document claims (§1: "the exact description that was
used to generate the LoRA"). Confirmed empirically for aqua: `descriptions[0]` had textgrad's
own `val_accuracy=0.64`; the description that actually produced the SFT training responses
(iteration 10, reusing iteration 9's prompt) was `descriptions[7]`, `val_accuracy=0.83`.
Checked across 4 tasks — only 2/4 had the best description coincidentally at index 0.

Fixed by threading through an explicit `best_description_index` field: `TaskMetadata` gained the
field (`src/steerable_t2l/data/metadata.py`), `build_tasks_from_textgrad_repro_v2.py` computes
it (which description was live at the best textgrad iteration) and writes it into every task's
`metadata.yaml`, and `condition_desc` now prefers it over `pool[0]`, falling back to `pool[0]`
only when unset (legacy tasks) or when the best description is itself D-held-out. Rebuilt every
v2 task's metadata with this field (`--min-samples 0`, to also cover the two tasks --min-samples
50's default had been silently excluding: `bbh_dyck_languages` (19 correct rows) and
`bbh_word_sorting` (40) — those were already-trained, already-in-splits.json tasks that the
default min-samples threshold would have dropped entirely had their directories not already
existed from an earlier build). Verified byte-identical jsonl content before/after the rebuild
(`diff`, 0 lines) — only `metadata.yaml` gained the new field.

One counterintuitive result from the fix: on aqua, using the *actually-optimized* description
(textgrad `val_accuracy=0.83`) made `prompted` score *worse* (0.402) than the old buggy run using
the unoptimized seed prompt (0.535). Plausible explanation: textgrad optimized these
descriptions against Qwen3-14B (the teacher), and its later iterations get longer and more
elaborate ("restate constraints in bullets," "verify units," "compare computed value to answer
choices with precise language...") — that kind of instruction may help a 14B model and overload
a 1.5B model's instruction-following instead.

**Bug 2 — `classify_answer_parser`'s strict `all()` let one bad data point break an entire
task.** BBH's official `movie_recommendation` test JSON has one row whose `target` is
`"Monsters, Inc"` — a full movie title, not a letter (upstream split that title across two
lettered options, `(A) "Monsters"` / `(B) "Inc"`, and left the target unsplit). Requiring
*every* gold value to match the mcq-letter pattern before selecting that parser meant this one
outlier (1/100) forced the whole task onto the `"exact"` parser, which can't match any free-text
model response — silently producing a false **0.000 across every condition** for this task. Not
a model failure: manually generating on these exact rows showed the model correctly answering
and the *gold* values also parsing fine individually; the classifier itself was the bug.

Fixed with a 90%-majority-vote threshold instead of `all()` (`classify_answer_parser`) — the one
outlier row still can't be scored correctly (costs at most 1/100), but no longer breaks parsing
for the other 99. Swept every domain's full test set afterward to confirm no other task has a
similar hidden outlier (`movie_recommendation` was the only one; every other task classified as
`"exact"` has genuinely 0% mcq/integer-shaped gold values — real free-text tasks, not bugs).

### Result

Macro-averaged over the same 8 trained tasks as §11 (`gsm8k`/`bbh_hyperbaton` still excluded —
T-holdout, still untested in accuracy, see §12's open-gaps list):

| condition | accuracy |
|---|---|
| `base` | 0.213 |
| `prompted` | 0.280 |
| `oracle` | 0.366 |
| `t2l_train_desc` | **0.355** |
| `t2l_other_task_desc` (control) | 0.323 |
| `t2l_gibberish_desc` (control) | **0.378** |

Comparisons: `t2l_train_desc − base` = **+0.142**, `t2l_train_desc − prompted` = **+0.076**
(both positive, same direction as §12, smaller margins), `t2l_train_desc − t2l_other_task_desc`
= +0.033 (correct direction), **`t2l_train_desc − t2l_gibberish_desc` = −0.023** (wrong
direction — the gibberish control scores higher, macro-averaged). `t2l_train_desc / oracle`
(macro-averaged per-task ratio, matching §12's methodology) = **1.100** — but by simple
macro-averaged accuracy, `oracle` (0.366) now edges out `t2l_train_desc` (0.355); see the
oracle discussion below for why these two ways of comparing them disagree.

Per-task (`outputs/eval/downstream_accuracy_full_v2.json`), oracle added 2026-08-05 (run added
after the rest of this table — see the former TODO, now resolved, at the end of this section):

| task | base | prompted | oracle | t2l_train_desc | t2l_other_task | t2l_gibberish |
|---|---|---|---|---|---|---|
| aqua (n=254) | 0.421 | 0.402 | 0.488 | 0.433 | 0.398 | 0.449 |
| bbh_causal_judgement (n=37) | 0.054 | 0.297 | 0.000 | 0.000 | 0.135 | 0.216 |
| bbh_date_understanding (n=100) | 0.420 | 0.400 | 0.590 | 0.560 | 0.560 | 0.530 |
| bbh_dyck_languages (n=100) | 0.000 | 0.030 | 0.050 | 0.100 | 0.000 | 0.000 |
| bbh_formal_fallacies (n=100) | 0.010 | 0.340 | 0.340 | 0.210 | 0.380 | 0.430 |
| bbh_geometric_shapes (n=100) | 0.170 | 0.130 | 0.470 | 0.530 | 0.430 | 0.430 |
| bbh_logical_deduction_seven_objects (n=100) | 0.200 | 0.250 | 0.270 | 0.310 | 0.110 | 0.330 |
| bbh_movie_recommendation (n=100) | 0.430 | 0.390 | 0.720 | 0.700 | 0.570 | 0.640 |

### What holds up from §12, and what doesn't

**Holds up, macro-averaged:**
- **Point 1 (beats doing nothing).** +0.142, same direction, smaller margin than §12's +0.426.
- **Point 2 (beats prompting).** +0.076, same direction, smaller margin than §12's +0.185.
- `t2l_train_desc` beats `t2l_other_task_desc` on the macro average (+0.033) and on 5/8 tasks
  (ties on 1, loses on 2 — `causal_judgement` and `formal_fallacies`).

**Does not hold up:**
- **Point 3 (discriminates real steering from gibberish) fails macro-averaged.**
  `t2l_gibberish_desc` (0.378) > `t2l_train_desc` (0.355). Task-by-task, `t2l_train_desc` beats
  gibberish on exactly 4/8 tasks (`date_understanding`, `dyck_languages`, `geometric_shapes`,
  `movie_recommendation`) and loses on the other 4 (`aqua`, `causal_judgement`,
  `formal_fallacies`, `logical_deduction_seven_objects`) — a coin flip, not a discrimination.
  `formal_fallacies` is the starkest loss: 0.210 vs. 0.430.
- Only 3/8 tasks (`dyck_languages`, `geometric_shapes`, `movie_recommendation`) show
  `t2l_train_desc` strictly ahead of *all four* other conditions at once, which is the bar §12's
  "real task-specific steering signal" claim implicitly needs.
- **Point 5 (beats the per-task oracle) also reverses macro-averaged, though the picture depends
  on which comparison you trust.** §12's headline `t2l_train_desc / oracle` = 2.02x was computed
  the same "macro-average of per-task ratios" way as the 1.100 figure above, so on that exact
  metric the direction technically still favors `t2l_train_desc` — but the 1.100 is now almost
  entirely carried by `dyck_languages`' 2.0x per-task ratio (0.100 vs. 0.050, i.e. 10 vs. 5
  correct out of 100 — a real but small absolute gap that the ratio formula inflates). Drop that
  one task and the remaining 6 scoreable tasks average **0.950x**, i.e. roughly a wash. The
  simpler macro-averaged-accuracy comparison agrees: `oracle` (0.366) now edges out
  `t2l_train_desc` (0.355). Per-task, `t2l_train_desc` beats `oracle` on 3/8 (`dyck_languages`,
  `geometric_shapes`, `logical_deduction_seven_objects`), loses on 4/8 (`aqua`,
  `date_understanding`, `formal_fallacies`, `movie_recommendation`), and ties on 1/8
  (`causal_judgement`, both 0.000). §12's "the LoRA roughly doubles the oracle's accuracy" claim
  does not survive contact with the full test sets — the honest read is that `t2l_train_desc` and
  a from-scratch single-task `oracle` LoRA land in roughly the same range here, with the oracle
  trained on far less data (its own tiny per-task pool) and no cross-task pretraining, which is
  itself a nontrivial result for the hypernetwork, just a much less flattering one than §12's
  ratio suggested.

**Revised bottom line:** steering still beats doing nothing and still (narrowly) beats prompting
the frozen model, macro-averaged over the full test sets — the method is not worthless. But the
strongest evidence in §12 (that the hypernetwork has learned genuine instruction-conditional
behavior rather than "always emit a generically-helpful LoRA") does not survive contact with
larger, less noisy test sets. §12's per-task numbers were built on samples as small as n=2
(`bbh_dyck_languages`); at that scale a single flipped answer is a 50-point swing, and several of
§12's most quoted results (aqua 0.875 vs 0.750, dyck_languages 1.0 vs 0.0) turned out to be
exactly that kind of artifact once re-measured on 100-254 rows.

### `oracle` on the full-test-set eval — implemented and run (2026-08-05)

`oracle` was initially excluded from this eval; that reasoning was about *expectations*, not a
technical blocker — `run_downstream_eval` already supported `oracle_dir` regardless of
`rows_for_task`, and canonicalized oracle adapters already exist at `outputs/oracle_loras_v2/`
for all 8 trained tasks. Wired in and run:
- `scripts/eval_downstream_accuracy_full.py` now takes `--oracle-dir` (default: omitted, same
  convention as `eval_downstream_accuracy.py` — pass it to opt in). When passed, `"oracle"` is
  appended to `CONDITIONS` and `oracle_dir` is threaded through to `run_downstream_eval`
  instead of the old hardcoded `oracle_dir=None`.
- `run_04b_downstream_eval_v2.sh --full` now runs **both** `eval_downstream_accuracy.py` (small
  Q-holdout) and `eval_downstream_accuracy_full.py` (full official test sets) back to back, both
  with `--oracle-dir "$ORACLE_DIR"` (default `outputs/oracle_loras_v2`) and
  `--gen-batch-size "$GEN_BATCH_SIZE"` (default 64, per the batch-size finding above) — so
  `bash run_all_v2.sh --full` produces both `outputs/eval/downstream_accuracy_v2.json` and
  `outputs/eval/downstream_accuracy_full_v2.json` end to end, with no separate manual step.
  `FORCE=1` re-runs both from scratch; `OUT`/`OUT_FULL` override the two output paths.
- **`TRAIN_TASKS` in `run_04b_downstream_eval_v2.sh` now defaults to the exact 8-task list**
  (`aqua` + the 7 `bbh_*` tasks with an `outputs/oracle_loras_v2/` adapter), not a
  `textgrad_repro_v2_*` glob. Caught the hard way while running this manually: `tasks-root` also
  holds ~12 more `bbh_*` task dirs exposed by the earlier `--min-samples 0` metadata rebuild
  (docs/03's 2026-08-04 changelog), none of which have an oracle adapter; a glob default would
  silently run real generation on all of them for every non-oracle condition too. Override via
  the `TRAIN_TASKS` env var (space-separated patterns) if a deliberately wider run is wanted —
  see the not-yet-executed 27-domain expansion this session had separately discussed and
  deferred.
- Results merged into the tables above. Headline: `t2l_train_desc / oracle` (macro-average of
  per-task ratios, §12's exact methodology) = **1.100**, but macro-averaged accuracy now has
  `oracle` (0.366) slightly *ahead* of `t2l_train_desc` (0.355) — another case, like point 3,
  where §12's small-sample headline number doesn't survive the full test sets. See the point 5
  writeup above for the per-task breakdown and why the two comparison methods disagree
  (`dyck_languages`' 2.0x ratio, from a genuine-but-small absolute 10-vs-5-correct gap on n=100,
  single-handedly moves the ratio average from ~0.95 to 1.10).

---

## 14. v3 dataset run (2026-08-11) — recon warm-start collapses; from-scratch SFT is now the arm that steers

> ⚠️ **This reverses §12 point 4 / docs/03's v1-v2 conclusion.** Those were built on v1/v2 data,
> where recon warm-start was clearly the working arm (steering margin 6-50x larger than
> from-scratch). On v3 data (`docs/03`'s new "v3 dataset run" section has the full loss-side
> diagnosis: recon flatlines at a mean-baseline fit for its entire 2000-step run), recon
> warm-start collapses to a near-description-independent LoRA instead, and **from-scratch SFT is
> now the arm that shows real, growing steering signal**. `run_04c_downstream_eval_v3.sh` still
> defaults `HYPERNET_CKPT` to `sft_warmstart_v3` — until the recon stage is fixed (more genuine
> task diversity + per-task description-paraphrase augmentation, see docs/03), **trust
> `sft_scratch_v3`, not `sft_warmstart_v3`**.

Same eval scripts, same 45-task winning-instruction scope (`data/best_prompt_tasks_v3.txt`, 38
after excluding T-holdout tasks — see `run_04c_downstream_eval_v3.sh`'s header), run twice: once
against `outputs/checkpoints/sft_warmstart_v3/latest.pt` (the default) and once against
`outputs/checkpoints/sft_scratch_v3/latest.pt`, via:

```bash
HYPERNET_CKPT=outputs/checkpoints/sft_scratch_v3/latest.pt \
OUT=outputs/eval/downstream_accuracy_scratch_v3.json \
OUT_FULL=outputs/eval/downstream_accuracy_full_scratch_v3.json \
bash run_04c_downstream_eval_v3.sh --full
```

### Result

| condition | small-holdout / warmstart | small-holdout / scratch | full-test-set / warmstart | full-test-set / scratch |
|---|---|---|---|---|
| `base` | 0.328 | 0.328 | 0.309 | 0.309 |
| `prompted` | 0.440 | 0.440 | 0.416 | 0.416 |
| `oracle` | 0.593 | 0.593 | 0.579 | 0.579 |
| `t2l_train_desc` | 0.471 | **0.543** | 0.387 | **0.484** |
| `t2l_other_task_desc` | 0.465 | 0.435 | 0.386 | 0.414 |
| `t2l_gibberish_desc` | 0.454 | 0.458 | 0.400 | 0.389 |

(`base`/`prompted`/`oracle` are identical warm vs. scratch as expected — those three conditions
never touch the hypernet checkpoint at all; a useful sanity check that only the `t2l_*`
conditions, the ones that actually depend on the checkpoint, differ.)

| comparison (macro) | small/warmstart | small/scratch | full/warmstart | full/scratch |
|---|---|---|---|---|
| `t2l_train_desc − prompted` | +0.031 | **+0.103** | −0.029 | **+0.068** |
| `t2l_train_desc − t2l_other_task_desc` | +0.006 | **+0.108** | +0.001 | **+0.070** |
| `t2l_train_desc − t2l_gibberish_desc` | +0.017 | **+0.085** | −0.013 | **+0.095** |
| `t2l_train_desc / oracle` | 0.89x | 1.07x | 0.69x | 0.91x |

Warmstart is essentially flat across all three discrimination deltas (0.001-0.031, and *negative*
on two of them on the full test set) — no real separation between the correct description, a
wrong task's description, or gibberish. Scratch shows a consistent +0.07 to +0.11 gap on every
comparison, in both eval scopes — the same pattern the loss-side `steering_margin` predicted
(docs/03's new v3 section), now confirmed against real held-out generation accuracy rather than
an internal training metric.

### Takeaways

- **Recon warm-start is actively harmful on v3 data, not merely unhelpful.** It leaves the
  hypernetwork in a state 2000 SFT steps can't escape, suppressing steering behavior the same
  architecture clearly can learn from scratch.
- **From-scratch SFT works, modestly.** `t2l_train_desc` beats `prompted` by a real, positive
  margin in both eval scopes now, and recovers 91-107% of the gap to `oracle` — a coherent result,
  unlike warmstart's near-random pattern.
- **Still real headroom.** Even scratch's `t2l_train_desc` (0.543/0.484) is competitive with, not
  clearly ahead of, `oracle` (0.593/0.579), and the correct-vs-wrong-description gap (+0.07-0.11)
  is real but modest — consistent with docs/03's diagnosis that 27 genuinely distinct tasks with
  zero within-task paraphrase diversity is still a much thinner training signal than reference
  T2L's ~479 tasks × 128 paraphrases (docs/03's v3 section has the full comparison). From-scratch
  SFT partially routes around the broken recon stage; the underlying data-scale gap has not gone
  away.
- **Practical recommendation**: report/trust `sft_scratch_v3` for v3 results until recon is
  fixed. Consider changing `run_04c_downstream_eval_v3.sh`'s `HYPERNET_CKPT` default, or fixing
  recon's data starvation (more real task diversity + per-task description-paraphrase
  augmentation, more training budget), before relying on `sft_warmstart_v3` again.
