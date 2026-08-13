# 05 — Comprehensive-feedback T2L input (v4 experiment)

**Status: IMPLEMENTED, not yet run.** New scripts pass `ruff`/`pytest` on CPU; no GPU run has been
done yet (feedback generation needs Qwen3-14B via vLLM, and full training/eval needs the B200
node, same as every other `--full` stage in this repo).

## Motivation

The v3 pipeline (`run_all_v3.sh`) conditions T2L on the literal TextGrad-optimized **prompt text**
as its input description, with the LoRA target trained on the `(question, response)` pairs that
prompt produced. Each round's prompt is itself derived from the previous prompt plus that round's
3 textual gradients, via one LLM call inside TextGrad's own `optimizer.step()`.

This experiment asks whether T2L conditions better on a **comprehensive, generalized feedback**
string instead -- built with the same "previous state + this round's 3 textual gradients -> new
state" recipe, but explicitly written as reusable guidance/critique for solving this kind of
problem, not as an instruction/prompt to a model. The LoRA target is unchanged (still
`question -> response`); only the T2L input text differs. GEPA is out of scope: it doesn't produce
the "previous X + 3 textual feedback -> new X" structure this recipe depends on (its mechanism is
reflective mutation over multiple candidates, not a single accumulating textual-gradient chain).

## Constraints this design satisfies

- **v4, fully namespaced.** Every new file/output path is suffixed `_v4`
  (`comprehensive_feedback_v4_*` task dirs, `data/splits_v4.json`, `outputs/*_v4`,
  `outputs/checkpoints/*_v4`, `outputs/eval/*_v4.json`, `configs/data_v4.yaml`,
  `data/textgrad_repro_comprehensive_feedback_v4/`), matching the repo's existing `_v2`/`_v3`
  convention.
- **v3 is untouched.** `data/textgrad_repro/` is read-only to every script in this experiment
  (only `iterations.jsonl` and `forward_outputs.jsonl` are ever opened, never written); every v4
  output lives at a path disjoint from anything v3 reads or writes.
- **Additive-only derived JSON.** `comprehensive_feedback_v4.jsonl` (written by
  `scripts/generate_comprehensive_feedback_v4.py`) copies every key from its source
  `iterations.jsonl` row forward unchanged (`iteration`, `prompt`, `val_accuracy`, `n_correct`,
  `n_total`, `reverted`, `n_sampled_for_gradient`, `textual_gradients`, `updated_prompt`) and adds
  exactly one new key, `comprehensive_feedback`. No key is ever renamed, dropped, or overwritten.

## The comprehensive-feedback chain algorithm

Source of truth is each run's `iterations.jsonl`
(`data/textgrad_repro/qwen-qwen3-14b_<task>_textgrad-repro/iterations.jsonl`). Every row already
has `prompt` (the prompt in effect before this round), `textual_gradients` (the 3 critiques
computed against it), `updated_prompt` (the prompt TextGrad settled on after this round --
post-revert-decision, see `scripts/textgrad_repro.py`), and `reverted` (true if the proposed edit
scored worse on val and TextGrad discarded it, keeping the prior prompt for the next round).

Walk the rows in order, maintaining `cf_by_prompt: dict[prompt text -> feedback text so far]`,
keyed by literal prompt text (not iteration index) so that if a later round reverts back to a
prompt this chain has already reached, it resumes from the right accumulated state rather than
restarting:

1. Seed `cf_by_prompt[iterations[0]["prompt"]] = ""` — the baseline prompt has no feedback yet.
2. For each row, in order:
   - `cf_before = cf_by_prompt[row["prompt"]]`
   - **If not reverted:** call Qwen3-14B once to merge `cf_before` with this round's 3
     `textual_gradients` into a new generalized paragraph `cf_new`; store
     `cf_by_prompt[row["updated_prompt"]] = cf_new`; this row's assigned feedback is `cf_new`.
   - **If reverted:** no LLM call. This round's gradients are **not** folded into the chain — a
     proposal that scored worse doesn't get to extend the accumulated feedback. This row's
     assigned feedback is simply `cf_before`, unchanged. (`row["prompt"]` is also the next row's
     `prompt`, since TextGrad reverted the live value, so the next lookup resolves correctly.)

This decision — reverted rounds don't advance the feedback state — was made explicitly rather than
accumulating every round's gradients regardless of outcome, on the reasoning that a round's
gradients describing "what's wrong" are entangled with the specific (ultimately-rejected) rewrite
attempt, and the accepted-prompt lineage is the more defensible thing to track feedback against.

**Efficiency:** the ~30 source-dir chains are independent of each other, so instead of walking
them one at a time (up to ~360 sequential LLM calls total), `generate_comprehensive_feedback_v4.py`
processes all chains **breadth-first by depth**: at depth `d`, it batches one merge prompt per
chain that has a non-reverted round at that depth into a single `llm.generate()` call, then
advances. This is ~1 batched call per depth (bounded by the longest run's iteration count, ~12),
not per row.

## Merge prompt (Qwen3-14B, thinking off)

Adapted from `guide_rest/feedback.py`'s two-stage merge pattern — here the 3 textual gradients
already play the role of that script's Stage-1 per-example critiques, so no separate Stage-1 call
is needed, only the Stage-2-style merge:

```
You are maintaining a running, generalized set of guidance notes for how to correctly solve
problems of this kind. The notes must generalize beyond any single question -- do not reference
specific numbers, names, or exact wording from the examples below; state the underlying principle
or strategy instead.

Previous guidance notes: {previous}

New feedback from this round (per-example critiques of what went wrong and how to fix it):
{gradients}

Merge the previous guidance notes (if any) with the new feedback into a single, self-contained,
generalized paragraph of guidance for solving problems of this kind. Remove redundant points.
Resolve contradictions in favor of the more recent feedback. Do not write a system prompt or
instruction to a model -- write guidance/feedback notes only. Keep it concise: no more than
{max_words} words. Output only the merged paragraph, nothing else.
```

`enable_thinking=False` for these calls (text analysis over feedback, not the hard task itself —
same rationale `guide_rest/feedback.py` and `gepa_repro_common.VLLMLanguageModel` give for their
own reflection/critique roles).

## How T2L task dirs get built from the chain

`scripts/build_tasks_from_comprehensive_feedback_v4.py` is a structural mirror of
`scripts/build_tasks_from_textgrad_repro_v3.py`. The only difference: instead of grouping
`forward_outputs.jsonl`'s `split == "val"` rows by their own literal `prompt` field, it joins each
row's `iteration` field against `comprehensive_feedback_v4.jsonl`'s `iteration ->
comprehensive_feedback` mapping and groups by *that* text (first-appearance order, `_d<K>`
suffix). Everything else — the correctness filter, `<think>` drop, `(question, response)` dedup,
`--min-samples` drop-group threshold, `metadata.yaml` shape — is identical to the v3 builder.

Because a reverted round's assigned feedback is byte-identical to its parent's (see above), those
rounds' rows land in the *same* group as the parent's own rows — the feedback-side analogue of how
the v3 builder pools rows that reused an identical prompt text after a revert. Groups whose
feedback text is empty (only possible if iteration 0 itself was reverted, so no real feedback ever
accumulated) are dropped — an empty description carries no signal worth training a LoRA against.

Output task dirs: `<tasks-out>/comprehensive_feedback_v4_<task>_d<K>/metadata.yaml`, disjoint from
`textgrad_repro_v3_*`/`gepa_repro_v3_*`.

## Files

| File | Role |
|---|---|
| `scripts/generate_comprehensive_feedback_v4.py` | Stage A (GPU/vLLM, Qwen3-14B). Reads `iterations.jsonl`, writes `<out-root>/<task>/comprehensive_feedback_v4.jsonl`. |
| `scripts/build_tasks_from_comprehensive_feedback_v4.py` | Stage B (CPU). Reads `forward_outputs.jsonl` + Stage A output, writes `comprehensive_feedback_v4_*` task dirs. |
| `configs/data_v4.yaml` | `DataConfig` pointed at `comprehensive_feedback_v4_*`, own `cache_root: data/.cache_v4`. |
| `run_03_training_validation_v4.sh` | Stage A + B, then splits/oracle/canonicalize/recon/SFT(x2)/ablation — mirrors `run_03c_training_validation_v3.sh`. |
| `run_04_downstream_eval_v4.sh` | Downstream accuracy eval against the v4 checkpoint — mirrors `run_04c_downstream_eval_v3.sh`, but both the Q-holdout and full-test-set evals default to *every* successful `comprehensive_feedback_v4_*` group (no "winning instruction" selector exists for the feedback axis — see the script's header for why). |
| `run_all_v4.sh` | Top-level wrapper: env -> phase 3 -> phase 4, the one script to run end-to-end. |

## How to run

```bash
bash run_all_v4.sh          # lint + tests only, CPU-safe
bash run_all_v4.sh --full   # the real end-to-end run -- B200 node only, hours-long
```

`--full` is resumable: `generate_comprehensive_feedback_v4.py` skips any source dir whose
`comprehensive_feedback_v4.jsonl` already exists; every downstream stage inherits the same
skip-if-already-built convention as the v3 pipeline.

## Design notes surfaced while running the first real `--full` pass

**The target model never sees the description text — only the bare question, for both v3 and
v4.** `src/steerable_t2l/data/formatting.py::format_example` is what actually builds the text fed
to the target model, for oracle training, recon, SFT, and every downstream-eval condition except
one (see below): `user_content = metadata.user_prompt_template.format(**row)` -- literally just
`"{question}"` -- wrapped in a user-only chat turn (`system_message` is always `""`, and an empty
system message is deliberately not the same as no system role at all -- see that function's
docstring). The task's `descriptions` entry (the optimized prompt in v3, the comprehensive
feedback in v4) is consumed in exactly one place: `hypernet.encode(descs)`, to condition which
LoRA weights get generated. It never gets concatenated into the target model's own context. So
switching from "prompt" to "comprehensive feedback" changes what steers the LoRA's *weights*; it
changes nothing about what the target model *reads* at generation time -- that was already
description-free before this experiment existed.

One deliberate, eval-only exception: `eval_accuracy.py`'s `prompted` condition (no LoRA at all)
does inject the description as a literal system turn (`build_prompted_prompt`), purely as an
ablation baseline -- "how well does just telling the model the instruction/feedback in-context
do, with zero LoRA steering?" This is condition-gated and separate from `t2l_train_desc` (the
actual LoRA-steered condition), which still goes through the same bare-question
`format_example` path as training does.

**So how does the model learn the required answer format (e.g. `"Answer: $VALUE"`, MCQ letter,
Yes/No) if it's never told at inference time?** Purely by imitation. Every `response` in the
`(question, response)` training pairs already ends in the task's correct format, because it was
originally generated by Qwen3-14B under the *full* TextGrad prompt (which does spell the format
out) during the original `textgrad_repro.py` run -- confirmed on `multiarith` training rows, which
end in `"...Answer: $15"` / `"...Answer: $21"` verbatim. SFT trains the LoRA with next-token loss
directly against that exact completion text, trailer included, so the model is fine-tuned to
always produce the right format rather than told to. This mechanism is identical for v3 and v4 --
it does not depend on whether the hypernetwork's input was a prompt or comprehensive feedback --
so it is not a v4-specific gap. The one difference: the literal optimized prompt usually restates
the format requirement explicitly (extra signal for the hypernetwork to condition on, even though
the target model itself never reads it), while comprehensive feedback mostly doesn't, since it's
deliberately framed as generalized guidance rather than restated instructions (spot-checked: 28/28
task chains still incidentally mention "format"/"Answer:" somewhere, since the underlying textual
gradients themselves complain about formatting when the model gets it wrong -- so the signal isn't
absent, just less consistent than in the raw prompt).

**Decision: leave this as-is, don't inject any format hint back into the target model's context.**
Reasoning: (1) if the hypernetwork fails to learn task-appropriate formatting from comprehensive
feedback as reliably as it does from the literal prompt, that is exactly the kind of thing
`t2l_train_desc` downstream accuracy is supposed to measure -- a real, informative result to
compare against v3 rather than something to preempt; (2) injecting format text back into the
target's context (even as a small "fixed, non-steered" addition) would blur the very comparison
this experiment exists to make -- whether feedback-conditioned *weight* steering works as well as
prompt-conditioned steering -- and would need to apply symmetrically to v3 for the comparison to
stay fair, which is out of this experiment's scope. If `outputs/eval/downstream_accuracy_v4.json`
comes back with a lot of unparseable-answer failures specifically (as opposed to wrong-but-
parseable answers), that's the signal to revisit this.

**`data/textgrad_repro/qwen-qwen3-14b_aime_textgrad-repro/` was deleted** (2026-08-09) -- it was
the crashed, 2-iteration-only run documented in `textgrad_repro_README.md`'s known-issues list
("`aime` -- crashed (`optimizer.step()`'s rewrite prompt exceeded `max_model_len`...); data
deleted"). Confirmed safe to delete: neither `textgrad_repro_v3_aime_*` nor
`comprehensive_feedback_v4_aime_*` task dirs ever existed in the first place -- both of aime's
2-iteration groups (9 correct rows each) were already silently dropped by every run's
`--min-samples 50` filter, so nothing trained, split, or checkpointed in v3 or v4 depended on it.
`data/gepa_repro/qwen-qwen3-14b_aime_gepa-repro/` is untouched (separate directory, separate
algorithm, not affected).

**v4 has no "no feedback yet" baseline group, unlike v3's baseline-prompt group -- intentional.**
TextGrad logs a baseline eval at `iteration == -1` (the pristine, unedited task description, zero
gradient influence) as well as its own per-round `iteration >= 0` evals. v3's builder groups by
literal prompt text with no iteration filter, so `iteration == -1`'s rows form their own group --
confirmed on `multiarith`, `textgrad_repro_v3_multiarith_d0`'s description is the literal
untouched baseline prompt, giving v3 13 `multiarith` variants where v4 has 12. v4's builder
explicitly skips any row whose `iteration` has no entry in `comprehensive_feedback_v4.jsonl` (`if
not feedback: continue`), and the chain starts accumulating at `iteration == 0`, so `-1` never
gets a feedback entry -- there is no "no feedback given yet" analogue of v3's raw-baseline group.
Confirmed as intended, not a bug to fix.

Also worth noting: `forward_outputs.jsonl`'s val rows at iteration `i` are generated using
`updated_prompt` (the prompt *after* round `i`'s edit, logged before the revert decision) --
train-split rows use `prompt_before` instead, the state *entering* round `i`. So "round 0"'s
(question, response) pairs (which v4 pairs with round 0's freshly-merged comprehensive feedback)
were never generated from the pristine baseline prompt either -- they already reflect round 0's
own gradient-informed edit, the same round whose gradients produced the paired feedback text. The
pairing is causally consistent (same round's gradients drive both the responses and the feedback
text assigned to them); the genuinely gradient-free baseline responses only exist at `iteration ==
-1`, which is the group v4 excludes per the point above.

## Verification plan

- CPU unit test for `build_tasks_from_comprehensive_feedback_v4.py`'s grouping/dedup/empty-group-
  drop logic against a small hand-built fixture (`tests/test_build_tasks_comprehensive_feedback_v4.py`).
- Smoke test `generate_comprehensive_feedback_v4.py` against one real source dir: output row count
  equals input `iterations.jsonl` row count, iteration indices match, every original key is
  present unchanged, and a reverted row's `comprehensive_feedback` exactly equals its parent's.
- After a real `--full` run, diff `outputs/eval/downstream_accuracy_v4.json` /
  `..._full_v4.json` against the existing v3 eval outputs to see whether feedback-as-input changes
  downstream accuracy vs. prompt-as-input, and confirm `data/textgrad_repro/`, `outputs/*_v3*`,
  and `data/splits_v3.json` are byte-identical before/after (v3 untouched).

## Results: first real `--full` run (2026-08-13, B200)

`bash run_all_v4.sh --full` completed end-to-end with no errors (lint, full pytest suite, then the
whole GPU pipeline: feedback generation -> task build -> oracle LoRAs -> recon warm-start -> SFT
scratch -> downstream eval) across all 86 `comprehensive_feedback_v4_*` task groups. Outputs:
`outputs/eval/downstream_accuracy_scratch_v4.json` (Q-holdout) and
`outputs/eval/downstream_accuracy_full_scratch_v4.json` (full official test sets).

Macro-averaged accuracy, compared against the equivalent v3 (scratch) eval outputs:

| condition | v4 Q-holdout | v4 full test | v3 Q-holdout (scratch) | v3 full (scratch) |
|---|---|---|---|---|
| base | 0.349 | 0.342 | 0.328 | 0.309 |
| prompted | 0.333 | 0.321 | 0.440 | 0.416 |
| t2l_train_desc | **0.579** | **0.551** | 0.543 | 0.484 |
| t2l_other_task_desc | 0.422 | 0.415 | 0.435 | 0.414 |
| t2l_gibberish_desc | 0.502 | 0.477 | 0.458 | 0.389 |
| oracle | 0.599 | 0.564 | 0.593 | 0.579 |

Takeaways:

- **`t2l_train_desc` improves over v3 on both eval splits** (0.579 vs. 0.543 Q-holdout; 0.551 vs.
  0.484 full test) and closes most of the gap to `oracle` (~97% of oracle in v4 vs. ~92% in v3) --
  conditioning the hypernetwork on comprehensive feedback is at least as good a T2L input as the
  literal optimized prompt, and slightly better here.
- **`prompted` collapses in v4** (0.333/0.321, below even `base`), as anticipated in the "Design
  notes" section above: comprehensive feedback is generalized guidance text, not an instruction,
  and it carries no explicit answer-format information the way v3's optimized prompt does -- so
  injecting it verbatim as a system-turn instruction (no LoRA) actively hurts rather than helps.
  This is consistent with, and further confirms, the earlier recommendation to leave the
  no-base-prompt design as-is: the format-learning mechanism (SFT imitation on `response` text) is
  doing the real work, not the description text at inference time.
- **`t2l_gibberish_desc` (control) is somewhat higher in v4 than v3** (0.502 vs. 0.458
  Q-holdout), narrowing the train-vs-gibberish margin a bit, though `t2l_train_desc` still clears
  it by a comfortable margin on both splits.
- **`oracle` is essentially unchanged** between v3 and v4 (~0.59-0.60 Q-holdout, ~0.56-0.58 full),
  as expected: oracle LoRAs are trained per-group directly on the same `(question, response)`
  pairs regardless of which description text (prompt vs. feedback) is attached, so the LoRA target
  never changed between v3 and v4.
