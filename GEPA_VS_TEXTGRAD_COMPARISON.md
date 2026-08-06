# GEPA vs. TextGrad: how each generates and validates textual feedback

Comparison of the actual algorithms as run by `scripts/textgrad_repro.py` (using the real
`textgrad` package, vendored at `textgrad_repro/`) and `scripts/gepa_repro.py` (using the real
`gepa` package, vendored at `gepa_repro/`). Both scripts share the same task registry
(`textgrad_repro.TASKS`), the same seed prompt, the same 50/100/N train/val/test splits, and the
same `batch_size=3` minibatch size, so the comparison below is apples-to-apples — the differences
are in the optimizers themselves, not the setup.

## 1. Overall shape

| | TextGrad | GEPA |
|---|---|---|
| Candidate structure | One linear chain — a single running prompt that gets overwritten each iteration | A branching tree/pool of candidates — parents are re-selected via Pareto sampling, mutations fan out |
| Parent selection | Always the current prompt (whatever it was last set to) | `ParetoCandidateSelector`: samples a parent from the whole accepted-candidate pool, weighted toward candidates that are best on at least one val example |
| Termination | Fixed iteration count: `max_epochs * steps_per_epoch` (default 3×4=12) | Rollout-budget exhaustion: `max_metric_calls` (default 3936, chosen to match TextGrad's total call count for comparability) — not a fixed number of accepted candidates |

## 2. Minibatch sampling

Both use `batch_size=3` by default, and both use an **epoch-shuffled, without-replacement**
sampler (not i.i.d. random sampling each step):

- **TextGrad**: `tg.tasks.DataLoader(train_set, batch_size=3, shuffle=True)`. Reshuffles only when
  an epoch is exhausted.
- **GEPA**: `EpochShuffledBatchSampler`, same semantics (shuffle per epoch, pad to a multiple of
  the minibatch size).

The structural difference is *what* gets a minibatch each iteration: TextGrad always evaluates the
single running prompt on the next minibatch. GEPA evaluates whatever parent
`ParetoCandidateSelector` picked for that iteration on the next minibatch — so consecutive
iterations in GEPA can be mutating completely different ancestors, not the same lineage.

## 3. How textual feedback is generated

This is the biggest divergence between the two, and it's not just labeling — the feedback going
into the rewrite call is qualitatively different in origin.

**TextGrad**: feedback is LLM-generated critique text, produced by TextGrad's own autograd
backward pass. For each of the 3 minibatch examples:
1. `StringBasedFunction.backward()` — one `backward_engine(...)` call — critiques the *response*
   given the eval outcome.
2. `LLMCall.backward()` — a second `backward_engine(...)` call — propagates that into a critique of
   the *system prompt* given the question and the response-level critique.

So there are 6 backward LLM calls per iteration under the hood, but only the 3 prompt-level ones
(one per example) are what's logged as `system_prompt.gradients` / `gradients.jsonl`, and only
those 3 feed into the rewrite step.

**GEPA (as implemented in `gepa_repro.py`'s adapter)**: feedback is a **deterministic template
string**, not LLM-generated:
- `"Correct. The gold answer is 'X'."`
- `"Incorrect. The gold answer is 'X'. Explain what went wrong and how the approach should change."`
- a parse-failure variant when no answer could be extracted

No LLM call is spent writing this feedback. All the actual reasoning about *why* a response was
wrong and *how* to fix the instruction is deferred entirely to the single reflection call in the
next stage, which sees the raw (question, response, predicted, gold, feedback-tag) records
directly. This is a choice made in this repro's adapter, not a constraint of the GEPA library
itself — GEPA supports LLM-authored per-example feedback via a custom adapter; this
reproduction opted for the cheap-template path, presumably to save reflection-LM budget.

## 4. How the new prompt (rewrite) is generated

Structurally near-identical shape — "N examples → 1 rewritten instruction" — but different inputs:

- **TextGrad**: `optimizer.step()` — one `TextualGradientDescent` call. Concatenates the current
  prompt + all 3 prompt-level gradients (pre-digested critiques) into a single rewrite prompt →
  one new candidate prompt.
- **GEPA**: one `InstructionProposalSignature` reflection call (`temperature=0.7`). Concatenates
  the parent instruction + a markdown dump of the 3 raw (question, response, predicted, gold,
  feedback-tag) records → one new candidate instruction. The reflection LM does the "what should
  change" reasoning itself, working from raw transcripts rather than pre-digested critiques.

## 5. Validation and acceptance

This is the second major divergence, and it drives how many val-verified prompts each method ends
up producing:

**TextGrad**: every proposed prompt is validated on the **full 100-example val set**, every
single iteration, regardless of how promising the minibatch looked (there is no minibatch-based
pre-filter). If `val_accuracy` doesn't improve (or ties), `run_validation` reverts the prompt to
its pre-iteration value — a strict, monotonic, single-chain accept/reject. 100 val-eval calls are
spent every iteration whether or not the new prompt is kept.

**GEPA**: acceptance is two-staged and cheap-first:
1. The child is first re-scored on the **same 3-example minibatch** it was written from.
2. `StrictImprovementAcceptance` (default) only lets it through if its minibatch score sum
   **strictly beats** the parent's minibatch score sum on that same minibatch.
3. Only if that passes does GEPA spend a full 100-example val pass and add the candidate to the
   pool (`_run_full_eval_and_add` → `_add_evaluated_program`).

A candidate that fails the 3-example check is discarded outright — no val pass is ever spent on
it, and the parent is untouched (nothing is "reverted"; it just isn't selected as a future parent
as often). This means most proposals GEPA generates never reach a full-val evaluation at all,
unlike TextGrad where every proposal costs a full val pass.

A candidate, once accepted and given its one full val pass, is **never re-evaluated on val
again** — there is no dedup-by-content or cache-reuse step in `_add_evaluated_program` that would
cause the same tree node to be scored twice.

## 6. Distinct prompt count, in practice

- **TextGrad**: at most `max_epochs * steps_per_epoch + 1` prompts (13 by default: baseline +
  12 iterations), fewer if any iteration reverts to an already-seen prompt. A predictable upper
  bound.
- **GEPA**: not a fixed count. Depends entirely on the acceptance rate at the cheap
  minibatch-screening stage — could be well above or below 13 for the same total rollout budget,
  because most proposed candidates never clear the 3-example bar and therefore never become a
  pool member at all.

## 7. Summary table

| Aspect | TextGrad | GEPA |
|---|---|---|
| Parent per iteration | Always the current single prompt | Pareto-sampled from the whole accepted pool |
| Minibatch sampler | Epoch-shuffled `DataLoader`, size 3 | Epoch-shuffled `BatchSampler`, size 3 |
| Feedback source | LLM-generated critique (2 backward calls/example → 3 prompt-level gradients/iteration) | Deterministic template string (correct/incorrect + gold answer), no LLM call |
| Rewrite call | 1 call, combines prompt + 3 gradients | 1 call, combines prompt + 3 raw QA+feedback records |
| Validation cost per proposal | Always: full 100-example val pass | Only if minibatch check passes: full 100-example val pass |
| Rejection mechanism | Revert to prior prompt if val accuracy doesn't improve | Discard the child before any val pass if minibatch score doesn't strictly improve |
| Re-evaluation of an existing candidate | N/A (single chain, prompt is overwritten) | Never — each accepted node gets exactly one val pass |
| Termination | Fixed iteration count (12 default) | Rollout-budget exhaustion (3936 default) |
| Distinct-prompt count | Predictable upper bound (≤13 default) | Budget- and acceptance-rate-dependent, not fixed |

---

## v3 training-dataset construction (`build_tasks_from_*_repro_v3.py`)

Both v3 builders emit one task dir (one oracle LoRA) per **distinct instruction text**, each
paired only with rows drawn from the **100-example val set** — no 3-example minibatch rows leak
into the training data for either method.

- **TextGrad** (`build_tasks_from_textgrad_repro_v3.py`): groups `forward_outputs.jsonl` rows by
  their literal `prompt` text, keeping only rows with `split == "val"`. Train-minibatch rows
  (`split == "train"`) are excluded by construction. Each group has **up to 100 rows** (one val
  pass per accepted prompt).
- **GEPA** (`build_tasks_from_gepa_repro_v3.py`): GEPA's `forward_outputs.jsonl` has no `split`
  field (minibatch and full-val calls share the same log shape), so this builder cross-references
  `row["question"]` against `val_set.jsonl`'s question set to identify val rows, then groups by
  literal `candidate` text. Each group normally has **up to 100 rows**, one val pass per accepted
  candidate.
- Edge case (rare, not a designed feature): if two independently-accepted GEPA tree nodes happen
  to produce byte-identical instruction text (reflection at `temperature=0.7` converging twice on
  the same wording), each node still gets its own separate ≤100-row val pass, and the v3 builder's
  group-by-text logic merges them — so a GEPA instruction group can occasionally exceed 100 rows.
  This is a coincidence of content collision, not re-evaluation of the same tree node.
- Both builders keep only `correct == True` rows by default (`--filter-correct`, since these rows
  double as SFT targets), drop any response still containing `<think>`, dedupe by
  `(question, response)` pair, and drop an instruction group entirely if fewer than
  `--min-samples` (default 50) rows survive.
- A GEPA candidate that failed its cheap 3-example acceptance check never appears in the v3 data
  at all — it has zero rows with a val-set question, since it was discarded before ever reaching a
  val pass.
