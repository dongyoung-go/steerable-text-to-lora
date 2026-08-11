# Guide-ReST: Feedback-Guided Reinforced Self-Training

## Motivation

Reinforced Self-Training (ReST) improves a model through a two-step loop — **Grow**
(sample many candidates per question) and **Improve** (filter the correct ones, fine-tune
on them, repeat). Grow's proposal mechanism is blind: it relies on temperature or
best-of-N sampling to occasionally stumble onto a correct completion by chance, and the
filter simply discards the rest. Nothing in the loop tells the model *why* an attempt
failed, so much of the sampling budget is spent on retries that fail for the same reason
as before.

This experiment tests a minimal, targeted addition: after each round, distill the
model's own successes and failures into a single natural-language description of the
corrective policy — a **comprehensive textual feedback** — and use that description to
bias the *next* round's Grow step, instead of sampling blind. Everything else about ReST
(the filter, the fine-tuning step, the round structure) is left untouched.

The question this experiment answers: **does conditioning Grow on a self-generated
textual description of past mistakes produce a better self-training trajectory than
unconditioned resampling, at matched compute?** This isolates the contribution of the
feedback mechanism itself, cleanly separated from the filter and fine-tuning steps that
ordinary ReST already relies on.

---

## Related Work

**ReST and its offline self-training lineage.** ReST is a growing-batch self-training
method: an initial policy generates samples, samples are filtered/ranked, and the model
is fine-tuned on the filtered set via offline RL/SFT, with the whole cycle repeated.
ReST-MCTS* extends this line by integrating process-reward-guided tree search to collect
higher-quality reasoning traces, addressing the observation that filtering purely on
final-answer correctness can retain flawed intermediate reasoning even when the final
answer happens to be right. Guide-ReST stays closer to vanilla ReST's simplicity
(no tree search, no process reward model) and instead targets the *proposal* step
directly with a textual signal.

**RLVR / RFT (GRPO, DAPO, VAPO).** The dominant current successor to offline batch
self-training is online reinforcement learning with verifiable, rule-based rewards —
replacing ReST's discrete Grow/Improve rounds with continuous policy-gradient updates.
Guide-ReST deliberately keeps ReST's discrete-round structure instead, since later use of
the resulting checkpoints requires clean, per-round checkpoints rather than a
continuously drifting policy.

**In-context self-correction (Reflexion, Self-Refine).** These methods have a model
critique its own output and revise it within the same context window or across a
conversational memory buffer, with no weight updates at all. Guide-ReST borrows the
"generate a critique, use it to steer the next attempt" mechanic from this line, but
relocates it: the critique steers *sampling for training data*, and its effect is meant
to persist through fine-tuning, not just through one in-context revision.

**Multi-turn self-improvement fine-tuning (RISE).** RISE fine-tunes a model to improve
its own responses across turns by training on self-generated correction trajectories,
treating a single-turn prompt as a multi-turn correction process. This is the closest
existing method to Guide-ReST's mechanism. The distinction: RISE folds the correction
into the rollout itself (a failed attempt plus feedback as context within one trajectory),
while Guide-ReST distills the correction into a separate, reusable, cumulative text
artifact that persists and gets explicitly merged across many independent rounds, rather
than living inside any single trajectory.

**Textual-gradient prompt optimization (TextGrad).** TextGrad formalizes the idea that an
LLM critique of outputs can function like a gradient with respect to a text variable.
Given a batch of (question, model output, ground truth) examples, TextGrad computes a
per-example textual gradient — a criticism of that specific output — and then performs a
"step": combining the batch of per-example gradients with the current prompt to produce
an updated prompt. Guide-ReST's feedback-generation procedure is directly modeled on this
two-stage structure (per-example critique, then a combining step), with two changes:
the target of the step is a standalone comprehensive feedback text rather than a system
prompt, and that target is never re-injected as a prompt for the base model to imitate —
it exists only to condition Grow's sampling and, later, to serve as a training label. See
**Method → Feedback Generation Procedure** below for the exact adaptation.

---

## Method

### Two conditions, identical except for one step

**Condition A — Vanilla ReST (control).** Grow samples with no feedback, at every round.

**Condition B — Guide-ReST (treatment).** Grow samples with the current round's
comprehensive feedback prepended to every prompt, starting from round 1 onward. The feedback prefix is used *only* at Grow time — it is stripped before the resulting (question, correct completion) pairs are used for fine-tuning, and it is never prepended at evaluation time. 

Everything else — filtering, fine-tuning, round count, sampling budget per question — is
identical between conditions, so any difference in outcome is attributable to the
feedback-conditioning step alone.

### Round structure

**Step 1 — Grow.**
For every question `q` in the task's validation pool, sample `k` completions from the
current model `M_t` at temperature `T`. In Condition B (from round 1 on), prepend the
current comprehensive feedback text to the prompt; in Condition A, sample unconditioned.

```
Condition A:  y ~ M_t( · | q )
Condition B:  y ~ M_t( · | feedback_t, q )     (t ≥ 1; round 0 is unconditioned in both)
```

Note: the feedback prefix shown to the model in this step is *not* retained as part of the training example in Step 4 — see below.

**Step 2 — Filter.**
Score every sampled completion with a verifier and keep only correct ones. Because small
and mid-sized models have real sample-to-sample variance, correctness is judged from
multiple draws per question, rather than a single sample — this avoids mistaking ordinary
sampling noise for a genuine feedback effect.

**Step 3 — Feedback generation (Condition B only).**
Described in detail below.

**Step 4 — Improve.**
Fine-tune `M_t` on the round's filtered (question, correct completion) pairs, producing
`M_{t+1}`. Identical procedure in both conditions. That is, in Condition B, the feedback prefix used during Grow is stripped from the input before fine-tuning — the model is trained to map the *bare* question to the correct completion, not the feedback-conditioned prompt to the completion. 

**Step 5 — Repeat** for `T` rounds.

### Feedback Generation Procedure (TextGrad-inspired)

This follows TextGrad's two-stage structure — per-example critique, then a combining
step — adapted to produce a standalone comprehensive feedback rather than an updated
prompt.

**Stage 1 — Per-example textual feedback.**
Sample `N` (question, incorrect completion, correct completion) triples from the round's
filtered results. For each triple independently, prompt the model to write a short
critique explaining what went wrong and how to fix it:

```
for i in 1..N:
    local_feedback_i = LLM(
        "Here is a question, an incorrect answer, and a correct answer.
         Explain what the incorrect answer got wrong and how to avoid
         this mistake."
        + triple_i
    )
```

`N` is a tunable hyperparameter, and does not need to match TextGrad's typical batch size.
A smaller `N` risks a critique that only reflects one or two idiosyncratic failures rather
than a generalizable pattern; a larger `N` gives more evidence per critique but produces
more text to fold into Stage 2, and produces diminishing returns once the failures start
repeating the same pattern. This should be tuned empirically per task rather than fixed
in advance — worth trying at least two values of `N` (e.g. 3 and 8) on one task before
committing to a default for the full run.

**Stage 2 — Merge into a comprehensive feedback.**
Combine the `N` local feedbacks from this round with the previous round's comprehensive
feedback into a single, self-contained paragraph, explicitly asking the model to resolve
redundancy and contradictions rather than concatenate:

```
feedback_t = LLM(
    "Previous guidance: " + feedback_{t-1} +
    "New critiques from this round: " + [local_feedback_1, ..., local_feedback_N] +
    "Merge these into a single, self-contained paragraph of guidance.
     Remove redundant points. Resolve any contradictions. Keep it concise."
)
```

Round 1's feedback_1 has no `feedback_0` to merge with, so it is built from Stage 1
outputs alone.

**Length control.** Because feedback_t is built by merging into the previous round's
already-merged text, its length can grow across rounds if the merge step is not
explicitly constrained. A comprehensive feedback that grows too long risks the model
failing to attend to or apply all of it correctly at Grow time — long, itemized
instruction lists are harder for a model to consistently follow than a short, focused
paragraph. The merge prompt should include an explicit length cap (e.g. "no more than
150 words") and should be checked periodically across rounds (e.g. plotting feedback_t's
token length across `t`) to confirm it is not drifting upward uncontrolled.

### Worked example

```
Round 0 (both conditions, unconditioned):
  Q: "A train travels 120 km in 2 hours. What is its average speed?"
  Sampled answer (incorrect): "The speed is 120 km/h."
  Sampled answer (correct):   "120 / 2 = 60. The average speed is 60 km/h."
  → correct answer kept, incorrect discarded.

Stage 1 (Condition B, N=3 triples from round 0):
  local_feedback_1: "The model stated a numeric answer without performing
  the division, effectively repeating the distance value as if it were
  the speed."
  local_feedback_2: "The model skipped writing out the formula (distance /
  time) before substituting numbers."
  local_feedback_3: "In another case, the model computed the right formula
  but transposed distance and time."

Stage 2 (merge, no previous feedback for round 1):
  feedback_1 = "When solving rate/speed problems, explicitly write the
  formula (e.g. speed = distance / time) before substituting values, and
  double check that each quantity is substituted into the correct position
  in the formula. Do not state a final numeric answer without showing the
  calculation that produced it."

Round 1 Grow:
  Condition A: M_1 samples "The speed is 65 km/h." (unconditioned, still guessing)
  Condition B: M_1, prompted with feedback_1, samples
    "speed = distance / time = 120 / 2 = 60 km/h."
    (more likely to pass the filter, since the feedback names the exact
    formula and substitution error this task exposed)
```

The comparison of interest is precisely this: does Condition B's Grow step produce a
higher proportion of filter-passing completions than Condition A's, at the same `k` and
`T` — and does that compound into a better final model after `T` rounds of fine-tuning.

---

## Experimental Design

### Datasets

Verifiable-answer domains are used throughout, to keep the filter step objective and
avoid reintroducing the calibration and reward-hacking issues of learned reward models.
Code domains (e.g. LiveCodeBench) are deferred for a later phase, so the initial scope
stays within single-answer-verifiable domains that are simpler to filter and to inspect
feedback quality against by hand.

- **GSM8K / MATH (MATH500)** — standard grade-school and competition math. Used as the
primary anchor domain: well understood, easy to verify, and easy to qualitatively
inspect feedback text against (as in the worked example above).
- **Guru (non-code domains)** — a multi-domain verifiable-reward dataset spanning Math,
Code, Science, Logic, Simulation, and Tabular tasks. For this experiment, only the
non-code domains are used (e.g. Science, Logic, Tabular) to add task diversity beyond
math without introducing code-domain complexity yet.

Recommended starting scope: GSM8K/MATH plus one or two non-code Guru domains (e.g. Logic
and Tabular), run independently as separate tasks.

### Setup

- **Base model:** Qwen3-14B.
- **Validation pool:** ~100 verifiable questions per task, used for both Grow sampling and
round-over-round tracking.
- **Sampling budget:** `k` completions per question per round (e.g. `k=8`), fixed
identically across both conditions.
- **Rounds:** `T` rounds per task (e.g. `T=4–6`), sufficient to see whether any gap
between conditions compounds or plateaus.
- **Feedback batch size `N`:** tuned per task as described above, starting from a small
sweep (e.g. `N=3` vs `N=8`) before fixing a default.

### Comparison conditions

1. **Vanilla ReST** — unconditioned Grow throughout. The control.
2. **Guide-ReST** — feedback-conditioned Grow from round 1 onward, as described above.

### Metrics

- **Per-round filter-pass rate** — fraction of Grow's `k` samples per question that pass
the verifier, tracked separately per round and condition. This is the most direct,
round-local test of whether feedback is doing anything at Grow time.
- **Held-out pass@1** — accuracy of `M_t` on a held-out question set from the *same* task
(not used in Grow/filtering), tracked round over round, evaluated with the question only (no feedback prefix, in either condition), to measure whether gains
compound through fine-tuning rather than just appearing in-sample.
- **Sample efficiency** — total samples needed to reach a fixed accuracy target, compared
between conditions, since Guide-ReST's core hypothesis is that guided proposals waste
fewer samples on repeat-failure modes.
- **Feedback length over rounds** — token length of feedback_t tracked across `t`, as a
direct check on the length-control concern raised above.

### Controls and diagnostics

- **Resampling for causal attribution.** Multiple independent draws per question, both
pre- and post-feedback within a round, so that a measured filter-pass-rate improvement
reflects the feedback's effect rather than ordinary sampling variance.
- **Feedback-text inspection.** Manual reading of feedback_t at each round to confirm it
remains specific and grounded in the round's actual failure modes rather than drifting
toward generic, uninformative phrasing as rounds accumulate.
- **Round-0 parity check.** Both conditions must start from an identical, unconditioned
round 0, so any divergence measured from round 1 onward is attributable solely to
feedback conditioning.

## TODO additions

**Failure-mode recurrence analysis:** Classify incorrect rollouts by failure mode (e.g. missing constraint, algebraic manipulation, etc) and measure whether Guide-ReST reduces recurrence of previously observed failure modes across rounds. Compare (P(f_{t+1}\mid f_t)) between Vanilla ReST and Guide-ReST to test whether feedback specifically suppresses repeated mistakes, providing mechanistic evidence beyond aggregate accuracy gains.

**Confound baseline — few-shot exemplar control:** Add a third condition: Grow conditioned on 2–3 raw correct completions from the previous round (no critique text, just exemplars), stripped before fine-tuning like Condition B's feedback. This isolates whether Guide-ReST's gain comes from the *diagnostic critique* or just from showing the model in-context demonstrations. Without this, a positive result for Condition B is confounded with a generic few-shot effect.

**Compute accounting — matched-compute baseline:** Stage 1 (N critiques) + Stage 2 (merge) add extra LLM calls per round in Condition B, and its Grow prompts run longer — so "same k, same T" is not "same compute." Add a Vanilla ReST run at higher k (e.g. k=16 vs. Guide-ReST's k=8) chosen so total sampling+critique tokens roughly match, and re-plot filter-pass-rate / pass@1 / sample-efficiency against that matched-compute control rather than only against equal-k Vanilla ReST.