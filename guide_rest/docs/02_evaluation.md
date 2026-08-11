# 02: Evaluation

Metrics and diagnostics from `guide_rest_README.md`'s "Metrics" and "Controls and
diagnostics" sections, computed from the artifacts `round_loop.py` (see `01_train.md`)
already writes — no separate eval pass is needed beyond what each round already produces.

## Metrics

All four live in `data/guide_rest/<task>/<condition>/summary.jsonl` (one row per round) or
are derived from it directly.

1. **Per-round filter-pass-rate.** `summary.jsonl`'s `filter_pass_rate` field
   (= `grow_stats.json`'s `n_correct / n_total` over that round's `k`-completions-per-question
   Grow batch). The most direct, round-local read on whether feedback is doing anything at
   Grow time. Plot per condition, per task, round on the x-axis — a Guide-ReST effect should
   show Condition B's curve above Condition A's from round 1 on (round 0 is unconditioned in
   both, so should match within noise — see "Round-0 parity check" below).

2. **Held-out pass@1.** `summary.jsonl`'s `heldout_pass_at_1` field
   (= `heldout_eval.json`'s `pass_at_1`, greedy decode, question only, no feedback prefix in
   either condition). Tracks whether a Grow-time filter-pass-rate gain actually compounds
   through the Improve step into a genuinely better model, versus only showing up in-sample
   at Grow time.

3. **Sample efficiency.** Not a stored field — computed post hoc from `summary.jsonl`: for a
   fixed `heldout_pass_at_1` target, find the smallest round `t` (per condition) at which
   that target is first reached, then convert to total samples drawn
   (`round * grow_pool_size * k`, plus Condition B's `N` extra critique/merge calls per
   round if comparing token/call budgets rather than raw sample count). Compare the two
   conditions' round-to-target directly.

4. **Feedback length over rounds.** `summary.jsonl`'s `feedback_word_count` field (Condition
   B only — word count of that round's `feedback.txt`, i.e. the text used to condition
   *next* round's Grow). Plot against round to check the README's length-control concern:
   flag if it's still increasing near-linearly by the last round rather than plateauing
   under the `--max_words` cap (a plateau near the cap is expected and fine; monotonic
   growth up against the cap every round suggests the Stage-2 merge prompt isn't actually
   compressing, just truncating).

## Diagnostics

- **Resampling for causal attribution.** Already satisfied structurally: `sampling.py`
  draws `k` completions per question in one `SamplingParams(n=k)` call, and
  `filter_pass_rate` is computed over all `k * grow_pool_size` draws, not a single sample
  per question — so a measured pass-rate delta between conditions reflects the feedback
  effect at the batch level rather than one lucky/unlucky draw. No extra code needed; just
  don't compute filter-pass-rate from a single completion when eyeballing results.

- **Feedback-text inspection.** Manual, not automated: read `feedback.txt` across a few
  rounds per task (`data/guide_rest/<task>/B/round_{0,2,4}/feedback.txt` is a reasonable
  sample) and confirm it stays specific to real failure modes seen in that task's
  `local_feedback.jsonl`, rather than drifting toward generic phrasing ("be careful",
  "double-check your work") as rounds accumulate. Do this before trusting a filter-pass-rate
  or pass@1 gap as evidence the mechanism is working, not just correlated with something
  else changing round to round.

- **Round-0 parity check.** Both conditions' round 0 is unconditioned by construction
  (`round_loop.py` only prepends a feedback prefix for Condition B when `t >= 1`), so
  `summary.jsonl`'s round-0 `filter_pass_rate` and `heldout_pass_at_1` should match between
  Condition A and Condition B within ordinary sampling noise. A large round-0 gap would mean
  the conditioning logic is leaking into round 0 somewhere (check `sampling.py`'s
  `feedback_file` argument isn't accidentally set for round 0) and would invalidate
  attributing any later-round gap to the feedback mechanism specifically.

## Reading results across a run

For each task, plot (2 lines: Condition A vs. B) x (2 panels: filter-pass-rate, held-out
pass@1) against round. A clean positive result looks like: round-0 parity holds, Condition
B's filter-pass-rate pulls ahead from round 1, and that pull-ahead is at least partially
visible in held-out pass@1 by the last round (not just in-sample at Grow time). A null or
negative result is still informative — report it as such rather than only reporting
favorable rounds.
