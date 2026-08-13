"""Adapter layer: verl-facing entry points that wrap tmgrpo's own logic.

verl's custom reward function contract (confirmed via verl docs,
https://verl.readthedocs.io/en/latest/preparation/reward_function.html):
    def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

Wired in per-arm via `custom_reward_function.path=tmgrpo/verl_hooks.py` and (implicitly)
`custom_reward_function.name=compute_score`.

This module is the seam between the pure-Python, CPU-testable tmgrpo/ package and verl's actual
training loop. Recomputing log-probs under the unconditioned prompt (README section 3 steps 2-3,
arms 3/5) needs live access to the trainer's tokenizer/actor-worker-group/`_compute_old_log_prob`,
so that logic lives on `tmgrpo.verl_trainer.TMGrpoTrainer` instead of as a standalone function here.
"""

from __future__ import annotations

import os
import sys

# verl loads this file standalone via importlib (see verl.utils.import_utils.load_module),
# not as part of the `tmgrpo` package, so relative imports fail with
# "attempted relative import with no known parent package". Add the repo root to sys.path
# and import absolutely instead.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmgrpo.reward import check_answer


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None
) -> float:
    """verl's reward-function contract: 1.0 if solution_str's boxed answer matches ground_truth."""
    return 1.0 if check_answer(solution_str, ground_truth) else 0.0


def truncate_head_tail(text: str, head: int, tail: int = 0) -> str:
    """Truncate `text` to at most `head + tail` characters, keeping both ends.

    Used by `tmgrpo.verl_trainer.TMGrpoTrainer._build_step_summary` (README section 3 step 4's
    input): a plain head-only cut on a math solution tends to drop the final boxed answer, which
    is usually the most diagnostic part of the response for the frontier model's textual-gradient
    call. Keeping a tail preserves that even when the middle "working" gets cut. `tail=0` degrades
    to a plain head truncation, used for the (short, no informative tail) problem statement.
    """
    if len(text) <= head + tail:
        return text
    if tail == 0:
        return text[:head] + " …[truncated]"
    return text[:head] + " …[truncated]… " + text[len(text) - tail :]


def inject_conditioning_context(prompt: list[dict], context: str) -> list[dict]:
    """Arms 2-5: append a critique or momentum directive to the prompt before rollout sampling.

    `prompt` is verl's chat-message list (`[{"role": "user", "content": ...}]`, matching the
    vendored MATH jsonl schema). Appends the context as a second user turn rather than mutating
    the original problem statement, so `recompute_unconditioned_logprobs` below can reconstruct
    the unconditioned prompt by simply dropping the last turn.
    """
    if not context:
        return prompt
    return [*prompt, {"role": "user", "content": f"Guidance for this attempt:\n{context}"}]
