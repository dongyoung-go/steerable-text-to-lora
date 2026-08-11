"""Adapter layer: verl-facing entry points that wrap tmgrpo's own logic.

verl's custom reward function contract (confirmed via verl docs,
https://verl.readthedocs.io/en/latest/preparation/reward_function.html):
    def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

Wired in per-arm via `custom_reward_function.path=tmgrpo/verl_hooks.py` and (implicitly)
`custom_reward_function.name=compute_score`.

This module is the seam between the pure-Python, CPU-testable tmgrpo/ package and verl's actual
training loop -- it is the one place expected to need adjustment once verl is actually installed
and its Python API can be inspected directly (see configs/base.yaml's header comment and
docs/build_and_run_guide.md for this flagged risk). The pieces that need genuine verl-internals
knowledge -- injecting momentum/critique text into the rollout prompt but recomputing log-probs
under the unconditioned prompt for the gradient (README section 3, arms 3/5) -- are stubbed here
with a clear NotImplementedError rather than a guessed, untested implementation.
"""

from __future__ import annotations

from .reward import check_answer


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None
) -> float:
    """verl's reward-function contract: 1.0 if solution_str's boxed answer matches ground_truth."""
    return 1.0 if check_answer(solution_str, ground_truth) else 0.0


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


def recompute_unconditioned_logprobs(*args, **kwargs):
    """Arms 3/5 internalization (README section 3, step 2): recompute each response's log-probs
    under the prompt WITHOUT the conditioning context appended by `inject_conditioning_context`,
    for use as the internalized gradient target and as the numerator of the calibration ratio
    w_t (tmgrpo/calibration.py).

    Not implemented here: this requires a second forward pass through verl's actor (or reference)
    worker with a different prompt than the one used for sampling, which depends on verl's actual
    worker/DataProto API. That API could not be verified in the GPU-less build sandbox (see
    configs/base.yaml's header comment). Implement this against verl's real
    `actor_rollout_ref.actor.compute_log_prob` (or equivalent) once installed on the GPU node --
    do not guess at the call shape here.
    """
    raise NotImplementedError(
        "wire this against verl's actual log-prob-recomputation API once verl is installed on "
        "the GPU node -- see this function's docstring and configs/base.yaml's header comment."
    )
