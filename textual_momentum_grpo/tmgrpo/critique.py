"""Arm 2/3: same-iteration critique generation + Critique-GRPO-style rollout pooling.

README section 4, arm (2) construction note: "copy Critique-GRPO's pipeline exactly (sample ->
critique -> refined rollout -> pool original + refined in the GRPO group) with no internalization
or calibration, to reproduce their published numbers as a sanity check before trusting any of the
other arms." Arm (3) reuses the same critique generator but adds internalization/calibration
(see calibration.py) on top -- this module only produces the critique text and the pooled group,
it does not decide how those groups get scored (that's arm-specific, wired in the verl configs).
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm_client import LLMClient

CRITIQUE_SYSTEM_PROMPT = (
    "You are a careful math-problem reviewer. Given a problem and one attempted solution, write "
    "a short, specific critique of the attempt: point out any incorrect steps, unjustified leaps, "
    "or arithmetic errors, or confirm the reasoning is sound if it is. Do not reveal or restate "
    "the final numeric answer. Keep the critique under 150 words."
)

CRITIQUE_USER_TEMPLATE = "Problem:\n{problem}\n\nAttempted solution:\n{response}\n\nCritique:"


def generate_critique(client: LLMClient, problem: str, response: str) -> str:
    """One same-iteration critique of `response` to `problem`, via the frontier model."""
    return client.complete(
        system_prompt=CRITIQUE_SYSTEM_PROMPT,
        user_prompt=CRITIQUE_USER_TEMPLATE.format(problem=problem, response=response),
    )


@dataclass(frozen=True)
class RolloutSample:
    """One rollout in a GRPO group: the generated text and its scalar reward."""

    response: str
    reward: float
    conditioning_context: str | None = None  # e.g. the critique text this rollout was generated under


def pool_original_and_refined(
    original: list[RolloutSample], refined: list[RolloutSample]
) -> list[RolloutSample]:
    """Pool original + critique-refined rollouts into a single GRPO group.

    Matches Critique-GRPO's published pipeline: both the pre-critique and post-critique attempts
    are members of the same advantage-normalization group, each scored under its own true
    generating context (README section 4, arm (2): "scored under true generating context, as
    published" -- i.e. no internalization here; that's arm (3)'s addition via calibration.py).
    """
    return [*original, *refined]
