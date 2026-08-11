"""README section 6 risk: "Textual gradients may be confabulated rather than causally accurate --
spot-check against ground-truth batch statistics."

This is deliberately a lightweight heuristic check, not a confabulation-detector: it flags textual
gradients whose claimed direction (accuracy improving / worsening / stuck) disagrees with the
step's actual measured accuracy delta, for a human to review. It does not try to verify claims
about *specific* error patterns -- that would require NLI-grade claim extraction, out of scope
for a go/no-go pass.
"""

from __future__ import annotations

from dataclasses import dataclass

IMPROVING_WORDS = ("improv", "progress", "better", "increas", "succeed")
WORSENING_WORDS = ("worse", "regress", "declin", "fail", "struggl", "stuck")


@dataclass(frozen=True)
class SpotCheckResult:
    claimed_direction: str  # "improving" | "worsening" | "stuck" | "unclear"
    measured_direction: str  # "improving" | "worsening" | "stuck"
    agrees: bool


def _claimed_direction(textual_gradient: str) -> str:
    text = textual_gradient.lower()
    improving = any(w in text for w in IMPROVING_WORDS)
    worsening = any(w in text for w in WORSENING_WORDS)
    if improving and not worsening:
        return "improving"
    if worsening and not improving:
        return "worsening"
    return "unclear"


def _measured_direction(accuracy_delta: float, flat_threshold: float = 0.02) -> str:
    if accuracy_delta > flat_threshold:
        return "improving"
    if accuracy_delta < -flat_threshold:
        return "worsening"
    return "stuck"


def spot_check(
    textual_gradient: str, prev_step_accuracy: float, this_step_accuracy: float
) -> SpotCheckResult:
    """Compare a textual gradient's claimed trend against the step's actual accuracy delta.

    `agrees=False` doesn't prove confabulation (the gradient may be diagnosing sub-population
    patterns invisible in a single scalar accuracy number) -- it's a flag for manual review, per
    the README's own framing of this as a heuristic signal.
    """
    claimed = _claimed_direction(textual_gradient)
    measured = _measured_direction(this_step_accuracy - prev_step_accuracy)
    agrees = claimed == "unclear" or claimed == measured
    return SpotCheckResult(claimed_direction=claimed, measured_direction=measured, agrees=agrees)
