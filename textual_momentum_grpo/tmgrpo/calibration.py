"""Arm 3/5: unconditioned-scoring internalization + the token-level calibration ratio w_t.

README section 3, steps 2-3:
  - Internalization: GRPO advantages are computed as usual, but the log-probs used for the
    gradient are computed UNCONDITIONED on the conditioning context (critique text for arm 3,
    textual momentum M_{t-1} for arm 5) -- the update targets pi_theta(.|q) rather than
    pi_theta(.|q, context).
  - Calibration ratio: w_t = pi_rollout(y_t | q, y_<t) / pi_rollout(y_t | q, context, y_<t),
    computed under the FIXED rollout policy (a same-parameter, different-context ratio, not an
    old/new-policy ratio). min(w_t, w_max) is multiplied into the standard clipped GRPO term.

This module is pure array math over per-token log-probs supplied by the caller (verl, on the GPU
node, provides the two log-prob arrays per response). It has no verl/torch dependency itself so it
is unit-testable here against synthetic log-prob arrays.

W_MAX is an unconfirmed placeholder (not something the user was asked to fix a value for) --
README section 6 itself says to watch the w_t trajectory over training as a diagnostic for
whether this value is too aggressive (collapsing toward uniformly low w_t). Treat this as a
starting point to be tuned once real w_t histograms are observed on a GPU run, not a settled
choice.
"""

from __future__ import annotations

import numpy as np

W_MAX = 5.0  # placeholder -- see module docstring; retune from observed w_t trajectories.


def calibration_ratio(
    logp_unconditioned: np.ndarray,
    logp_conditioned: np.ndarray,
    w_max: float = W_MAX,
) -> np.ndarray:
    """Per-token w_t = exp(logp_unconditioned - logp_conditioned), clipped to at most w_max.

    Both inputs are 1D arrays of per-token log-probs under the SAME (fixed) rollout policy
    parameters, differing only in whether the conditioning context (critique/momentum) was in
    the prefix. A token much more likely WITH the context than without it gets logp_conditioned
    >> logp_unconditioned, so its ratio is small -- down-weighting tokens that leaned heavily on
    the context. A token the unconditioned policy already favored just as much keeps ratio ~= 1.
    """
    if logp_unconditioned.shape != logp_conditioned.shape:
        raise ValueError(
            f"shape mismatch: logp_unconditioned {logp_unconditioned.shape} vs "
            f"logp_conditioned {logp_conditioned.shape}"
        )
    ratio = np.exp(logp_unconditioned - logp_conditioned)
    return np.minimum(ratio, w_max)


def apply_calibration(
    pg_loss_per_token: np.ndarray,
    logp_unconditioned: np.ndarray,
    logp_conditioned: np.ndarray,
    w_max: float = W_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiply the calibration ratio into a per-token clipped-GRPO loss array.

    Returns (calibrated_loss, w_t) so callers can log the w_t trajectory (README section 6
    diagnostic) alongside the loss actually used.
    """
    w_t = calibration_ratio(logp_unconditioned, logp_conditioned, w_max=w_max)
    return pg_loss_per_token * w_t, w_t
