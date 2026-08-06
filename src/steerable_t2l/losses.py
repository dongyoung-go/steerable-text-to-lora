"""Shared loss primitives used by both the SFT trainer and validation scoring.

Kept separate from ``trainers/sft.py`` (rather than duplicated) because ``validation.py``
needs the identical per-sequence-normalized CE definition -- docs/03_training_validation.md
§4 is explicit that every validation metric must come from the same forward pass and the
same loss definition as training, or steering-margin numbers would not be comparable to the
training loss curve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sequence_normalized_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted CE over response tokens only, length-normalized per sequence.

    Returns ``[bs]`` -- one loss value per sequence -- so callers can both average for a
    scalar training loss and keep per-sample values for per-task validation aggregation.
    Masked positions (``labels == -100``) never contribute to the numerator or ``seq_len``.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    bs, seq_len = shift_labels.shape

    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(bs, seq_len)

    n_real = (shift_labels != -100).sum(-1).clamp_min(1)
    return per_token.sum(-1) / n_real
