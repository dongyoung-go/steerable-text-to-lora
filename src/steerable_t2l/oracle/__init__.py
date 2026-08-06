"""Oracle LoRAs and SVD canonicalization. See ``docs/03_training_validation.md`` §2.

``train_oracle.py``
    One vanilla PEFT LoRA per task on the target model, with a config identical to
    ``TargetSpec`` (asserted). Same data path as SFT so oracle and hypernetwork see
    byte-identical text.

``canonicalize.py``
    ``B @ A`` is not unique: ``(BR)(R⁻¹A)`` is the same function for any invertible ``R``, so
    independently trained oracles land in mutually inconsistent bases and regressing onto
    their raw weights is ill-posed. Canonicalizes each ``(layer, module)`` pair via the exact
    QR + tiny-SVD trick, in float64, with a deterministic sign gauge, implementing PEFT's
    swapped ``ΔW = lora_B · lora_A`` convention rather than ``Design.md``'s ``ΔW = A·B``.
"""

from steerable_t2l.oracle.canonicalize import (
    canonicalize_adapter,
    canonicalize_state_dict,
    fix_svd_signs,
    load_and_canonicalize_oracle,
)
from steerable_t2l.oracle.train_oracle import (
    EarlyStopper,
    OracleConfig,
    build_oracle_peft_model,
    train_one_oracle,
)

__all__ = [
    "EarlyStopper",
    "OracleConfig",
    "build_oracle_peft_model",
    "canonicalize_adapter",
    "canonicalize_state_dict",
    "fix_svd_signs",
    "load_and_canonicalize_oracle",
    "train_one_oracle",
]
