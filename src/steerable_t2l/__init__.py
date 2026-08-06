"""Behavior-level Text-to-LoRA.

Implemented (see ``docs/01_env.md``, ``docs/02_model.md``):
    ``target_spec``  -- LoRA shapes and the query index layout, derived from AutoConfig alone
    ``hooks``        -- differentiable per-sample LoRA injection into a frozen target model
    ``hypernet``     -- the instruction-conditioned adapter generator

Specified but not implemented (see ``docs/03_training_validation.md``):
    ``data``, ``oracle``, ``trainers``, ``validation``
"""

__version__ = "0.1.0"
