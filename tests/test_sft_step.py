"""sft_step: dedup -> generate_for_batch -> build_sites -> lora_hooks -> CE + L2 reg.

See docs/03_training_validation.md §3.
"""

from __future__ import annotations

import torch

from steerable_t2l.trainers.sft import SFTConfig, sft_step


def test_sft_step_finite_loss_and_zero_init_grad_pattern(hypernet, target_model_for_tokenizer, spec):
    hypernet.zero_grad(set_to_none=True)
    bs, seq_len = 3, 6
    vocab = target_model_for_tokenizer.config.vocab_size
    input_ids = torch.randint(0, vocab, (bs, seq_len))
    attention_mask = torch.ones(bs, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "descs": ["a", "b", "a"],
    }
    config = SFTConfig(l2_reg_generated_w=1e-3)
    out = sft_step(batch, hypernet, target_model_for_tokenizer, spec, config)

    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    grad_norms = {
        name: sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
        for name, params in hypernet.parameter_groups().items()
    }
    # At zero-init, only the decoder heads move on step 0 -- see docs/02_model.md.
    assert grad_norms["heads"] > 0
    assert all(p.grad is None for p in target_model_for_tokenizer.parameters())


def test_sft_step_reg_scales_with_l2_reg_generated_w(hypernet, target_model_for_tokenizer, spec):
    bs, seq_len = 2, 5
    vocab = target_model_for_tokenizer.config.vocab_size
    input_ids = torch.randint(0, vocab, (bs, seq_len))
    attention_mask = torch.ones(bs, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "descs": ["x", "y"]}

    out_small = sft_step(batch, hypernet, target_model_for_tokenizer, spec, SFTConfig(l2_reg_generated_w=0.0))
    out_big = sft_step(batch, hypernet, target_model_for_tokenizer, spec, SFTConfig(l2_reg_generated_w=1.0))

    assert out_small["reg_loss"].item() == 0.0
    assert out_big["reg_loss"].item() > 0.0


def test_sft_step_with_target_gradient_checkpointing_and_backward_fn(hypernet, target_model_for_tokenizer, spec):
    """Regression test: gradient checkpointing recomputes the target's forward during
    backward(). A bare `with lora_hooks(...): forward()` block removes its hooks before that
    recompute happens, so the recompute would silently run with no LoRA injection at all --
    either corrupting gradients or (when tensor counts also happen to mismatch) raising
    torch.utils.checkpoint.CheckpointError. `backward_fn` keeps the hooks attached through the
    whole backward call, which is what this test would fail without.
    """
    target_model_for_tokenizer.config.use_cache = False
    target_model_for_tokenizer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    target_model_for_tokenizer.train()

    hypernet.zero_grad(set_to_none=True)
    bs, seq_len = 3, 6
    vocab = target_model_for_tokenizer.config.vocab_size
    input_ids = torch.randint(0, vocab, (bs, seq_len))
    attention_mask = torch.ones(bs, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :2] = -100
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "descs": ["a", "b", "a"],
    }
    config = SFTConfig(l2_reg_generated_w=1e-3)

    backward_calls = []
    out = sft_step(
        batch, hypernet, target_model_for_tokenizer, spec, config,
        backward_fn=lambda loss: (backward_calls.append(1), loss.backward())[-1],
    )

    assert backward_calls == [1]
    assert torch.isfinite(out["loss"])
    grad_norms = {
        name: sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
        for name, params in hypernet.parameter_groups().items()
    }
    assert grad_norms["heads"] > 0
    assert all(p.grad is None for p in target_model_for_tokenizer.parameters())
