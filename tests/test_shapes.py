"""TargetSpec derives every LoRA width from AutoConfig, including the GQA asymmetries."""

from __future__ import annotations

import pytest

from steerable_t2l.target_spec import N_ROLES, TargetSpec, module_widths


def test_gqa_widths(target_config, spec):
    hidden = target_config.hidden_size
    head_dim = hidden // target_config.num_attention_heads
    kv_width = target_config.num_key_value_heads * head_dim

    # The bug this guards: assuming every projection is hidden x hidden. Under GQA the
    # k/v projections are narrower, and o_proj *consumes* num_heads*head_dim.
    assert spec.out_features["k_proj"] == kv_width
    assert spec.out_features["v_proj"] == kv_width
    assert spec.out_features["k_proj"] < hidden

    assert spec.in_features["q_proj"] == hidden
    assert spec.out_features["q_proj"] == target_config.num_attention_heads * head_dim
    assert spec.in_features["o_proj"] == target_config.num_attention_heads * head_dim
    assert spec.out_features["o_proj"] == hidden


def test_o_proj_in_features_not_hardcoded_to_hidden():
    """A config where num_heads*head_dim != hidden_size must still produce correct widths."""
    from transformers import Qwen2Config

    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,  # 4 * 32 = 128 != hidden_size 64
    )
    assert module_widths(config, "q_proj") == (64, 128)
    assert module_widths(config, "o_proj") == (128, 64)
    assert module_widths(config, "k_proj") == (64, 64)


def test_mlp_module_widths(target_config):
    inter = target_config.intermediate_size
    hidden = target_config.hidden_size
    assert module_widths(target_config, "gate_proj") == (hidden, inter)
    assert module_widths(target_config, "down_proj") == (inter, hidden)


def test_query_count_and_scaling(spec, target_config):
    assert spec.n_queries == target_config.num_hidden_layers * spec.n_modules * N_ROLES
    assert spec.scaling == pytest.approx(spec.lora_alpha / spec.r)

    rs = spec.replace(use_rslora=True)
    assert rs.scaling == pytest.approx(spec.lora_alpha / spec.r * spec.r**0.5)


def test_roundtrip_through_dict(spec):
    restored = TargetSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.in_features == spec.in_features


def test_rejects_unknown_module(target_config):
    with pytest.raises(ValueError, match="unsupported target module"):
        module_widths(target_config, "not_a_proj")


def test_verify_against_live_model(spec, target_model):
    spec.verify_against(target_model)

    wrong = spec.replace(module_out_features=tuple(w + 1 for w in spec.module_out_features))
    with pytest.raises(ValueError, match="spec says in/out"):
        wrong.verify_against(target_model)
