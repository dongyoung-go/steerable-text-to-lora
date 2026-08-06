"""Tiny CPU fixtures shared by the test suite and ``hypernet --self-check``.

Everything here builds randomly-initialized models small enough to run on a login node.
The only external artifact needed is a Qwen2.5 *tokenizer* (a few MB, and already in the
local HF cache); no model weights are downloaded.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from steerable_t2l.hypernet import HyperNetConfig, SteerableHyperLoRA
from steerable_t2l.target_spec import DEFAULT_TARGET_MODULES, TargetSpec

# Any Qwen2.5 tokenizer will do; try the ones most likely to be cached first.
TOKENIZER_CANDIDATES = (
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B",
)


def get_tokenizer():
    """A Qwen2.5 tokenizer, or ``None`` if none is reachable (tests skip in that case)."""
    for name in TOKENIZER_CANDIDATES:
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception:  # noqa: BLE001 - any failure means "try the next one"
            continue
    return None


def tiny_target_config(
    hidden: int = 32,
    layers: int = 4,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    vocab: int = 256,
) -> Qwen2Config:
    """A target-model config with genuine GQA (``n_kv_heads < n_heads``).

    The asymmetry is deliberate: it is what makes ``k_proj``/``v_proj`` narrower than
    ``q_proj``/``o_proj``, which is the shape bug most likely to slip through.
    """
    return Qwen2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=512,
        attn_implementation="eager",
    )


def tiny_target_model(config: Qwen2Config | None = None) -> Qwen2ForCausalLM:
    config = config or tiny_target_config()
    model = Qwen2ForCausalLM(config).to(torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tiny_target_model_for_tokenizer(
    tokenizer, config: Qwen2Config | None = None
) -> Qwen2ForCausalLM:
    """A tiny target model whose vocabulary covers ``tokenizer``'s real id range.

    ``tiny_target_model`` alone uses a fixed small vocab (256) and is meant to be paired with
    synthetic ``randint`` token ids, as the model-architecture tests do. Anything that runs
    real tokenized text (the data/validation/training pipeline) through a tiny target needs a
    vocab sized to the real tokenizer instead, or embedding lookups go out of range.
    """
    config = config or tiny_target_config()
    vocab = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0)) + 64
    config = Qwen2Config(
        vocab_size=vocab,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=max(4096, config.max_position_embeddings),
        attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(config).to(torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tiny_backbone(tokenizer, hidden: int = 32, layers: int = 2) -> Qwen2Model:
    """A tiny decoder whose embedding table covers the real tokenizer's id range."""
    vocab = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0)) + 64
    config = Qwen2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        attn_implementation="eager",
    )
    return Qwen2Model(config).to(torch.float32)


def tiny_spec(config: Qwen2Config | None = None, **kwargs) -> TargetSpec:
    config = config or tiny_target_config()
    kwargs.setdefault("target_modules", DEFAULT_TARGET_MODULES)
    kwargs.setdefault("r", 4)
    kwargs.setdefault("lora_alpha", 8)
    return TargetSpec.from_config(config, model_dir="<synthetic>", **kwargs)


def tiny_hypernet(
    spec: TargetSpec | None = None,
    tokenizer=None,
    *,
    zero_init: bool = True,
    seed: int | None = 0,
    backbone_hidden: int = 32,
    **config_overrides,
) -> SteerableHyperLoRA:
    """A fully-wired ``SteerableHyperLoRA`` on tiny random models, in fp32 on CPU."""
    tokenizer = tokenizer or get_tokenizer()
    if tokenizer is None:  # pragma: no cover - handled by a pytest skip
        raise RuntimeError("no Qwen2.5 tokenizer available")
    spec = spec or tiny_spec()

    cfg_kwargs = {
        "refiner_layers": 1,
        "refiner_heads": 4,
        "refiner_mlp_ratio": 2,
        "decoder_mlp_ratio": 2,
        "head_rank": 8,
        "backbone_lora_r": 4,
        "backbone_lora_alpha": 8,
        "max_desc_len": 64,
        "attn_implementation": "eager",
        "gradient_checkpointing": False,
    }
    cfg_kwargs.update(config_overrides)

    return SteerableHyperLoRA(
        spec,
        HyperNetConfig(**cfg_kwargs),
        zero_init=zero_init,
        backbone=tiny_backbone(tokenizer, hidden=backbone_hidden),
        tokenizer=tokenizer,
        dtype=torch.float32,
        device="cpu",
        seed=seed,
    )
