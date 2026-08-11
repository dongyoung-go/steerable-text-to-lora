"""The instruction-conditioned LoRA generator.

Implements ``Design.md``:

    instruction text
       -> [instruction tokens || N compositional query tokens]
       -> causal forward through a frozen, LoRA-tuned decoder-only backbone
       -> non-causal refinement over the query tokens only
       -> shared MLP -> module-specific low-rank head
       -> LoRA A / B for every (layer, module)

See ``docs/02_model.md`` for the rationale behind each non-obvious choice; the ones that are
easy to get wrong and hard to notice are called out inline below.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from steerable_t2l.target_spec import N_ROLES, TargetSpec

DEFAULT_BACKBONE = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


@dataclass
class HyperNetConfig:
    """Architecture of the hypernetwork itself (the target model is described by ``TargetSpec``)."""

    backbone_dir: str = DEFAULT_BACKBONE

    # LoRA on the backbone. Dropout is deliberately 0: nonzero dropout would make the
    # generated LoRA stochastic for a fixed instruction, breaking the "identical
    # descriptions => identical LoRAs" invariant that batch-level description
    # deduplication depends on, and making train and eval behaviour diverge silently.
    backbone_lora_r: int = 16
    backbone_lora_alpha: int = 32
    backbone_lora_dropout: float = 0.0
    backbone_lora_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    refiner_layers: int = 2
    refiner_heads: int = 16
    refiner_mlp_ratio: int = 4

    decoder_mlp_ratio: int = 2
    head_rank: int = 128

    max_desc_len: int = 512
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> HyperNetConfig:
        d = dict(d)
        for key in ("backbone_lora_modules",):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------


def peft_lora_a_init(in_features: int, r: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """A ``[r, in_features]`` tensor initialized exactly as PEFT initializes ``lora_A``.

    PEFT's default ``init_lora_weights=True`` applies ``kaiming_uniform_(w, a=sqrt(5))`` to a
    weight of shape ``[r, in_features]``, i.e. the default init of ``nn.Linear(in_features, r)``.
    With ``a=sqrt(5)`` the gain is ``sqrt(2/6)`` and the bound collapses to ``1/sqrt(fan_in)``.
    """
    bound = 1.0 / math.sqrt(in_features)
    w = torch.empty(r, in_features)
    with torch.no_grad():
        w.uniform_(-bound, bound, generator=generator)
    return w


class CompositionalQueries(nn.Module):
    """``q_i = q_base + e_layer[l] + e_module[m] + e_role[role]``, then LayerNorm.

    ``1 + n_layers + n_modules + 2`` learned vectors instead of ``n_queries`` independent
    embeddings, so layer / module / role identity are disentangled additive factors that a
    rare (layer, module, role) combination can still be assembled from.
    """

    def __init__(self, spec: TargetSpec, d_model: int, emb_rms: float = 1.0, init_std: float = 0.02):
        super().__init__()
        self.spec = spec
        self.d_model = d_model

        self.q_base = nn.Parameter(torch.zeros(d_model))
        self.e_layer = nn.Embedding(spec.n_layers, d_model)
        self.e_module = nn.Embedding(spec.n_modules, d_model)
        self.e_role = nn.Embedding(N_ROLES, d_model)
        self.ln = nn.LayerNorm(d_model)

        for table in (self.e_layer, self.e_module, self.e_role):
            nn.init.normal_(table.weight, std=init_std)

        # These vectors are fed straight into a pretrained LLM alongside real token
        # embeddings. The final LayerNorm pins their RMS to `ln.weight`, so initializing
        # that to the backbone's input-embedding RMS is what keeps them in-distribution for
        # the first attention layer -- at the default gain of 1 they would be off by roughly
        # an order of magnitude.
        nn.init.constant_(self.ln.weight, emb_rms)

        layer_idx, module_idx, role_idx = spec.query_indices()
        self.register_buffer("layer_idx", layer_idx, persistent=False)
        self.register_buffer("module_idx", module_idx, persistent=False)
        self.register_buffer("role_idx", role_idx, persistent=False)

    def forward(self) -> torch.Tensor:
        """``[n_queries, d_model]``, in the serialization order defined by ``TargetSpec``."""
        q = (
            self.q_base
            + self.e_layer(self.layer_idx)
            + self.e_module(self.module_idx)
            + self.e_role(self.role_idx)
        )
        return self.ln(q)


class RefinerBlock(nn.Module):
    """Pre-LN self-attention + MLP, with no attention mask."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.SiLU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.ln_mlp(x))


class QueryRefiner(nn.Module):
    """Non-causal fixup over the query tokens only (``Design.md``, "Query Refinement").

    The backbone pass is causal, so query ``i`` only sees queries ``< i`` in an arbitrary
    serialization order. The queries are a *set*, not a sequence, so that ordering is an
    artifact. This stack lets every query exchange information with every other one --
    layer-0 representations informed by layer-31 and vice versa.

    It never attends to the instruction text (already folded into each query by the causal
    pass), so attention here is only ever over ``n_queries`` tokens and is cheap.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            RefinerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(d_model)

        # GPT-2 style: shrink residual-branch output projections so the stack starts near
        # the identity rather than immediately scrambling the backbone's hidden states.
        if n_layers:
            scale = 1.0 / math.sqrt(2 * n_layers * 2)
            for block in self.blocks:
                block.attn.out_proj.weight.data.mul_(scale)
                block.mlp[-1].weight.data.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)


class SharedDecoder(nn.Module):
    """The shared MLP applied identically to all query representations.

    Ends in a LayerNorm, which guarantees O(1) output RMS. That is not cosmetic: see the
    dead-gradient note on ``SteerableHyperLoRA._apply_zero_init``. Never zero-initialize
    this module and never gate its entire output to zero.
    """

    def __init__(self, d_model: int, mlp_ratio: int):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModuleHead(nn.Module):
    """Module-specific projection from a query representation to flattened LoRA A / B.

    ``Design.md`` asks for one projection head per module, shared across layers. Sharing
    across layers is exactly what happens (layer identity is carried additively by
    ``e_layer``, so a per-layer head would be redundant). Sharing a single head across the
    A and B *roles* is not possible -- A's width is ``r * in_features`` and B's is
    ``r * out_features``, and under GQA those differ (12288 vs 2048 for k/v_proj). So the
    ``Linear(d -> k)`` bottleneck is shared per module across both roles -- there really
    are ``n_modules`` module-specific projections -- and only the small ``Linear(k -> r*f)``
    output layer is per-role.

    The rank-``k`` factorization costs ``d*k + k*r*f`` instead of ``d*r*f``: a ~15x
    reduction at d=2048, k=128, r=8, f=2048, with negligible expressivity loss since the
    target is already a low-rank structured object.
    """

    def __init__(self, d_model: int, head_rank: int, r: int, in_features: int, out_features: int):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features

        self.bottleneck = nn.Linear(d_model, head_rank)
        self.out_A = nn.Linear(head_rank, r * in_features)
        self.out_B = nn.Linear(head_rank, r * out_features)

        # Only the *out* linears are ever zeroed (see _apply_zero_init). The bottleneck must
        # stay nonzero or the gradient into out_B.weight would vanish and B would be pinned
        # at zero forever.
        nn.init.normal_(self.bottleneck.weight, std=0.02)
        nn.init.zeros_(self.bottleneck.bias)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``h_a``/``h_b``: ``[bs, n_layers, d]`` -> ``A [bs, n_layers, r, in]``, ``B [bs, n_layers, out, r]``."""
        lead = h_a.shape[:-1]
        A = self.out_A(self.bottleneck(h_a)).view(*lead, self.r, self.in_features)
        B = self.out_B(self.bottleneck(h_b)).view(*lead, self.r, self.out_features)
        return A, B.transpose(-1, -2)


# --------------------------------------------------------------------------------------
# the hypernetwork
# --------------------------------------------------------------------------------------


def dedup(descs: list[str]) -> tuple[list[str], torch.Tensor]:
    """``(unique_descs, inverse_index)`` preserving first-appearance order.

    The backbone is LoRA-tuned, so description encodings cannot be cached across steps as
    they can when the encoder is frozen. Within a batch, though, every sample drawn from the
    same task shares a description -- run the backbone *and the heads* on the uniques and
    ``index_select`` back. Expanding before the heads would waste ~5.6 GFLOP per duplicated
    sample.
    """
    order: dict[str, int] = {}
    inverse = []
    for d in descs:
        if d not in order:
            order[d] = len(order)
        inverse.append(order[d])
    return list(order), torch.tensor(inverse, dtype=torch.long)


class SteerableHyperLoRA(nn.Module):
    """Maps a natural-language steering instruction to a full LoRA parameter set."""

    def __init__(
        self,
        spec: TargetSpec,
        config: HyperNetConfig | None = None,
        *,
        zero_init: bool = True,
        backbone: nn.Module | None = None,
        tokenizer=None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ):
        super().__init__()
        self.spec = spec
        self.config = config or HyperNetConfig()
        self.dtype = dtype

        self.backbone, self.tokenizer = self._build_backbone(backbone, tokenizer, dtype, device)
        d_model = int(self.backbone.config.hidden_size)
        self.d_model = d_model

        emb_rms = self._embedding_rms()
        self.queries = CompositionalQueries(spec, d_model, emb_rms=emb_rms)
        self.refiner = QueryRefiner(
            d_model,
            self.config.refiner_heads,
            self.config.refiner_layers,
            self.config.refiner_mlp_ratio,
            self.config.dropout,
        )
        self.shared_decoder = SharedDecoder(d_model, self.config.decoder_mlp_ratio)
        self.heads = nn.ModuleDict(
            {
                m: ModuleHead(
                    d_model, self.config.head_rank, spec.r, spec.in_features[m], spec.out_features[m]
                )
                for m in spec.target_modules
            }
        )

        # Cached [n_layers] gather indices for the role-A query of each module.
        for module in spec.target_modules:
            self.register_buffer(
                f"_qbase_{module}", spec.query_base_indices(module), persistent=False
            )

        if zero_init:
            self._apply_zero_init(seed=seed)

        self.to(device)
        # AutoModel.from_pretrained loads the backbone in eval() mode; nn.Module assigns a
        # freshly-constructed submodule's own .training as-is rather than syncing it to the
        # parent's, so self.backbone silently stayed in eval() even though self.training was
        # already True. HF's GradientCheckpointingLayer no-ops when self.training is False (see
        # trainers/sft.py's identical target.train() gotcha for the target model), so every
        # trainer that never explicitly called hypernet.train() got the backbone's full,
        # unchecked 36-layer autograd graph on every encode() call instead of the intended
        # one-layer-at-a-time recompute -- silently, since nothing errors, it just costs ~8x the
        # memory a checkpointed forward should (repro: bs=8 forward-only peak was the same
        # 21.8GB with gradient_checkpointing True or False). Eval-only callers already call
        # hypernet.eval() explicitly afterwards, so defaulting to train() here is safe.
        self.train()

    # -- construction helpers -------------------------------------------------------

    def _build_backbone(self, backbone, tokenizer, dtype, device):
        from transformers import AutoModel, AutoTokenizer

        if backbone is None:
            # AutoModel, never AutoModelForCausalLM: the causal-LM wrapper would compute
            # logits over a ~152k vocabulary for all (L + n_queries) positions -- over a
            # gigabyte of activations held in the graph for a tensor we discard.
            backbone = AutoModel.from_pretrained(
                self.config.backbone_dir,
                dtype=dtype,
                attn_implementation=self.config.attn_implementation,
            )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_dir)

        # Left padding puts the query tokens at the last n_queries positions of every row,
        # making the slice a constant `[:, -n_queries:]` instead of a per-row gather.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        backbone.config.use_cache = False
        for p in backbone.parameters():
            p.requires_grad_(False)

        from peft import LoraConfig, get_peft_model

        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=self.config.backbone_lora_r,
                lora_alpha=self.config.backbone_lora_alpha,
                lora_dropout=self.config.backbone_lora_dropout,
                bias="none",
                target_modules=list(self.config.backbone_lora_modules),
            ),
            autocast_adapter_dtype=True,  # adapters in fp32
        )
        if self.config.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return backbone.to(device), tokenizer

    def _embedding_rms(self) -> float:
        with torch.no_grad():
            w = self.backbone.get_input_embeddings().weight
            return float(w.float().pow(2).mean().sqrt())

    def _apply_zero_init(self, seed: int | None = None) -> None:
        """Make the generated LoRA an *exactly* zero delta at step 0.

        ``out_B`` has both weight and bias zeroed, so ``B == 0`` for every input regardless
        of what the refiner, shared decoder and bottleneck produce -- ``deltaW = B @ A`` is
        exactly zero, not approximately, and step-0 loss equals the frozen base model's.
        ``out_A`` emits a constant equal to PEFT's Kaiming init, matching standard LoRA.

        The hazard here is not a nonzero delta, it is a dead gradient. Note carefully what
        is and is not live on the first step. With ``out_B.weight == 0``::

            dL/dA                    ~  B == 0                                -> out_A idle
            dL/d(bottleneck output)  =  out_A.weight^T dL/dA + out_B.weight^T dL/dB
                                     =           0           +          0     -> idle

        so on step 0 **only ``out_B``'s weight and bias move** -- exactly as in standard
        LoRA, where ``B = 0`` means only B moves first. The bottleneck, shared decoder,
        refiner, query bank and backbone LoRA all unblock on step 1, once ``out_B.weight``
        is nonzero. That one-step lag is intended and is not a saddle.

        What would be fatal is that unblocking never happening. ``dL/d(out_B.weight)`` is
        proportional to ``dL/dB (x) bottleneck_activation``; if the shared decoder or the
        bottleneck emitted zeros, it too would be zero and B would be pinned at zero
        forever -- a total training failure that raises nothing and shows up only as a flat
        loss curve. Hence: the shared decoder ends in a LayerNorm (O(1) output RMS), the
        bottleneck is normal-initialized (only the *out* linears are zeroed), and neither
        contains dropout. ``tests/test_grad_flow.py`` enforces all of this.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        with torch.no_grad():
            for module, head in self.heads.items():
                a_init = peft_lora_a_init(self.spec.in_features[module], self.spec.r, generator)
                nn.init.zeros_(head.out_A.weight)
                head.out_A.bias.copy_(a_init.flatten().to(head.out_A.bias.dtype))
                nn.init.zeros_(head.out_B.weight)
                nn.init.zeros_(head.out_B.bias)

    # -- forward --------------------------------------------------------------------

    def tokenize(self, descs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Chat-template and left-pad the instructions. Cheap; safe to cache at startup."""
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": d}], tokenize=False, add_generation_prompt=True
            )
            for d in descs
        ]
        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.config.max_desc_len,
            return_tensors="pt",
        )
        device = self.queries.q_base.device
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def encode(self, descs: list[str]) -> torch.Tensor:
        """Instructions -> ``[len(descs), n_queries, d_model]`` refined query representations."""
        input_ids, desc_mask = self.tokenize(descs)
        return self.encode_tokenized(input_ids, desc_mask)

    def encode_tokenized(self, input_ids: torch.Tensor, desc_mask: torch.Tensor) -> torch.Tensor:
        bs = input_ids.shape[0]
        n_q = self.spec.n_queries

        tok_emb = self.backbone.get_input_embeddings()(input_ids)
        q = self.queries().to(tok_emb.dtype).unsqueeze(0).expand(bs, -1, -1)
        inputs_embeds = torch.cat([tok_emb, q], dim=1)

        ones = torch.ones(bs, n_q, dtype=desc_mask.dtype, device=desc_mask.device)
        attn_mask = torch.cat([desc_mask, ones], dim=1)

        # Explicit position_ids is required, not cosmetic. The default is arange(L + n_q),
        # which *counts pad tokens*: the same instruction batched next to a longer neighbour
        # would get different RoPE phases on its query tokens and emit a different LoRA --
        # silent, batch-composition-dependent nondeterminism. This places the first real
        # token at 0 and query j at (true length + j) regardless of padding.
        position_ids = (attn_mask.cumsum(-1) - 1).clamp(min=0)

        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        # The default causal mask is exactly what Design.md asks for: every query attends to
        # the whole instruction plus the queries before it. The non-causal correction is the
        # refiner, below.
        h = out.last_hidden_state[:, -n_q:, :]

        # fp32 for the from-scratch stack: it only ever touches [bs, n_queries, d], so the
        # cost is negligible, while the exact-zero-init contract needs bit-exact zeros and
        # early gradients (which pass through B ~ 0) would be flushed by bf16.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            return self.shared_decoder(self.refiner(h))

    def heads_forward(self, h: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Refined queries -> ``{module: (A [bs, n_layers, r, in], B [bs, n_layers, out, r])}``."""
        out = {}
        for module, head in self.heads.items():
            base = getattr(self, f"_qbase_{module}").to(h.device)
            out[module] = head(h[:, base], h[:, base + 1])
        return out

    def forward(self, descs: list[str]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return self.heads_forward(self.encode(descs))

    def generate_for_batch(
        self, descs: list[str]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Like ``forward``, but encodes only the unique instructions and expands afterwards.

        Numerically identical to ``forward`` (``tests/test_dedup.py``), and the expansion is
        differentiable -- backward scatter-adds into the unique rows.
        """
        uniq, inverse = dedup(descs)
        per_module = self.heads_forward(self.encode(uniq))
        inverse = inverse.to(next(iter(per_module.values()))[0].device)
        return {m: (A[inverse], B[inverse]) for m, (A, B) in per_module.items()}

    # -- interop --------------------------------------------------------------------

    @torch.no_grad()
    def to_peft_state_dict(self, desc: str, prefix: str = "base_model.model.model.layers") -> dict:
        """A single instruction's LoRA as a PEFT-compatible state dict.

        Not used by training (hooks are), but needed to hand an adapter to anything that
        speaks PEFT -- and it is what ``tests/test_hook_vs_peft.py`` cross-checks against.
        """
        per_module = self.forward([desc])
        state: dict[str, torch.Tensor] = {}
        for module, (A, B) in per_module.items():
            for layer in range(self.spec.n_layers):
                stem = f"{prefix}.{layer}.self_attn.{module}"
                state[f"{stem}.lora_A.weight"] = A[0, layer].detach().cpu().contiguous()
                state[f"{stem}.lora_B.weight"] = B[0, layer].detach().cpu().contiguous()
        return state

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Trainable parameters, grouped for reporting and for per-group learning rates."""
        groups: dict[str, list[nn.Parameter]] = {
            "backbone_lora": [],
            "queries": [],
            "refiner": [],
            "shared_decoder": [],
            "heads": [],
        }
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                groups["backbone_lora"].append(p)
            elif name.startswith("queries."):
                groups["queries"].append(p)
            elif name.startswith("refiner."):
                groups["refiner"].append(p)
            elif name.startswith("shared_decoder."):
                groups["shared_decoder"].append(p)
            elif name.startswith("heads."):
                groups["heads"].append(p)
        return groups

    def parameter_report(self) -> str:
        groups = self.parameter_groups()
        lines = [f"{name:<16} {sum(p.numel() for p in ps) / 1e6:8.2f} M" for name, ps in groups.items()]
        total = sum(p.numel() for ps in groups.values() for p in ps)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        lines.append(f"{'-' * 26}")
        lines.append(f"{'trainable':<16} {total / 1e6:8.2f} M   (fp32 + grads + AdamW ~ {total * 16 / 1e9:.2f} GB)")
        lines.append(f"{'frozen':<16} {frozen / 1e6:8.2f} M")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


def estimate_memory_gb(model: SteerableHyperLoRA, target_params: int | None = None) -> str:
    """A rough training-memory budget from the *actual* parameter counts of this model."""
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        f"  frozen backbone (bf16)     {frozen * 2 / 1e9:6.2f} GB   ({frozen / 1e6:.1f} M params)",
        f"  trainable + grads + Adam   {trainable * 16 / 1e9:6.2f} GB   ({trainable / 1e6:.1f} M params, fp32)",
    ]
    if target_params is not None:
        lines.insert(1, f"  frozen target   (bf16)     {target_params * 2 / 1e9:6.2f} GB")
    lines.append("  + activations, dominated by the target forward and the CE over the vocabulary")
    lines.append("  see docs/03_training_validation.md for the full budget at bs=16")
    return "\n".join(lines)


def self_check(real: bool = False) -> int:  # pragma: no cover - dev tool
    from steerable_t2l.testing import get_tokenizer, tiny_hypernet, tiny_spec

    if real:
        spec = TargetSpec.from_pretrained(DEFAULT_TARGET)
        config = HyperNetConfig()
        model = SteerableHyperLoRA(spec, config, device="cpu", dtype=torch.float32)
    else:
        if get_tokenizer() is None:
            print("self-check needs a Qwen2.5 tokenizer (cached or downloadable)")
            return 1
        spec = tiny_spec()
        model = tiny_hypernet(spec)
        config = model.config

    print(spec.summary())
    print()
    print(model.parameter_report())
    print()

    per_module = model(["Solve the problem step by step and verify every arithmetic operation."])
    for module, (A, B) in per_module.items():
        print(f"  {module:<10} A {tuple(A.shape)}   B {tuple(B.shape)}")
        assert (B == 0).all(), f"zero-init violated: {module} B is nonzero"
        assert (A != 0).any(), f"{module} A is all zero -- PEFT init did not land"
    print("\nzero-init contract holds: every B is exactly 0, so deltaW == 0 at step 0")

    print(f"\nestimated memory ({'real' if real else 'synthetic'} config):")
    print(estimate_memory_gb(model))
    return 0


def main() -> int:  # pragma: no cover - dev tool
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true", help="build a model and verify invariants")
    ap.add_argument("--real", action="store_true", help="use the real Qwen2.5 backbone/target (downloads)")
    args = ap.parse_args()
    if not args.self_check:
        ap.print_help()
        return 0
    return self_check(real=args.real)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
