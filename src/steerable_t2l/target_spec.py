"""LoRA shapes and the query-token index layout for the frozen target model.

``TargetSpec`` is derived from ``AutoConfig`` **alone** -- no model weights are loaded. The
reconstruction trainer needs the target's shapes but never its weights, so requiring a live
model (as the reference implementation does) would mean holding several GB for nothing.

See ``docs/02_model.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PretrainedConfig

# LoRA roles, in the order they appear in the query serialization.
ROLE_A, ROLE_B = 0, 1
N_ROLES = 2
ROLE_NAMES = ("A", "B")

DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

SUPPORTED_MODULES: frozenset[str] = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def _head_dim(config: PretrainedConfig) -> int:
    dim = getattr(config, "head_dim", None)
    if dim is None:
        dim = config.hidden_size // config.num_attention_heads
    return int(dim)


def module_widths(config: PretrainedConfig, module: str) -> tuple[int, int]:
    """Return ``(in_features, out_features)`` of ``module`` in a decoder layer of ``config``.

    Every width is derived from the config rather than assumed equal to ``hidden_size``.
    Two of them differ under grouped-query attention and would otherwise be silently wrong:

    * ``k_proj`` / ``v_proj`` project to ``num_key_value_heads * head_dim`` -- 256, not 1536,
      for Qwen2.5-1.5B.
    * ``o_proj`` consumes ``num_attention_heads * head_dim``, which merely *happens* to equal
      ``hidden_size`` for this model and does not in general.
    """
    if module not in SUPPORTED_MODULES:
        raise ValueError(f"unsupported target module {module!r}; expected one of {sorted(SUPPORTED_MODULES)}")

    hidden = int(config.hidden_size)
    inter = int(getattr(config, "intermediate_size", hidden))
    hd = _head_dim(config)
    q_width = int(config.num_attention_heads) * hd
    kv_width = int(getattr(config, "num_key_value_heads", config.num_attention_heads)) * hd

    return {
        "q_proj": (hidden, q_width),
        "k_proj": (hidden, kv_width),
        "v_proj": (hidden, kv_width),
        "o_proj": (q_width, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }[module]


@dataclass(frozen=True)
class TargetSpec:
    """Everything the hypernetwork needs to know about the frozen target model.

    Widths are stored as tuples aligned with ``target_modules`` (rather than dicts) so the
    dataclass round-trips cleanly through ``asdict`` / ``TargetSpec(**d)`` in checkpoints.
    Use the ``in_features`` / ``out_features`` properties for keyed access.
    """

    model_dir: str
    n_layers: int
    hidden_size: int
    target_modules: tuple[str, ...]
    module_in_features: tuple[int, ...]
    module_out_features: tuple[int, ...]
    r: int = 8
    lora_alpha: int = 16
    use_rslora: bool = False
    lora_dropout: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.target_modules)
        if not n:
            raise ValueError("target_modules must be non-empty")
        if len(self.module_in_features) != n or len(self.module_out_features) != n:
            raise ValueError("module width tuples must align with target_modules")
        if len(set(self.target_modules)) != n:
            raise ValueError(f"duplicate entries in target_modules: {self.target_modules}")
        if self.r < 1:
            raise ValueError("r must be >= 1")

    # -- construction ---------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        *,
        model_dir: str,
        target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
        r: int = 8,
        lora_alpha: int = 16,
        use_rslora: bool = False,
        lora_dropout: float = 0.0,
    ) -> TargetSpec:
        widths = [module_widths(config, m) for m in target_modules]
        return cls(
            model_dir=model_dir,
            n_layers=int(config.num_hidden_layers),
            hidden_size=int(config.hidden_size),
            target_modules=tuple(target_modules),
            module_in_features=tuple(w[0] for w in widths),
            module_out_features=tuple(w[1] for w in widths),
            r=r,
            lora_alpha=lora_alpha,
            use_rslora=use_rslora,
            lora_dropout=lora_dropout,
        )

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs) -> TargetSpec:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_dir)
        return cls.from_config(config, model_dir=model_dir, **kwargs)

    # -- derived views --------------------------------------------------------------

    @property
    def n_modules(self) -> int:
        return len(self.target_modules)

    @property
    def n_queries(self) -> int:
        """One query token per (layer, module, role): 28 x 4 x 2 = 224 by default."""
        return self.n_layers * self.n_modules * N_ROLES

    @property
    def in_features(self) -> dict[str, int]:
        return dict(zip(self.target_modules, self.module_in_features, strict=True))

    @property
    def out_features(self) -> dict[str, int]:
        return dict(zip(self.target_modules, self.module_out_features, strict=True))

    @property
    def module_to_idx(self) -> dict[str, int]:
        return {m: i for i, m in enumerate(self.target_modules)}

    @property
    def scaling(self) -> float:
        """The multiplier applied to ``B @ A``. Must match PEFT's, or hooks and any saved
        adapter would disagree -- the classic 'trains fine, evaluates at baseline' bug."""
        s = self.lora_alpha / self.r
        return s * math.sqrt(self.r) if self.use_rslora else s

    # -- query index layout ---------------------------------------------------------
    #
    # One convention, used by the query bank, the decoder heads and every test:
    #
    #     idx = layer * (n_modules * N_ROLES) + module_idx * N_ROLES + role
    #
    # It is layer-major so that all queries for a layer are contiguous.

    def query_index(self, layer: int, module: str, role: int) -> int:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range [0, {self.n_layers})")
        if role not in (ROLE_A, ROLE_B):
            raise IndexError(f"role must be {ROLE_A} (A) or {ROLE_B} (B), got {role}")
        return layer * (self.n_modules * N_ROLES) + self.module_to_idx[module] * N_ROLES + role

    def query_base_indices(self, module: str, device=None) -> torch.Tensor:
        """The ``[n_layers]`` index of the role-A query for ``module``, one per layer.

        The role-B index is this plus one. Used by the decoder heads to gather their tokens.
        """
        layers = torch.arange(self.n_layers, device=device)
        return layers * (self.n_modules * N_ROLES) + self.module_to_idx[module] * N_ROLES

    def query_indices(self, device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(layer_idx, module_idx, role_idx)``, each ``[n_queries]``, in serialization order."""
        n_mod = self.n_modules
        layer_idx = torch.arange(self.n_layers, device=device).repeat_interleave(n_mod * N_ROLES)
        module_idx = torch.arange(n_mod, device=device).repeat_interleave(N_ROLES).repeat(self.n_layers)
        role_idx = torch.arange(N_ROLES, device=device).repeat(self.n_layers * n_mod)
        return layer_idx, module_idx, role_idx

    # -- interop --------------------------------------------------------------------

    def to_lora_config(self):
        """A PEFT ``LoraConfig`` matching this spec exactly."""
        from peft import LoraConfig

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            use_rslora=self.use_rslora,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(self.target_modules),
        )

    def verify_against(self, model) -> None:
        """Cross-check the config-derived widths against a live model's actual Linear shapes."""
        from steerable_t2l.hooks import get_layers, resolve_module

        layers = get_layers(model)
        if len(layers) != self.n_layers:
            raise ValueError(f"spec says {self.n_layers} layers, model has {len(layers)}")
        for module in self.target_modules:
            linear = resolve_module(layers[0], module)
            want = (self.in_features[module], self.out_features[module])
            got = (linear.in_features, linear.out_features)
            if want != got:
                raise ValueError(f"{module}: spec says in/out {want}, model has {got}")

    def replace(self, **kwargs) -> TargetSpec:
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TargetSpec:
        d = dict(d)
        for key in ("target_modules", "module_in_features", "module_out_features"):
            d[key] = tuple(d[key])
        return cls(**d)

    def summary(self) -> str:
        rows = [
            f"{m:<10} in={self.in_features[m]:<6} out={self.out_features[m]:<6} "
            f"A={self.r * self.in_features[m]:<7} B={self.r * self.out_features[m]}"
            for m in self.target_modules
        ]
        return (
            f"TargetSpec({self.model_dir})\n"
            f"  layers={self.n_layers}  hidden={self.hidden_size}  "
            f"r={self.r}  alpha={self.lora_alpha}  scaling={self.scaling:g}\n"
            f"  queries = {self.n_layers} x {self.n_modules} x {N_ROLES} = {self.n_queries}\n"
            + "\n".join("  " + row for row in rows)
        )
