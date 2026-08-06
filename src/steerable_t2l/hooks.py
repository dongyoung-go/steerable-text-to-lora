"""Differentiable injection of generated LoRA weights into a frozen target model.

Forward hooks on the target's ``nn.Linear`` modules, closing over tensors that are still
attached to the hypernetwork's autograd graph. So the target model's loss backpropagates
through the generated ``A``/``B`` into the hypernetwork, without a single gradient update on
the target itself.

Because ``A`` and ``B`` carry a batch dimension, **every sample in the batch gets its own
LoRA** -- which is what makes multi-task batching possible.

See ``docs/02_model.md``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from operator import attrgetter

import torch
import torch.nn.functional as F
from torch import nn

_ATTN_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_MLP_MODULES = frozenset({"gate_proj", "up_proj", "down_proj"})


def get_layers(model: nn.Module) -> nn.ModuleList:
    """The decoder-layer ``ModuleList`` of ``model``, unwrapping PEFT and CausalLM wrappers."""
    base = model
    if hasattr(base, "get_base_model"):  # PeftModel
        base = base.get_base_model()
    if hasattr(base, "get_decoder"):
        try:
            base = base.get_decoder()
        except (AttributeError, NotImplementedError):  # pragma: no cover
            pass

    layers = getattr(base, "layers", None)
    if isinstance(layers, nn.ModuleList) and len(layers):
        return layers

    for module in base.modules():  # pragma: no cover - fallback for unusual architectures
        if isinstance(module, nn.ModuleList) and len(module) and hasattr(module[0], "self_attn"):
            return module

    raise ValueError(f"could not locate the decoder layers of {type(model).__name__}")


def qualified_name(module_name: str) -> str:
    """``'q_proj'`` -> ``'self_attn.q_proj'``: the path from a decoder layer to the Linear."""
    if module_name in _ATTN_MODULES:
        return f"self_attn.{module_name}"
    if module_name in _MLP_MODULES:
        return f"mlp.{module_name}"
    return module_name


def resolve_module(layer: nn.Module, module_name: str) -> nn.Linear:
    """The ``nn.Linear`` named ``module_name`` inside one decoder ``layer``."""
    try:
        return attrgetter(qualified_name(module_name))(layer)
    except AttributeError as exc:
        raise AttributeError(f"{type(layer).__name__} has no {qualified_name(module_name)}") from exc


@dataclass(frozen=True)
class Site:
    """One injection point: LoRA ``A``/``B`` for a single (layer, module), batched per sample.

    ``A`` is ``[bs, in_features, r]`` and ``B`` is ``[bs, r, out_features]`` -- i.e. already
    transposed into the orientation the hook multiplies in.
    """

    layer_idx: int
    module_name: str
    A: torch.Tensor
    B: torch.Tensor


def build_sites(
    spec,
    per_module: dict[str, tuple[torch.Tensor, torch.Tensor]],
    layer_indices: Sequence[int] | None = None,
) -> list[Site]:
    """Turn the hypernetwork's output into a flat list of injection sites.

    ``per_module`` maps a module name to ``(A, B)`` with ``A: [bs, n_layers, r, in_features]``
    and ``B: [bs, n_layers, out_features, r]`` -- the natural layout of ``heads_forward``.

    No layer-major flattening contract is needed (unlike the reference implementation, whose
    slicing silently assumes the requested layers are exactly ``arange(n_layers)``): the
    layer axis is explicit, so we just index it.
    """
    layers = range(spec.n_layers) if layer_indices is None else layer_indices
    sites: list[Site] = []
    for module_name, (A, B) in per_module.items():
        if A.dim() != 4 or B.dim() != 4:
            raise ValueError(
                f"{module_name}: expected A [bs, n_layers, r, in] and B [bs, n_layers, out, r], "
                f"got {tuple(A.shape)} and {tuple(B.shape)}"
            )
        for layer_idx in layers:
            sites.append(
                Site(
                    layer_idx=int(layer_idx),
                    module_name=module_name,
                    A=A[:, layer_idx].transpose(-1, -2),  # [bs, r, in]  -> [bs, in, r]
                    B=B[:, layer_idx].transpose(-1, -2),  # [bs, out, r] -> [bs, r, out]
                )
            )
    return sites


def _make_hook(A: torch.Tensor, B: torch.Tensor, scaling: float, dropout: float):
    """Post-hook returning ``W x + scaling * (x A) B``.

    ``args[0]`` is the wrapped Linear's input ``x`` and ``output`` is ``W x``.

    Note the absence of any ``repeat_interleave``. The reference implementation expands ``A``
    to ``[bs * seq_len, in_features, r]`` so it can use a per-token ``bmm``; that tensor is
    kept alive for backward and costs ~201 MB per site at bs=8, seq=1024 for ``q_proj``
    (~22 GB across 28 layers x 4 modules) for data that is ``seq_len``-fold redundant.
    ``torch.bmm`` already broadcasts over the leading batch dimension, so the only extra
    tensor saved here is ``[bs, seq_len, r]`` -- about 131 KB per site.
    """

    def hook(module: nn.Module, args: tuple, output):
        out = output[0] if isinstance(output, tuple) else output
        x = args[0]
        if dropout:
            x = F.dropout(x, dropout, module.training)
        # [bs, seq, in] @ [bs, in, r] -> [bs, seq, r] @ [bs, r, out] -> [bs, seq, out]
        delta = torch.bmm(torch.bmm(x.to(A.dtype), A), B) * scaling
        new_out = out + delta.to(out.dtype)
        return (new_out, *output[1:]) if isinstance(output, tuple) else new_out

    return hook


@contextmanager
def lora_hooks(
    model: nn.Module,
    sites: Sequence[Site],
    scaling: float,
    dropout: float = 0.0,
) -> Iterator[None]:
    """Attach per-sample LoRA hooks for the duration of the block.

    ``try/finally`` matters: if the wrapped forward raises, leaked handles would silently
    corrupt every subsequent step. Removing the handles after the forward is safe even
    though ``backward`` has not run yet -- a handle only controls registration, and the
    autograd graph already holds references to the closed-over ``A``/``B``.

        with lora_hooks(target, sites, spec.scaling):
            loss = target(**batch).loss
        loss.backward()
    """
    layers = get_layers(model)
    handles = []
    try:
        for site in sites:
            linear = resolve_module(layers[site.layer_idx], site.module_name)
            handles.append(linear.register_forward_hook(_make_hook(site.A, site.B, scaling, dropout)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def delta_weights(A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    """The explicit ``[..., out_features, in_features]`` update, for tests and canonicalization.

    ``A`` is ``[..., r, in_features]`` and ``B`` is ``[..., out_features, r]`` (PEFT layout).
    """
    return (B @ A) * scaling
