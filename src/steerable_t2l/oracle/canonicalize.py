"""SVD canonicalization of independently trained oracle LoRAs.

See ``docs/03_training_validation.md`` §2. ``ΔW = B @ A`` is not unique: for any invertible
``R``, ``(BR)(R⁻¹A)`` computes the same function. Independently trained oracles land in
arbitrary, mutually inconsistent bases, so regressing onto raw ``A``/``B`` (as
``trainers/recon.py`` needs to) is ill-posed until every adapter is canonicalized first.

⚠️ Convention flip. ``Design.md`` writes ``ΔW = A·B`` with ``A_canon = UΣ^½``,
``B_canon = Σ^½Vᵀ``. PEFT uses ``ΔW = lora_B · lora_A`` with ``lora_A: [r, in]``,
``lora_B: [out, r]``, so the mapping used here is SWAPPED::

    lora_B_canon = U  Σ^½        # [out, r]
    lora_A_canon = Σ^½ Vᵀ        # [r, in]

Getting this backwards silently "works" on the square ``q_proj``/``o_proj`` and trains on
garbage. ``hooks.delta_weights`` is the reference to check round-trips against.
"""

from __future__ import annotations

import re

import torch

from steerable_t2l.hooks import qualified_name
from steerable_t2l.target_spec import TargetSpec

PEFT_KEY_PREFIX = "base_model.model.model.layers"
_KEY_RE = re.compile(r"\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.(lora_A|lora_B)\.weight$")


def fix_svd_signs(U: torch.Tensor, Vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic sign gauge: make the largest-magnitude entry of each left singular
    vector positive. ``(u_i, v_i)`` and ``(-u_i, -v_i)`` are both valid SVD components --
    without a rule, two oracles encoding the *same* function land on opposite-sign targets,
    defeating the point of canonicalizing at all. ``U``: ``[out, r]``, ``Vh``: ``[r, in]``.
    """
    r = U.shape[1]
    idx = U.abs().argmax(dim=0)  # [r] -- row of the largest-magnitude entry per column
    signs = torch.sign(U[idx, torch.arange(r, device=U.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return U * signs[None, :], Vh * signs[:, None]


def canonicalize_adapter(
    A: torch.Tensor, B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``A``: ``[r, in]`` (PEFT ``lora_A``), ``B``: ``[out, r]`` (PEFT ``lora_B``), for one
    ``(layer, module)`` -- fully independent of every other ``(layer, module)`` pair, since
    the non-uniqueness (``ΔW = BA`` for any invertible ``R``) holds per-adapter.

    Returns ``(A_canon [r, in], B_canon [out, r], S [r])`` -- same PEFT layout as the input,
    plus the singular-value spectrum for logging (near-tied values -> unstable rotation in
    the tied subspace).

    Exploits the existing rank-``r`` factorization for an exact ``O(d r²)`` result instead of
    a full ``[out, in]`` SVD, in float64 (``Rb @ Ra.T`` is a product of two ill-conditioned
    triangular factors).
    """
    Qb, Rb = torch.linalg.qr(B.double())
    Qa, Ra = torch.linalg.qr(A.double().T)
    U0, S, Vh0 = torch.linalg.svd(Rb @ Ra.T)
    U, Vh = fix_svd_signs(Qb @ U0, Vh0 @ Qa.T)
    s = S.clamp_min(0).sqrt()
    A_canon = (s[:, None] * Vh).float()
    B_canon = (U * s[None, :]).float()
    return A_canon, B_canon, S


def canonicalize_state_dict(
    state_dict: dict, spec: TargetSpec
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Canonicalize every ``(layer, module)`` ``lora_A``/``lora_B`` pair in a PEFT state dict.

    Reuses ``SteerableHyperLoRA.to_peft_state_dict``'s key convention
    (``base_model.model.model.layers.{l}.{self_attn|mlp}.{module}.lora_A/B.weight``) so
    canonicalized oracles and the hypernet's own output speak the same key scheme. Does
    **not** rescale by ``spec.scaling`` -- both sides use the same PEFT config (asserted at
    oracle-training time), so canonicalizing the raw ``B @ A`` is consistent.
    """
    per_pair: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        match = _KEY_RE.search(key)
        if not match:
            continue
        layer_idx, module_name, role = match.groups()
        if module_name not in spec.target_modules:
            continue
        per_pair.setdefault((int(layer_idx), module_name), {})[role] = tensor

    canon: dict[str, torch.Tensor] = {}
    spectra: dict[str, torch.Tensor] = {}
    for (layer_idx, module_name), pair in per_pair.items():
        A_canon, B_canon, S = canonicalize_adapter(pair["lora_A"], pair["lora_B"])
        stem = f"{PEFT_KEY_PREFIX}.{layer_idx}.{qualified_name(module_name)}"
        canon[f"{stem}.lora_A.weight"] = A_canon
        canon[f"{stem}.lora_B.weight"] = B_canon
        spectra[f"{module_name}.{layer_idx}"] = S
    return canon, spectra


def load_and_canonicalize_oracle(
    oracle_dir: str, spec: TargetSpec, device: str | torch.device = "cpu"
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """``peft.load_peft_weights(oracle_dir)`` -> canonicalize -> per-module, per-layer-stacked
    ``{module: (A [n_layers, r, in], B [n_layers, out, r])}`` -- the layout ``trainers/recon.py``
    needs as its regression target (the same layout ``heads_forward``'s output has, minus the
    batch dim).

    ``device`` is passed explicitly and defaults to ``"cpu"`` -- ``peft.load_peft_weights``
    otherwise calls ``infer_device()`` internally and silently lands on ``"cuda"`` whenever one
    is visible, which then mismatches a CPU target/hypernet (works by accident on a CPU-only
    node, breaks the moment a GPU is present). Callers on a GPU node must pass the target's or
    hypernet's actual device so the returned tensors line up with what they'll be used against.
    """
    import peft

    raw = peft.load_peft_weights(str(oracle_dir), device=str(device))
    canon, _ = canonicalize_state_dict(raw, spec)

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for module in spec.target_modules:
        A_layers, B_layers = [], []
        for layer in range(spec.n_layers):
            stem = f"{PEFT_KEY_PREFIX}.{layer}.{qualified_name(module)}"
            A_layers.append(canon[f"{stem}.lora_A.weight"])
            B_layers.append(canon[f"{stem}.lora_B.weight"])
        out[module] = (torch.stack(A_layers), torch.stack(B_layers))
    return out
