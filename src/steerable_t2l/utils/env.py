"""Environment check.

Verifies that the installed stack can actually run this project, and -- on a GPU node --
that torch was built for the local architecture. See ``docs/01_env.md``.

Usage::

    python -m steerable_t2l.utils.env

Exits nonzero if a hard requirement is missing. GPU-specific findings are reported as
warnings so the check still passes on a CPU login node.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import sys
from dataclasses import dataclass

# (import name, distribution name, minimum version) -- keep in sync with pyproject.toml
REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("torch", "torch", "2.9"),
    ("transformers", "transformers", "5.0"),
    ("peft", "peft", "0.18"),
    ("accelerate", "accelerate", "1.11"),
    ("datasets", "datasets", "4.0"),
)

OPTIONAL: tuple[tuple[str, str], ...] = (
    ("kernels", "hub-loaded flash-attn via attn_implementation='kernels-community/flash-attn'"),
    ("flash_attn", "locally built flash-attn via attn_implementation='flash_attention_2'"),
    ("wandb", "run logging"),
    ("vllm", "generation-based eval (not on the critical path)"),
)

# Blackwell / B200.
TARGET_ARCH = "sm_100"


def _parse(version: str) -> tuple[int, ...]:
    """Loose version parse: leading numeric components only ('2.9.0+cu128' -> (2, 9, 0))."""
    head = version.split("+")[0]
    out: list[int] = []
    for part in head.split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        out.append(int(digits))
    return tuple(out) or (0,)


@dataclass
class Finding:
    ok: bool
    line: str
    fatal: bool = False


def _check_packages() -> list[Finding]:
    findings: list[Finding] = []
    for import_name, dist_name, minimum in REQUIRED:
        try:
            importlib.import_module(import_name)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            findings.append(Finding(False, f"{dist_name:<14} MISSING ({exc})", fatal=True))
            continue
        try:
            found = md.version(dist_name)
        except md.PackageNotFoundError:
            found = getattr(importlib.import_module(import_name), "__version__", "unknown")
        ok = found == "unknown" or _parse(found) >= _parse(minimum)
        note = "" if ok else f"  <-- needs >={minimum}"
        findings.append(Finding(ok, f"{dist_name:<14} {found}{note}", fatal=not ok))
    return findings


def _check_optional() -> list[Finding]:
    findings: list[Finding] = []
    for import_name, purpose in OPTIONAL:
        try:
            importlib.import_module(import_name)
        except Exception:  # noqa: BLE001
            findings.append(Finding(True, f"{import_name:<14} not installed  ({purpose})"))
        else:
            try:
                found = md.version(import_name.replace("_", "-"))
            except md.PackageNotFoundError:
                found = "present"
            findings.append(Finding(True, f"{import_name:<14} {found}"))
    return findings


def _built_arches() -> set[str]:
    """Architectures this torch wheel has kernels for.

    ``torch.cuda.get_arch_list()`` returns [] without a CUDA driver, so it is useless on a
    CPU login node -- exactly where we most want to catch a wrong wheel. ``_cuda_getArchFlags``
    reads the same information straight out of the binary and works driver-free.
    """
    import torch

    try:
        flags = torch._C._cuda_getArchFlags()  # e.g. "sm_75 sm_80 ... sm_100 sm_120"
    except Exception:  # noqa: BLE001
        flags = None
    if flags:
        return set(flags.split())
    return set(torch.cuda.get_arch_list())


def _check_torch_device() -> list[Finding]:
    import torch

    arches = _built_arches()
    cuda_ver = torch.version.cuda or "none"
    findings = [
        Finding(True, f"torch cuda: {cuda_ver}   built for: {' '.join(sorted(arches)) or 'cpu only'}")
    ]

    if TARGET_ARCH not in arches:
        findings.append(
            Finding(
                False,
                f"WARNING: this torch wheel has no {TARGET_ARCH} (B200) kernels. "
                "Reinstall from the cu128 index before running on a B200.",
            )
        )

    if not torch.cuda.is_available():
        findings.append(
            Finding(True, "CUDA not available -- CPU node. Model construction and tests still run.")
        )
        return findings

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    cap = f"sm_{props.major}{props.minor}"
    findings.append(
        Finding(True, f"device 0: {props.name}  {cap}  {props.total_memory / 1024**3:.0f} GiB")
    )

    if cap not in arches:
        findings.append(
            Finding(
                False,
                f"WARNING: torch has no {cap} kernels (has: {sorted(arches)}). "
                "Expect 'no kernel image is available' at runtime. Reinstall torch from the cu128 index.",
            )
        )
    elif cap != TARGET_ARCH:
        findings.append(Finding(True, f"note: running on {cap}, not the target {TARGET_ARCH} (B200)"))
    return findings


def check_env(verbose: bool = True) -> bool:
    """Return True if every hard requirement is satisfied. Prints a report when ``verbose``."""
    sections: list[tuple[str, list[Finding]]] = [("required", _check_packages())]

    # Only probe the device once torch is known to import.
    if all(f.ok for f in sections[0][1]):
        sections.append(("torch / device", _check_torch_device()))
    sections.append(("optional", _check_optional()))

    if verbose:
        print(f"python         {sys.version.split()[0]}  ({sys.executable})")
        for title, findings in sections:
            print(f"\n[{title}]")
            for f in findings:
                print(f"  {'ok ' if f.ok else 'FAIL'} {f.line}")

    fatal = [f for _, fs in sections for f in fs if f.fatal]
    if verbose:
        print()
        print("environment OK" if not fatal else f"environment NOT OK ({len(fatal)} problem(s))")
    return not fatal


def main() -> int:
    return 0 if check_env() else 1


if __name__ == "__main__":
    raise SystemExit(main())
