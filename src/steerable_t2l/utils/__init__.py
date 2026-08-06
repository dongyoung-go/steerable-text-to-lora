"""Shared utilities.

``check_env`` is imported lazily so that ``python -m steerable_t2l.utils.env`` does not
trip runpy's "found in sys.modules before execution" warning.
"""

__all__ = ["check_env"]


def __getattr__(name: str):
    if name == "check_env":
        from steerable_t2l.utils.env import check_env

        return check_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
