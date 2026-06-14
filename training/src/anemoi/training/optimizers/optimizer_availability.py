from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from packaging.version import Version
import torch


@dataclass(frozen=True)
class OptimizerSupport:
    optimizer: str
    available: bool
    reason: str


def check_distributed_shampoo_support() -> OptimizerSupport:
    """Report whether Meta Distributed Shampoo can run in the active environment."""
    optimizer = "distributed_shampoo"
    torch_version = Version(torch.__version__.split("+", maxsplit=1)[0])
    if torch_version < Version("2.8.0"):
        return OptimizerSupport(
            optimizer=optimizer,
            available=False,
            reason=f"torch {torch.__version__} is below torch-shampoo's declared torch>=2.8.0 requirement",
        )

    try:
        import_module("distributed_shampoo")
    except ImportError as exc:
        return OptimizerSupport(
            optimizer=optimizer,
            available=False,
            reason=f"distributed_shampoo import failed: {exc}",
        )

    return OptimizerSupport(optimizer=optimizer, available=True, reason="")
