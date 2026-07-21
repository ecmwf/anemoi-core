# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import math
from typing import Any

from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.utils.parametrisation import DictParametrisation
from anemoi.utils.parametrisation import ParametrisationError

LOGGER = logging.getLogger(__name__)

# Default kernels used when the layer_kernels config omits an entry (falls back to torch.nn).
_DEFAULT_KERNELS = {
    "Linear": {"_target_": "torch.nn.Linear"},
    "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
    "Activation": {"_target_": "torch.nn.GELU"},
    "QueryNorm": {
        "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
        "_partial_": True,
        "bias": False,
    },
    "KeyNorm": {
        "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
        "_partial_": True,
        "bias": False,
    },
}


class LayerKernels(DictParametrisation):
    """A :class:`~anemoi.utils.parametrisation.Parametrisation` of layer kernels.

    Holds the (JSON-serialisable) kernel config and exposes the built partial factories by
    name via attribute or item access, so leaf layers keep using ``layer_kernels.Linear(...)``
    / ``layer_kernels["QueryNorm"]`` unchanged. Construction is Hydra-free (it uses the
    ``DictParametrisation`` engine), keeping ``anemoi.models`` importable at inference.

    Parameters
    ----------
    kernel_config : Parametrisation | dict, optional
        Kernel configuration, e.g. ``{"Linear": {"_target_": "torch.nn.Linear"}}``. Missing
        entries fall back to :data:`_DEFAULT_KERNELS`.
    instance : bool
        If True (default), build each kernel into a partial factory; if False, keep the raw
        config entries (useful for testing).
    """

    def __init__(self, kernel_config: Any = None, instance: bool = True) -> None:
        if isinstance(kernel_config, DictParametrisation):
            kernel_config = kernel_config.to_dict()
        merged = {**_DEFAULT_KERNELS, **(dict(kernel_config) if kernel_config else {})}
        super().__init__(merged)

        factories: dict[str, Any] = {}
        for name, entry in merged.items():
            if not instance:
                factories[name] = entry
                continue
            try:
                factories[name] = self.create_module(entry, _partial_=True)
            except ParametrisationError:
                LOGGER.info(
                    f"{entry['_target_']} not found! Check your model.layer_kernels {name} entry. "
                    "Maybe the kernel is not installed or the import string is incorrect?"
                )
                raise
            LOGGER.info(f"{name} kernel: {entry['_target_']}.")
        object.__setattr__(self, "_factories", factories)

    def __getattr__(self, name: str) -> Any:
        # Only fires for attributes not found normally; expose kernels, guard internals.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.__dict__["_factories"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> Any:
        return self._factories[name]

    def __contains__(self, name: str) -> bool:
        return name in self._factories


def compute_mlp_hidden_dim(num_channels: int, mlp_hidden_ratio: float) -> int:
    """Compute integer hidden dimension from a (possibly fractional) MLP ratio.

    Parameters
    ----------
    num_channels : int
        Base channel width.
    mlp_hidden_ratio : float
        Multiplier used to derive hidden width.

    Returns
    -------
    int
        Rounded hidden width.
    """
    if not math.isfinite(mlp_hidden_ratio):
        msg = f"`mlp_hidden_ratio` must be finite, got {mlp_hidden_ratio}."
        raise ValueError(msg)
    if mlp_hidden_ratio <= 0:
        msg = f"`mlp_hidden_ratio` must be > 0, got {mlp_hidden_ratio}."
        raise ValueError(msg)

    # Round to nearest integer with halves rounding up.
    hidden_dim = int(num_channels * mlp_hidden_ratio + 0.5)
    if hidden_dim <= 0:
        msg = f"Computed hidden_dim must be > 0, got {hidden_dim}."
        raise ValueError(msg)
    return hidden_dim


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


def maybe_checkpoint(func, enabled: bool, *args, **kwargs):
    """Conditionally apply gradient checkpointing to a function.

    Parameters
    ----------
    func : callable
        The function to potentially wrap with checkpointing
    enabled : bool
        Whether to apply gradient checkpointing
    *args, **kwargs
        Arguments to pass to the function

    Returns
    -------
    The result of calling func with the provided arguments
    """
    if enabled:
        return checkpoint(func, *args, **kwargs, use_reentrant=False)
    return func(*args, **kwargs)


def load_layer_kernels(kernel_config: Any = None, instance: bool = True) -> LayerKernels:
    """Load layer kernels as a :class:`LayerKernels` parametrisation.

    Missing entries fall back to the ``torch.nn`` defaults. The returned object exposes the
    built partial factories by name (``kernels.Linear(...)`` / ``kernels["QueryNorm"]``).

    Parameters
    ----------
    kernel_config : Parametrisation | dict, optional
        Kernel configuration, e.g. ``{"Linear": {"_target_": "torch.nn.Linear"}}``.
    instance : bool
        If True (default), build the kernels; if False, keep the raw config (for testing).

    Returns
    -------
    LayerKernels
        Parametrisation exposing the layer-kernel factories.
    """
    return LayerKernels(kernel_config, instance=instance)
