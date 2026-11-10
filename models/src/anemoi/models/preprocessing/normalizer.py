# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from typing import Optional

import numpy as np
import torch

from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing._caching import cached_parameters
from anemoi.models.preprocessing.normalization import NORMALIZATION_STATISTICS
from anemoi.models.preprocessing.normalization import missing_statistics
from anemoi.models.preprocessing.normalization import normalization_coefficients

LOGGER = logging.getLogger(__name__)


def _statistics_fingerprint(statistics: dict) -> tuple:
    """Hashable fingerprint of a statistics mapping, for keying cached normalisation parameters.

    Sources of the same variables can carry different statistics (another reader, or a
    tendency normalised per lead time), so the variable names alone do not identify the
    parameters.
    """
    return tuple(
        (key, np.asarray(value.detach().cpu() if torch.is_tensor(value) else value).tobytes())
        for key, value in sorted(statistics.items())
    )


def _norm_parameters_key(statistics: dict, name_to_index: dict, device: torch.device) -> tuple:
    return (tuple(name_to_index.keys()), str(device), _statistics_fingerprint(statistics))


class InputNormalizer(BasePreprocessor):
    """Normalizes input data with a configurable method.

    This preprocessor is stateless at forward time: normalization parameters
    are computed (and cached) from the statistics carried by each
    :class:`Source`, not from registered buffers.

    Its ``remap`` also applies to the thresholds of later processors and of the output boundings,
    see :attr:`~anemoi.models.preprocessing.BasePreprocessor.shares_statistics_remap`.
    """

    shares_statistics_remap = True

    def __init__(self, config=None, **kwargs) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        """
        super().__init__(config)

        self._validate_normalization_inputs()

        # Cache for norm parameters, keyed on (variable_set, device, statistics)
        self._param_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    @cached_parameters(key_fn=_norm_parameters_key)
    def get_norm_parameters(
        self, statistics: dict, name_to_index: dict, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalization parameters, cached per (variable_set, device, statistics).

        Parameters
        ----------
        statistics : dict
            Data statistics dictionary (numpy arrays).
        name_to_index : dict
            Dictionary mapping variable names to their indices: the
            :attr:`~anemoi.models.data.sources.Source.name_to_index` of the view being
            normalized, i.e. one entry per variable, indexed ``0..n-1`` in variable order and
            aligned with the ``statistics`` arrays. A sparse or model-space mapping (e.g. from
            ``data_indices``) is not valid here.
        device : torch.device
            Target device for the returned tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (norm_mul, norm_add) tensors of shape (len(name_to_index), ).
        """
        methods = {self.methods.get(name, self.default) for name in name_to_index}
        unknown = methods - NORMALIZATION_STATISTICS.keys()
        if unknown:
            raise ValueError(f"Unknown normalisation methods: {sorted(unknown)}")
        missing = missing_statistics(sorted(methods), statistics)
        if missing:
            raise ValueError(f"Normalization requires statistics: {missing}")
        required = {key for method in methods for key in NORMALIZATION_STATISTICS[method]}
        # Computed in float64 and rounded once to float32 at the end: rounding the statistics first
        # changes the last bit of many parameters.
        values = {key: torch.as_tensor(statistics[key], dtype=torch.float64, device=device) for key in required}
        for key, value in values.items():
            if value.ndim != 1 or value.shape[0] != len(name_to_index):
                raise ValueError(f"Statistic {key!r} must contain one value per variable.")

        # Optionally reuse the statistics of one variable for another (e.g. cp, sf -> tp, so that a
        # FractionBounding of cp by tp is a physical fraction). Every remapped variable takes the
        # source's original statistics, so the result does not depend on the order of the mapping.
        stats_index = list(range(len(name_to_index)))
        for name, source in self.remap.items():
            if name not in name_to_index:
                continue
            if source not in name_to_index:
                msg = f"Cannot normalise {name!r} with the statistics of {source!r}: {source!r} is not in this view."
                raise ValueError(msg)
            stats_index[name_to_index[name]] = name_to_index[source]
        values = {key: value[stats_index] for key, value in values.items()}

        norm_mul = torch.ones(len(name_to_index), dtype=torch.float64, device=device)
        norm_add = torch.zeros(len(name_to_index), dtype=torch.float64, device=device)
        for name, i in name_to_index.items():
            method = self.methods.get(name, self.default)
            norm_mul[i], norm_add[i] = normalization_coefficients(method, values, i, name)

        return norm_mul.to(torch.float32), norm_add.to(torch.float32)

    def reset_cache(self) -> None:
        """Clear the cached normalization parameters.

        Entries are keyed on the statistics, so new statistics never reuse stale parameters;
        this only frees the cached tensors.
        """
        self._param_cache.clear()

    def _validate_normalization_inputs(self):
        assert len(self.methods) == sum(len(v) for v in self.method_config.values()), (
            f"Error parsing methods in InputNormalizer methods ({len(self.methods)}) "
            f"and entries in config ({sum(len(v) for v in self.method_config)}) do not match."
        )

        # Check for typos in method config
        assert isinstance(self.methods, dict)
        for name, method in self.methods.items():
            assert method in NORMALIZATION_STATISTICS, (
                f"{method} is not a valid normalisation method for variable {name}."
            )

    def transform(
        self,
        x: torch.Tensor,
        statistics: Optional[dict[str, np.ndarray]] = None,
        name_to_index: Optional[dict[str, int]] = None,
        data_index: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Normalize a tensor in the variables dimension.

        Parameters
        ----------
        x : torch.Tensor
            Data to normalize.
        statistics : dict[str, np.ndarray]
            Statistics dictionary required for normalization.
        name_to_index : dict[str, int]
            Dictionary mapping variable names to their indices, required for normalization.
            Normally the view's :attr:`~anemoi.models.data.sources.Source.name_to_index`, passed in by
            :meth:`~anemoi.models.data.sources.Source.apply_func`; see :meth:`get_norm_parameters`.
        data_index : torch.Tensor, optional
            Unused deprecated argument. Select variables on the source view before normalization.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        assert statistics is not None, "Statistics must be provided for normalization."
        assert name_to_index is not None, "name_to_index must be provided for normalization."

        if data_index is not None:
            warnings.warn(
                "The 'data_index' parameter is deprecated and will be removed in a future release. "
                "Use Source.select_variables() to narrow the view before calling transform.",
                DeprecationWarning,
                stacklevel=2,
            )

        norm_mul, norm_add = self.get_norm_parameters(statistics, name_to_index, device=x.device)

        x = x.mul(norm_mul).add(norm_add)

        return x

    def inverse_transform(
        self,
        x: torch.Tensor,
        statistics: Optional[dict[str, np.ndarray]] = None,
        name_to_index: Optional[dict[str, int]] = None,
        data_index: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Denormalize a tensor in the variables dimension.

        Parameters
        ----------
        x : torch.Tensor
            Data to denormalize.
        statistics : dict[str, np.ndarray]
            Statistics dictionary required for normalization.
        name_to_index : dict[str, int]
            Dictionary mapping variable names to their indices, required for normalization.
            Normally the view's :attr:`~anemoi.models.data.sources.Source.name_to_index`, passed in by
            :meth:`~anemoi.models.data.sources.Source.apply_func`; see :meth:`get_norm_parameters`.
        data_index : torch.Tensor, optional
            Unused deprecated argument. Select variables on the source view before normalization.

        Returns
        -------
        torch.Tensor
            Denormalized tensor.
        """
        assert statistics is not None, "Statistics must be provided for normalization."
        assert name_to_index is not None, "name_to_index must be provided for normalization."

        if data_index is not None:
            warnings.warn(
                "The 'data_index' parameter is deprecated and will be removed in a future release. "
                "Use Source.select_variables() to narrow the view before calling inverse_transform.",
                DeprecationWarning,
                stacklevel=2,
            )

        norm_mul, norm_add = self.get_norm_parameters(statistics, name_to_index, device=x.device)

        x = x.sub(norm_add).div(norm_mul)

        return x
