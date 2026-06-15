# (C) Copyright 2024 Anemoi contributors.
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

LOGGER = logging.getLogger(__name__)


class InputNormalizer(BasePreprocessor):
    """Normalizes input data with a configurable method.

    This preprocessor is stateless at forward time: normalization parameters
    are computed (and cached) from the statistics carried by each
    :class:`SourceView`, not from registered buffers.
    """

    def __init__(self, config=None, **kwargs) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        """
        super().__init__(config)

        self._validate_normalization_inputs()

        # Cache for norm parameters, keyed on (variable_set, device), 2 entries: transform & inverse_transform
        self._param_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    @cached_parameters(key_fn=lambda statistics, name_to_index, device: (tuple(name_to_index.keys()), str(device)))
    def get_norm_parameters(
        self, statistics: dict, name_to_index: dict, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalization parameters, cached per (variable_set, device).

        Parameters
        ----------
        statistics : dict
            Data statistics dictionary (numpy arrays).
        name_to_index : dict
            Dictionary mapping variable names to their indices.
        device : torch.device
            Target device for the returned tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (norm_mul, norm_add) tensors of shape (len(name_to_index), ).
        """
        minimum = torch.tensor(statistics["minimum"], dtype=torch.float32, device=device)
        maximum = torch.tensor(statistics["maximum"], dtype=torch.float32, device=device)
        mean = torch.tensor(statistics["mean"], dtype=torch.float32, device=device)
        stdev = torch.tensor(statistics["stdev"], dtype=torch.float32, device=device)

        norm_mul = torch.ones(len(name_to_index), dtype=torch.float32, device=device)
        norm_add = torch.zeros(len(name_to_index), dtype=torch.float32, device=device)
        eps = 1e-8

        for name, i in name_to_index.items():
            method = self.methods.get(name, self.default)

            if method == "mean-std":
                if stdev[i] < eps:
                    warnings.warn(f"Variable {name} has near-zero variance. Skipping scale adjustments.")
                else:
                    norm_mul[i] = 1.0 / stdev[i]
                    norm_add[i] = -mean[i] / stdev[i]

            elif method == "std":
                if stdev[i] < eps:
                    warnings.warn(f"Variable {name} has near-zero variance. Skipping scale adjustments.")
                else:
                    norm_mul[i] = 1.0 / stdev[i]

            elif method == "min-max":
                rng = maximum[i] - minimum[i]
                if rng < eps:
                    warnings.warn(f"Variable {name} has a near-zero range. Skipping scale adjustments.")
                    norm_add[i] = -minimum[i]
                else:
                    norm_mul[i] = 1.0 / rng
                    norm_add[i] = -minimum[i] / rng

            elif method == "max":
                if torch.abs(maximum[i]) < eps:
                    warnings.warn(f"Variable {name} has a near-zero maximum. Skipping scale adjustments.")
                else:
                    norm_mul[i] = 1.0 / maximum[i]

            elif method == "none":
                continue
            else:
                raise ValueError(f"Unknown normalisation method for {name}: {method}")

        return norm_mul, norm_add

    def reset_cache(self) -> None:
        """Clear the cached normalization parameters.

        Call this if statistics are updated after initialization (rare).
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
            assert method in [
                "mean-std",
                "std",
                # "robust",
                "min-max",
                "max",
                "none",
            ], f"{method} is not a valid normalisation method for variable {name}."

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
                "Use SourceView.select_variables() to narrow the view before calling transform.",
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
                "Use SourceView.select_variables() to narrow the view before calling inverse_transform.",
                DeprecationWarning,
                stacklevel=2,
            )

        norm_mul, norm_add = self.get_norm_parameters(statistics, name_to_index, device=x.device)

        x = x.sub(norm_add).div(norm_mul)

        return x
