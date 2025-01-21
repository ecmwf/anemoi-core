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

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class InputNormalizer(BasePreprocessor):
    """Normalizes input data with a configurable method."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__(config, data_indices, statistics)

        name_to_index_training_input = self.data_indices.data.input.name_to_index

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]
        mean = statistics["mean"]
        stdev = statistics["stdev"]

        # Optionally reuse statistic of one variable for another variable
        statistics_remap = {}
        for remap, source in self.remap.items():
            idx_src, idx_remap = name_to_index_training_input[source], name_to_index_training_input[remap]
            statistics_remap[idx_remap] = (minimum[idx_src], maximum[idx_src], mean[idx_src], stdev[idx_src])

        # Two-step to avoid overwriting the original statistics in the loop (this reduces dependence on order)
        for idx, new_stats in statistics_remap.items():
            minimum[idx], maximum[idx], mean[idx], stdev[idx] = new_stats

        self._validate_normalization_inputs(name_to_index_training_input, minimum, maximum, mean, stdev)

        _norm_add = np.zeros((minimum.size,), dtype=np.float32)
        _norm_mul = np.ones((minimum.size,), dtype=np.float32)

        for name, i in name_to_index_training_input.items():
            method = self.methods.get(name, self.default)

            if method == "mean-std":
                LOGGER.debug(f"Normalizing: {name} is mean-std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = -mean[i] / stdev[i]

            elif method == "std":
                LOGGER.debug(f"Normalizing: {name} is std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = 0

            elif method == "min-max":
                LOGGER.debug(f"Normalizing: {name} is min-max-normalised to [0, 1].")
                x = maximum[i] - minimum[i]
                if x < 1e-9:
                    warnings.warn(f"Normalizing: the field {name} seems to have only one value {maximum[i]}.")
                _norm_mul[i] = 1 / x
                _norm_add[i] = -minimum[i] / x

            elif method == "max":
                LOGGER.debug(f"Normalizing: {name} is max-normalised to [0, 1].")
                _norm_mul[i] = 1 / maximum[i]

            elif method == "none":
                LOGGER.info(f"Normalizing: {name} is not normalized.")

            else:
                raise ValueError[f"Unknown normalisation method for {name}: {method}"]

        # register buffer - this will ensure they get copied to the correct device(s)
        self.register_buffer("_norm_mul", torch.from_numpy(_norm_mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(_norm_add), persistent=True)
        self.register_buffer("_input_idx", data_indices.data.input.full, persistent=True)
        self.register_buffer("_output_idx", self.data_indices.data.output.full, persistent=True)

    def _validate_normalization_inputs(self, name_to_index_training_input: dict, minimum, maximum, mean, stdev):
        assert len(self.methods) == sum(len(v) for v in self.method_config.values()), (
            f"Error parsing methods in InputNormalizer methods ({len(self.methods)}) "
            f"and entries in config ({sum(len(v) for v in self.method_config)}) do not match."
        )

        # Check that all sizes align
        n = minimum.size
        assert maximum.size == n, (maximum.size, n)
        assert mean.size == n, (mean.size, n)
        assert stdev.size == n, (stdev.size, n)

        # Check for typos in method config
        assert isinstance(self.methods, dict)
        for name, method in self.methods.items():
            assert name in name_to_index_training_input, f"{name} is not a valid variable name"
            assert method in [
                "mean-std",
                "std",
                # "robust",
                "min-max",
                "max",
                "none",
            ], f"{method} is not a valid normalisation method"

    def transform(
        self, x: torch.Tensor, in_place: bool = True, data_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normalizes an input tensor x of shape [..., nvars].

        Normalization done in-place unless specified otherwise.

        The default usecase either assume the full batch tensor or the full input tensor.
        A dataindex is based on the full data can be supplied to choose which variables to normalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to normalize
        in_place : bool, optional
            Normalize in-place, by default True
        data_index : Optional[torch.Tensor], optional
            Normalize only the specified indices, by default None

        Returns
        -------
        torch.Tensor
            _description_
        """
        if not in_place:
            x = x.clone()

        if data_index is not None:
            x.mul_(self._norm_mul[data_index]).add_(self._norm_add[data_index])
        elif x.shape[-1] == len(self._input_idx):
            x.mul_(self._norm_mul[self._input_idx]).add_(self._norm_add[self._input_idx])
        else:
            x.mul_(self._norm_mul).add_(self._norm_add)

        return x

    def inverse_transform(
        self, x: torch.Tensor, in_place: bool = True, data_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Denormalizes an input tensor x of shape [..., nvars | nvars_pred].

        Denormalization done in-place unless specified otherwise.

        The default usecase either assume the full batch tensor or the full output tensor.
        A dataindex is based on the full data can be supplied to choose which variables to denormalise.

        Parameters
        ----------
        x : torch.Tensor
            Data to denormalize
        in_place : bool, optional
            Denormalize in-place, by default True
        data_index : Optional[torch.Tensor], optional
            Denormalize only the specified indices, by default None

        Returns
        -------
        torch.Tensor
            Denormalized data
        """
        if not in_place:
            x = x.clone()

        # Denormalize dynamic or full tensors
        # input and predicted tensors have different shapes
        # hence, we mask out the forcing indices
        if data_index is not None:
            x.subtract_(self._norm_add[data_index]).div_(self._norm_mul[data_index])
        elif x.shape[-1] == len(self._output_idx):
            x.subtract_(self._norm_add[self._output_idx]).div_(self._norm_mul[self._output_idx])
        else:
            x.subtract_(self._norm_add).div_(self._norm_mul)
        return x
