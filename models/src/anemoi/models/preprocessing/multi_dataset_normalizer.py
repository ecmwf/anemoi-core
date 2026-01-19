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
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)
class TopNormalizer(BasePreprocessor):
    """Top-level normalizer for input, output data."""

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotConfig
            Configuration object
        statistics : Dicts
            Statistics for input, output data
        data_indices : dict
        """
        super().__init__(config, statistics, data_indices)

        self.normalizers = {}
        # this two-step process is done to allow for casting to the correct device
        # alternative is to install tensordict, or use ModuleList
        self.normalizer_input = FieldNormalizer(
            config=config,
            statistics=statistics[0],
            data_indices_ds=data_indices.data.input[0],
            dataset="input_lres",
        )
        self.normalizers["input_lres"] = self.normalizer_input 

        self.normalizer_input_hres = FieldNormalizer(
            config=config,
            statistics=statistics[1],
            data_indices_ds=data_indices.data.input[1],
            dataset="input_hres",
        )
        self.normalizers["input_hres"] = self.normalizer_input_hres
        self.normalizer_output = FieldNormalizer(
            config=config,
            statistics=statistics[2],
            data_indices_ds=data_indices.data.output,
            dataset="output",
        )
        self.normalizers["output"] = self.normalizer_output

    def forward(
        self,
        x: torch.Tensor,
        dataset: str,
        in_place: bool = True,
        inverse: bool = False,
    ) -> torch.Tensor:
        if inverse:
            return self.inverse_transform(x, dataset, in_place=in_place)
        return self.transform(x, dataset, in_place=in_place)

    def transform(self, data, dataset: str, in_place: bool = True):
        try:
            normalizer = self.normalizers[dataset]
        except ValueError:
            raise ValueError(f"No normalizer found for dataset type: {dataset}")
        return normalizer.transform(data, in_place=in_place)

    def inverse_transform(
        self,
        data,
        dataset: str,
        in_place: bool = True,
        data_index: Optional[torch.Tensor] = None,
    ):
        try:
            normalizer = self.normalizers[dataset]
        except ValueError:
            raise ValueError(f"No normalizer found for dataset type: {dataset}")
        return normalizer.inverse_transform(
            data, in_place=in_place, data_index=data_index
        )


class FieldNormalizer(nn.Module):
    """Normalizes data."""

    def __init__(
        self,
        *,
        config,
        statistics: dict,
        data_indices_ds: dict,
        dataset: str,
    ) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        zarr_metadata : Dict
            Zarr metadata dictionary
        """
        super().__init__()

        default = config["default"]

        self.dataset = dataset

        name_to_index = data_indices_ds.name_to_index

        methods = {
            variable: method
            for method, variables in config.items()
            if isinstance(variables, list)
            for variable in variables
        }

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]
        mean = statistics["mean"]
        stdev = statistics["stdev"]

        n = minimum.size
        assert maximum.size == n, (maximum.size, n)
        assert mean.size == n, (mean.size, n)
        assert stdev.size == n, (stdev.size, n)

        assert isinstance(methods, dict)
        for name, method in methods.items():
            # assert name in name_to_index, f"{name} is not a valid variable name"
            assert method in [
                "mean-std",
                "std",
                # "robust",
                "min-max",
                "max",
                "none",
            ], f"{method} is not a valid normalisation method"

        _norm_add = np.zeros((n,), dtype=np.float32)
        _norm_mul = np.ones((n,), dtype=np.float32)

        for name, i in name_to_index.items():
            m = methods.get(name, default)
            if m == "mean-std":
                LOGGER.debug(f"Normalizing: {name} is mean-std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(
                        f"Normalizing: the field seems to have only one value {mean[i]}"
                    )
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = -mean[i] / stdev[i]

            elif m == "std":
                LOGGER.debug(f"Normalizing: {name} is std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(
                        f"Normalizing: the field seems to have only one value {mean[i]}"
                    )
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = 0

            elif m == "min-max":
                LOGGER.debug(f"Normalizing: {name} is min-max-normalised to [0, 1].")
                x = maximum[i] - minimum[i]
                if x < 1e-9:
                    warnings.warn(
                        f"Normalizing: the field {name} seems to have only one value {maximum[i]}."
                    )
                _norm_mul[i] = 1 / x
                _norm_add[i] = -minimum[i] / x

            elif m == "max":
                LOGGER.debug(f"Normalizing: {name} is max-normalised to [0, 1].")
                _norm_mul[i] = 1 / maximum[i]

            elif m == "none":
                LOGGER.info(f"Normalizing: {name} is not normalized.")

            else:
                raise ValueError[f"Unknown normalisation method for {name}: {m}"]

        # register buffer - this will ensure they get copied to the correct device(s)
        self.register_buffer("_norm_mul", torch.from_numpy(_norm_mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(_norm_add), persistent=True)
        self.register_buffer(
            f"_{self.dataset}_idx", data_indices_ds.full, persistent=True
        )

    def transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        data_index: Optional[torch.Tensor] = None,
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

        dataset_idx = getattr(self, (f"_{self.dataset}_idx"))
        if data_index is not None:
            x[..., :] = (
                x[..., :] * self._norm_mul[data_index] + self._norm_add[data_index]
            )
        elif x.shape[-1] == len(dataset_idx):
            x[..., :] = (
                x[..., :] * self._norm_mul[dataset_idx] + self._norm_add[dataset_idx]
            )
        else:
            x[..., :] = x[..., :] * self._norm_mul + self._norm_add
        return x

    def forward(self, x: torch.Tensor, in_place: bool = True) -> torch.Tensor:
        return self.transform(x, in_place=in_place)

    def inverse_transform(
        self,
        x: torch.Tensor,
        in_place: bool = True,
        data_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denormalizes an input tensor x of shape [..., nvars | nvars_pred];

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

        # Denormalize dynamic or full tensors
        # input and predicted tensors have different shapes
        # hence, we mask out the forcing indices
        if not in_place:
            x = x.clone()

        dataset_idx = getattr(self, (f"_{self.dataset}_idx"))
        if data_index is not None:
            x[..., :] = (x[..., :] - self._norm_add[data_index]) / self._norm_mul[
                data_index
            ]
        elif x.shape[-1] == len(dataset_idx):
            x[..., :] = (x[..., :] - self._norm_add[dataset_idx]) / self._norm_mul[
                dataset_idx
            ]
        else:
            x[..., :] = (x[..., :] - self._norm_add) / self._norm_mul
        return x
