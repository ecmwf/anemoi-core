# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch

from anemoi.models.preprocessing import BasePreprocessor

LOGGER = logging.getLogger(__name__)


class BaseImputer(BasePreprocessor, ABC):
    """Base class for Imputers.

    Stateless preprocessor: detects NaN locations in the input and fills
    them with configured replacement values. The inverse_transform is a
    no-op — loss masking is handled externally from the target dataset.
    """

    def __init__(self, config=None, **kwargs) -> None:
        """Initialize the imputer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        """
        super().__init__(config)

    @abstractmethod
    def get_imputing_replacements(
        self, name_to_index: dict[str, int], statistics: dict[str, torch.Tensor]
    ) -> dict[int, float]: ...

    @abstractmethod
    def impute(
        self, data: torch.Tensor, replacements: dict[int, float], impute_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...

    def transform(
        self,
        x: torch.Tensor,
        statistics: Optional[dict[str, np.ndarray]] = None,
        name_to_index: Optional[dict[str, int]] = None,
        impute_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input data view (with NaNs).
        statistics : dict[str, np.ndarray]
            Statistics dictionary required for normalization.
        name_to_index : dict[str, int]
            Dictionary mapping variable names to their indices, required for normalization.
        impute_mask : torch.Tensor, optional (default = None)
            Optional boolean mask over the latlon dimension. When provided, NaNs are
            only filled where the mask is True. Passing in None imputes everywhere.

        Returns
        -------
        torch.Tensor
            View with NaNs replaced by configured values.
        """
        replacements = self.get_imputing_replacements(name_to_index, statistics)

        x = self.impute(x, replacements=replacements, impute_mask=impute_mask)

        return x

    def inverse_transform(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        """No-op: loss masking is handled externally from the target dataset."""
        return x


class InputImputer(BaseImputer):
    """Imputes missing values using the statistics supplied.

    Expects the config to have keys corresponding to available statistics
    and values as lists of variables to impute::

        default: "none"
        mean:
            - y
        maximum:
            - x
        minimum:
            - q
    """

    def get_imputing_replacements(
        self, name_to_index: dict[str, int], statistics: dict[str, torch.Tensor]
    ) -> dict[int, float]:
        """Get the replacement values for imputation."""
        assert name_to_index is not None, f"{self.__class__.__name__} require name_to_index for imputation."
        assert statistics is not None, f"{self.__class__.__name__} require statistics for imputation."
        replacements: dict[int, float] = {}
        for name, idx in name_to_index.items():
            method = self.methods.get(name, self.default)
            if method == "none":
                continue
            assert method in statistics, f"{method} is not a method in the statistics metadata"
            replacements[idx] = float(statistics[method][idx])
        return replacements

    def impute(
        self, data: torch.Tensor, replacements: dict[int, float], impute_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for idx, value in replacements.items():
            col = data[..., idx]
            fill_here = torch.isnan(col)
            if impute_mask is not None:
                fill_here = fill_here & impute_mask
            data[..., idx] = torch.where(fill_here, value, col)
        return data


class ConstantImputer(BaseImputer):
    """Imputes missing values using a constant value.

    Expects the config to have keys that are numeric values
    and values as lists of variables to impute::

        default: "none"
        1:
            - y
        5.0:
            - x
        3.14:
            - q
    """

    def get_imputing_replacements(self, name_to_index: dict[str, int], **_kwargs) -> dict[int, float]:
        assert name_to_index is not None, f"{self.__class__.__name__} require name_to_index for imputation."
        replacements: dict[int, float] = {}
        for name, idx in name_to_index.items():
            method = self.methods.get(name, self.default)
            if method == "none":
                continue
            replacements[idx] = float(method)
        return replacements

    def impute(
        self, data: torch.Tensor, replacements: dict[int, float], impute_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for idx, value in replacements.items():
            col = data[..., idx]
            fill_here = torch.isnan(col)
            if impute_mask is not None:
                fill_here = fill_here & impute_mask
            data[..., idx] = torch.where(fill_here, value, col)
        return data


class CopyImputer(BaseImputer):
    """Imputes missing values by copying from another variable.

    Config maps source variable names to lists of target variables::

        default: "none"
        variable_to_copy:
            - variable_missing_1
            - variable_missing_2
    """

    def get_imputing_replacements(self, name_to_index: dict[str, int], **_kwargs) -> dict[int, int]:
        """Return index → source_index mapping."""
        assert name_to_index is not None, f"{self.__class__.__name__} require name_to_index for imputation."
        copy_sources: dict[int, int] = {}
        for name, idx in name_to_index.items():
            source_name = self.methods.get(name, self.default)
            if source_name == "none":
                continue
            source_idx = name_to_index.get(source_name)
            if source_idx is not None:
                copy_sources[idx] = source_idx
        return copy_sources

    def impute(
        self, data: torch.Tensor, replacements: dict[int, int], impute_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for idx_dst, idx_src in replacements.items():
            if idx_dst is not None and idx_src is not None:
                nan_mask = torch.isnan(data[..., idx_dst])
                if impute_mask is not None:
                    nan_mask = nan_mask & impute_mask
                source_vals = data[..., idx_src]
                assert not torch.isnan(
                    source_vals[nan_mask]
                ).any(), "NaNs found in source variable at locations where target variable needs imputation."
                data[..., idx_dst] = torch.where(nan_mask, source_vals, data[..., idx_dst])
        return data
