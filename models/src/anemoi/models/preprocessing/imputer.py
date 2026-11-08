# (C) Copyright 2024-2026 Anemoi contributors.
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

    supports_skip_imputation = True

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
        self, name_to_index: dict[str, int], statistics: dict[str, np.ndarray]
    ) -> dict[int, float]: ...

    @abstractmethod
    def impute(self, data: torch.Tensor, replacements: dict[int, float]) -> torch.Tensor: ...

    def transform(
        self,
        x: torch.Tensor,
        statistics: Optional[dict[str, np.ndarray]] = None,
        name_to_index: Optional[dict[str, int]] = None,
        skip_imputation: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Impute missing values in the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (with NaNs), variables on the last axis.
        statistics : dict[str, np.ndarray]
            Statistics dictionary, required by imputers that fill with a statistic.
        name_to_index : dict[str, int]
            Dictionary mapping variable names to their indices, required for imputation.
        skip_imputation : bool, optional
            Return the input unchanged instead of filling missing values.

        Returns
        -------
        torch.Tensor
            Tensor with NaNs replaced by configured values.
        """
        if skip_imputation:
            return x

        replacements = self.get_imputing_replacements(
            name_to_index=name_to_index,
            statistics=statistics,
        )
        x = self.impute(x, replacements=replacements)

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
        self, name_to_index: dict[str, int], statistics: dict[str, np.ndarray]
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

    def impute(self, data: torch.Tensor, replacements: dict[int, float]) -> torch.Tensor:
        for idx, value in replacements.items():
            col = data[..., idx]
            data[..., idx] = torch.where(torch.isnan(col), value, col)
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

    def impute(self, data: torch.Tensor, replacements: dict[int, float]) -> torch.Tensor:
        for idx, value in replacements.items():
            col = data[..., idx]
            data[..., idx] = torch.where(torch.isnan(col), value, col)
        return data


class InputOnlyImputer(BaseImputer):
    """Imputes missing input values with a statistic or a constant, per variable.

    Combines :class:`InputImputer` and :class:`ConstantImputer` in one configuration: a key is
    either a statistic name or a number, mapping to the variables it fills::

        default: "none"
        mean:
            - var1
        0.0:
            - var2

    As every imputer, it only fills model inputs: the loss targets keep their missing values (they
    are pre-processed with ``skip_imputation``). ``multi_step`` (the number of input time steps the
    testeos version imputed, to keep target NaNs in a whole-sample tensor) is accepted for
    compatibility and has no effect: each view the pre-processors see here is either all inputs or
    all targets.
    """

    _STATISTICS = ("mean", "stdev", "minimum", "maximum")

    @classmethod
    def _process_config(cls, config):
        config = dict(config or {})
        if config.pop("multi_step", None) is not None:
            LOGGER.info("%s: multi_step has no effect; inputs are imputed at every time step.", cls.__name__)
        return super()._process_config(config)

    def get_imputing_replacements(
        self, name_to_index: dict[str, int], statistics: Optional[dict[str, torch.Tensor]] = None
    ) -> dict[int, float]:
        """Get the replacement values for imputation: a statistic of the variable or a constant."""
        assert name_to_index is not None, f"{self.__class__.__name__} require name_to_index for imputation."
        replacements: dict[int, float] = {}
        for name, idx in name_to_index.items():
            method = self.methods.get(name, self.default)
            if method in ("none", None):
                continue
            if method in self._STATISTICS:
                assert statistics is not None and method in statistics, f"{method} is not in the statistics"
                replacements[idx] = float(statistics[method][idx])
            else:
                replacements[idx] = float(method)
        return replacements

    def impute(self, data: torch.Tensor, replacements: dict[int, float]) -> torch.Tensor:
        for idx, value in replacements.items():
            col = data[..., idx]
            data[..., idx] = torch.where(torch.isnan(col), value, col)
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

    def impute(self, data: torch.Tensor, replacements: dict[int, int]) -> torch.Tensor:
        for idx_dst, idx_src in replacements.items():
            if idx_dst is not None and idx_src is not None:
                nan_mask = torch.isnan(data[..., idx_dst])
                source_vals = data[..., idx_src]
                assert not torch.isnan(
                    source_vals[nan_mask]
                ).any(), "NaNs found in source variable at locations where target variable needs imputation."
                data[..., idx_dst] = torch.where(nan_mask, source_vals, data[..., idx_dst])
        return data
