# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from anemoi.models.interface import AnemoiModelInterface

LOGGER = logging.getLogger(__name__)
SCALER_DTYPE = tuple[tuple[int], np.ndarray]


class BaseScaler(ABC):
    """Base class for all loss scalers."""

    scale_dims: tuple[TensorDim] = None

    def __init__(self, norm: str | None = None) -> None:
        """Initialise BaseScaler.

        Parameters
        ----------
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        self.norm = norm
        assert norm in [
            None,
            "unit-sum",
            "l1",
            "unit-mean",
        ], f"{self.__class__.__name__}.norm must be one of: None, unit-sum, l1, unit-mean"
        assert self.scale_dims is not None, f"Class {self.__class__.__name__} must define 'scale_dims'"
        if isinstance(self.scale_dims, TensorDim):
            self.scale_dims = (self.scale_dims,)

    @abstractmethod
    def get_scaling_values(self, **kwargs) -> np.ndarray:
        """Abstract method to get loss scaling."""
        ...

    def normalise(self, values: np.ndarray) -> np.ndarray:
        """Normalise the scaler values."""
        if self.norm is None:
            return values

        if self.norm.lower() in ["l1", "unit-sum"]:
            return values / np.sum(values)

        if self.norm.lower() == "unit-mean":
            return values / np.mean(values)

        error_msg = f"{self.norm} must be one of: None, unit-sum, l1, unit-mean."
        raise ValueError(error_msg)

    def get_scaling(self) -> SCALER_DTYPE:
        """Get scaler.

        Returns
        -------
        scale_dims : tuple[int]
            Dimensions over which the scalers are applied.
        scaler_values : np.ndarray
            Scaler values
        """
        scaler_values = self.get_scaling_values()
        scaler_values = self.normalise(scaler_values)
        scale_dims = tuple(x.value for x in self.scale_dims)
        return scale_dims, scaler_values


class BaseDelayedScaler(BaseScaler):
    """Base class for delayed Scalers.

    The delayed scalers are only initialised when creating all the scalers, but its value is
    computed during the first iteration of the training loop. This delayed scalers are suitable
    for scalers requiring information from the `model.pre_processors`.
    """

    # SL: why does this go via numpy??? Here we should pass the processors directly instead of the model
    @abstractmethod
    def get_delayed_scaling_values(self, **kwargs) -> np.ndarray: ...

    def get_delayed_scaling(self, model: AnemoiModelInterface) -> SCALER_DTYPE:
        scaler_values = self.get_delayed_scaling_values(model)
        scaler_values = self.normalise(scaler_values)
        scale_dims = tuple(x.value for x in self.scale_dims)
        return scale_dims, scaler_values
