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
import sys
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Optional

import torch

from anemoi.training.utils.enums import TensorDim

if TYPE_CHECKING:
    from anemoi.models.interface import AnemoiModelInterface
    from anemoi.training.losses.scaler_tensor import TENSOR_SPEC

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum

LOGGER = logging.getLogger(__name__)


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
    def get_scaling_values(self, **kwargs) -> torch.Tensor:
        """Abstract method to get loss scaling."""
        ...

    def normalise(self, values: torch.Tensor) -> torch.Tensor:
        """Normalise the scaler values."""
        if self.norm is None:
            return values

        if self.norm.lower() in ["l1", "unit-sum"]:
            return values / torch.sum(values)

        if self.norm.lower() == "unit-mean":
            return values / torch.mean(values)

        error_msg = f"{self.norm} must be one of: None, unit-sum, l1, unit-mean."
        raise ValueError(error_msg)

    def get_scaling(self) -> TENSOR_SPEC:
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


class AvailableCallbacks(StrEnum):
    INITIAL_SCALING_VALUES = "initial_scaling_values"
    ON_TRAINING_START = "on_training_start"
    ON_TRAIN_EPOCH_START = "on_train_epoch_start"
    ON_TRAIN_EPOCH_END = "on_train_epoch_end"
    ON_TRAIN_BATCH_START = "on_train_batch_start"
    ON_TRAIN_BATCH_END = "on_train_batch_end"

    ON_VALID_BATCH_START = "on_valid_batch_start"
    ON_VALID_BATCH_END = "on_valid_batch_end"


class BaseUpdatingScaler(BaseScaler):
    """Base class for updating scalers.

    The updating scalers have a variety of callback methods associated with them,
    which are called during the training loop. These methods allow the scalers to
    update their values based on the current state of the model and the training data.

    The callback methods are expected to return a np.ndarray of scaling values,
    which will be normalised and returned by the `get_scaling_values` method.

    Override `on_training_start` to provide initial scaling values if needed.
    The default implementation returns an array of ones.
    """

    _cached_scaling_values: Optional[torch.Tensor] = None

    def initial_scaling_values(self) -> Optional[torch.Tensor]:
        """Get initial scaling values.

        Returns
        -------
        torch.Tensor
            Initial scaling values, default is an array of ones.
        """
        return torch.ones(tuple([1] * len(self.scale_dims)))

    def on_training_start(self, model: AnemoiModelInterface) -> Optional[torch.Tensor]:  # noqa: ARG002
        """Callback method called at the start of training."""
        LOGGER.debug("%s.on_training_start called.", self.__class__.__name__)

    def on_train_epoch_start(self, model: AnemoiModelInterface) -> Optional[torch.Tensor]:  # noqa: ARG002
        """Callback method called at the start of each epoch."""
        LOGGER.debug("%s.on_train_epoch_start called.", self.__class__.__name__)

    def on_train_epoch_end(self, model: AnemoiModelInterface) -> Optional[torch.Tensor]:  # noqa: ARG002
        """Callback method called at the end of each epoch."""
        LOGGER.debug("%s.on_train_epoch_end called.", self.__class__.__name__)

    def on_train_batch_start(self, model: AnemoiModelInterface) -> Optional[torch.Tensor]:  # noqa: ARG002
        """Callback method called at the start of each batch."""
        LOGGER.debug("%s.on_train_batch_start called.", self.__class__.__name__)

    def on_train_batch_end(self, model: AnemoiModelInterface) -> Optional[torch.Tensor]:  # noqa: ARG002
        """Callback method called at the end of each batch."""
        LOGGER.debug("%s.on_train_batch_end called.", self.__class__.__name__)

    def on_valid_batch_start(self, model: AnemoiModelInterface) -> Optional[torch.Tensor]:  # noqa: ARG002
        """Callback method called at the start of each validation batch."""
        LOGGER.debug("%s.on_valid_batch_start called.", self.__class__.__name__)

    def on_valid_batch_end(self, model: AnemoiModelInterface) -> Optional[torch.Tensor]:  # noqa: ARG002
        """Callback method called at the end of each validation batch."""
        LOGGER.debug("%s.on_valid_batch_end called.", self.__class__.__name__)

    def get_scaling_values(self) -> torch.Tensor:
        """Get scaling values based on the initial scaling values callback or cache if set.

        Returns
        -------
        torch.Tensor
            Scaling values as a torch tensor.
        """
        if self._cached_scaling_values is not None:
            LOGGER.debug("Using cached scaling values for %s.", self.__class__.__name__)
            return self._cached_scaling_values

        return self.initial_scaling_values()

    def get_scaling(self) -> TENSOR_SPEC:
        """Get scaling values based on the initial scaling values callback."""
        if self._cached_scaling_values is None:
            self._cached_scaling_values = self.initial_scaling_values()

        scalar_values = self._cached_scaling_values

        scalar_values = self.normalise(scalar_values)
        scale_dims = tuple(x.value for x in self.scale_dims)
        return scale_dims, scalar_values

    def get_callback_scaling_values(self, callback: AvailableCallbacks, **kwargs) -> TENSOR_SPEC:
        """Get scaling values based on the callback.

        Will update the cached scaling values if the callback returns a value.
        Any subsequent calls to `get_scaling` will use the cached values.

        Parameters
        ----------
        callback : AvailableCallbacks
            The callback method to use for getting the scaling values.
        **kwargs : dict
            Additional keyword arguments to pass to the callback method.

        Returns
        -------
        TENSOR_SPEC
            A tuple containing the scale dimensions and the scaler values.
        """
        if not hasattr(self, callback):
            error_msg = f"{self.__class__.__name__} does not have a method {callback}."
            raise ValueError(error_msg)

        scalar_values = getattr(self, callback)(**kwargs)
        if scalar_values is not None:
            self._cached_scaling_values = scalar_values

        return self.get_scaling()
