# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import sys
from abc import ABC
from abc import abstractmethod

import torch

from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses.scaler_tensor import TENSOR_SPEC
from anemoi.training.utils.enums import TensorDim

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum

LOGGER = logging.getLogger(__name__)


class BaseScaler(ABC):
    """Base class for all loss scalers."""

    scale_dims: tuple[TensorDim, ...]

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
        scale_dims : tuple[int, ...]
            Dimensions over which the scalers are applied.
        scaler_values : np.ndarray
            Scaler values
        """
        scaler_values = self.get_scaling_values()
        scaler_values = self.normalise(scaler_values)
        scale_dims = tuple(x.value for x in self.scale_dims)
        return scale_dims, scaler_values


class AvailableCallbacks(StrEnum):
    ON_TRAINING_START = "on_training_start"
    ON_BATCH_START = "on_batch_start"


class BaseUpdatingScaler(BaseScaler):
    """Base class for updating scalers.

    The updating scalers have a variety of callback methods associated with them,
    which are called during the training loop. These methods allow the scalers to
    update their values based on the current state of the model and the training data.

    The callback methods are expected to return a np.ndarray of scaling values,
    which will be normalised. If they return None, the scaler will not update its values.

    Override `get_scaling_values` to provide initial scaling values if needed.
    The default implementation returns an array of ones.
    """

    def on_training_start(self, model: AnemoiModelInterface) -> torch.Tensor | None:  # noqa: ARG002
        """Callback method called at the start of training."""
        LOGGER.debug("%s.on_training_start called.", self.__class__.__name__)

    def on_batch_start(self, model: AnemoiModelInterface) -> torch.Tensor | None:  # noqa: ARG002
        """Callback method called at the start of each batch."""
        LOGGER.debug("%s.on_train_batch_start called.", self.__class__.__name__)

    def get_scaling_values(self, **_kwargs) -> torch.Tensor:
        """Get scaling values based on the initial scaling values callback.

        Returns
        -------
        torch.Tensor
            Scaling values as a torch tensor.
        """
        return torch.ones(tuple([1] * len(self.scale_dims)))

    def get_scaling(self) -> TENSOR_SPEC:
        """Get scaling values based on the initial scaling values callback."""
        scalar_values = self.get_scaling_values()

        scale_dims = tuple(x.value for x in self.scale_dims)
        return scale_dims, scalar_values

    def update_scaling_values(self, callback: AvailableCallbacks, **kwargs) -> TENSOR_SPEC | None:
        """Get scaling values based on the callback.

        Can return None if no scaling updates are available.
        In that case, the scaler should not update its values.

        Parameters
        ----------
        callback : AvailableCallbacks
            The callback method to use for getting the scaling values.
        **kwargs : dict
            Additional keyword arguments to pass to the callback method.

        Returns
        -------
        TENSOR_SPEC | None
            A tuple containing the scale dimensions and the scaler values.
        """
        if not hasattr(self, callback):
            error_msg = f"{self.__class__.__name__} does not have a method {callback}."
            raise ValueError(error_msg)

        scalar_values = getattr(self, callback)(**kwargs)
        if scalar_values is None:
            return None

        scalar_values = self.normalise(scalar_values)
        scale_dims = tuple(x.value for x in self.scale_dims)
        return scale_dims, scalar_values
