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

if TYPE_CHECKING:
    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BaseScaler(ABC):
    """Base class for all loss scalers."""

    def __init__(self, data_indices: IndexCollection, scale_dims: int | tuple[int], norm: str = None) -> None:
        """Initialise BaseScaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        scale_dims : int | tuple[int]
            Dimensions at which to scale. Must be between -4 and 3 inclusive.
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        """
        self.data_indices = data_indices
        self.scale_dims = scale_dims if isinstance(scale_dims, tuple) else (scale_dims,)
        self.norm = norm
        assert norm in [
            None,
            "unit-sum",
            "l1",
            "unit-mean",
        ], f"{self.__class__.__name__}.norm must be one of: None, unit-sum, l1, unit-mean"
        err_mesg = (
            "Invalid dimension for scaling. Expected dimensions are:"
            "\n  0 (or -4): batch dimension"
            "\n  1 (or -3): ensemble dimension"
            "\n  2 (or -2): spatial dimension"
            "\n  3 (or -1): variable dimension"
            "\nInput tensor shape: (batch_size, n_ensemble, n_grid_points, n_variables)"
        )
        assert all(-4 <= d <= 3 for d in self.scale_dims), err_mesg

    @property
    def is_variable_dim_scaled(self) -> bool:
        return -1 in self.scale_dims or 3 in self.scale_dims

    @property
    def is_spatial_dim_scaled(self) -> bool:
        return -2 in self.scale_dims or 2 in self.scale_dims

    @abstractmethod
    def get_scaling(self, **kwargs) -> np.ndarray:
        """Abstract method to get loss scaling."""
        ...

    def normalise(self, values: np.ndarray) -> np.ndarray:
        if self.norm is None:
            return values

        if self.norm.lower() in ["l1", "unit-sum"]:
            return values / np.sum(values)

        if self.norm.lower() == "unit-mean":
            return values / np.mean(values)

        raise ValueError(f"{self.norm} must be one of: None, unit-sum, l1, unit-mean.")

class BaseDelayedScaler(BaseScaler, ABC):
    pass
