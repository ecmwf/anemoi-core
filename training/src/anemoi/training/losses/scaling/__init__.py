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
    import numpy as np

    from anemoi.models.data_indices.collection import IndexCollection

LOGGER = logging.getLogger(__name__)


class BaseScaler(ABC):
    """Base class for all loss scalers."""

    def __init__(self, data_indices: IndexCollection, scale_dim: int | tuple[int, int]) -> None:
        """Initialise BaseScaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        """
        self.data_indices = data_indices
        self.scale_dim = scale_dim

    @abstractmethod
    def get_scaling(self) -> np.ndarray:
        """Abstract method to get loss scaling."""
        ...

    def get_values(self) -> tuple[int | tuple[int, int], np.ndarray]:
        return self.scale_dim, self.get_scaling()
