# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from torch import Tensor
from torch import nn

LOGGER = logging.getLogger(__name__)


class SpatialPreprocessor(nn.Module):
    """Base class for preprocessors that operate across the spatial (grid) dimension.

    Unlike ``BasePreprocessor`` which applies variable-wise arithmetic on a fixed grid,
    ``SpatialPreprocessor`` subclasses may change the grid dimension — for example
    projecting data from a low-resolution grid onto a high-resolution grid.

    Subclasses must implement ``forward``. The ``inverse`` method raises
    ``NotImplementedError`` by default because spatial projections are generally
    not invertible.

    Spatial preprocessors are registered on ``AnemoiModelInterface`` as
    ``self.spatial_pre_processors`` (a ``nn.ModuleDict`` keyed by dataset name)
    and are saved in the model checkpoint.  They are intentionally graph-topology-
    aware and therefore live in ``anemoi-models``, keeping ``anemoi-training`` free
    of any ``anemoi-graphs`` dependency.
    """

    def forward(self, x: Tensor, model_comm_group=None, grid_shard_sizes=None) -> Tensor:
        """Project input to a (potentially different) grid.

        Parameters
        ----------
        x : Tensor
            Input tensor, shape ``(batch, time, ensemble, grid_src, vars)``.

        Returns
        -------
        Tensor
            Output tensor, shape ``(batch, time, ensemble, grid_dst, vars)``.
        """
        raise NotImplementedError

    def inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not support inverse projection.")
