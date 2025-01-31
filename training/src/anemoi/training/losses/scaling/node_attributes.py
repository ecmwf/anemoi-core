# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
import torch

from anemoi.training.losses.scaling import BaseScaler

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


class GraphNodeAttributeScaler(BaseScaler, ABC):
    """Base class for all loss masks that are more than one-dimensional."""

    def __init__(
        self,
        data_indices: IndexCollection,
        graph_data: HeteroData,
        scale_dim: str,
        nodes_name: str,
        nodes_attribute_name: str | None = None,
        apply_output_mask: bool = False,
        norm: str = None,
        **kwargs,
    ) -> None:
        """Initialise Scaler.

        Parameters
        ----------
        data_indices : IndexCollection
            Collection of data indices.
        scale_dim : str
            Dimensions to scale in the format of a string.
        nodes_name : str
            Name of the nodes in the graph.
        nodes_attribute_name : str | None, optional
            Name of the node attribute to use for scaling, by default None
        apply_output_mask : bool, optional
            Whether to apply output mask to the scaling, by default False
        norm : str, optional
            Type of normalization to apply. Options are None, unit-sum, unit-mean and l1.
        **kwargs : dict
            Additional keyword arguments.
        """
        self.apply_output_mask = apply_output_mask
        self.attr_values = graph_data[nodes_name][nodes_attribute_name].squeeze()
        super().__init__(data_indices, scale_dim, norm)
        del kwargs

    def get_scaling(self) -> np.ndarray:
        return self.attr_values