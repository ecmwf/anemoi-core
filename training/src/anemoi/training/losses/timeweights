# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch

LOGGER = logging.getLogger(__name__)


class LeadTimeDecayWeight:
    """Base class to load and optionally change the weight attribute of each specific predicted lead time in the batch.

    Attributes
    ----------
    base_loss_weight: DictConfig or list of float, optional
        if lead time decay is applied to combined loss, the base loss weight is multiplied by the decay factor

    Methods
    -------
    weights(self, graph_data)
        Load time weight attribute.
    """

    def __init__(self, decay_factor: float = 0.15, method: str = "linear", inverse: bool = False):
        """Initialize graph node attribute with target nodes and node attribute.

        Parameters
        ----------
        target_nodes: str
            name of nodes, key in HeteroData graph object
        node_attribute: str
            name of node weight attribute, key in HeteroData graph object
        """
        self.method = method
        self.decay_factor = decay_factor
        self.inverse = inverse

    def forward_weights(self, relative_date_indices: list[int]) -> torch.Tensor:
        """Returns weight of type self.node_attribute for nodes self.target.

        Attempts to load from graph_data and calculates area weights for the target
        nodes if they do not exist.

        Parameters
        ----------
        decay_factor: float
           decay factor for the lead time weights computation

        Returns
        -------
        torch.Tensor
            weight of target nodes
        """
        if self.method == "exponential":
            return torch.exp(-self.decay_factor * torch.tensor(relative_date_indices))
        if self.method == "linear":
            return (
                1 - self.decay_factor * torch.tensor(relative_date_indices) / torch.tensor(relative_date_indices).max()
            )
        msg = f"Method {self.method} not supported"
        raise NotImplementedError(msg)

    def backward_weights(self, relative_date_indices: list[int]) -> torch.Tensor:
        """Returns weight of type self.node_attribute for nodes self.target.

        Attempts to load from graph_data and calculates area weights for the target
        nodes if they do not exist.

        Parameters
        ----------
        decay_factor: float
           decay factor for the lead time weights computation

        Returns
        -------
        torch.Tensor
            weight of target nodes
        """
        if self.method == "exponential":
            return 1 - torch.exp(-self.decay_factor * torch.tensor(relative_date_indices))
        if self.method == "linear":
            return self.decay_factor * torch.tensor(relative_date_indices) / torch.tensor(relative_date_indices).max()
        msg = f"Method {self.method} not supported"
        raise NotImplementedError(msg)

    def weights(self, relative_date_indices: list[int]) -> torch.Tensor:
        """Returns weight of type self.node_attribute for nodes self.target.

        Attempts to load from graph_data and calculates area weights for the target
        nodes if they do not exist.

        Parameters
        ----------
        graph_data: HeteroData
            graph object

        Returns
        -------
        torch.Tensor
            weight of target nodes
        """
        if self.inverse:
            return self.backward_weights(relative_date_indices)
        return self.forward_weights(relative_date_indices)
