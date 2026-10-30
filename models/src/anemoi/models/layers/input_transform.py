# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn


class BaseInputTransform(nn.Module, ABC):
    """Turns the raw input variables of a dataset into the encoder's input features.

    This is the single place where a model decides how variables are represented, so
    different embeddings can be compared without changing the model classes. One
    transform is shared by all datasets, so a variable gets the same representation in
    every domain; the variables of the current dataset are therefore passed on every call.

    Parameters
    ----------
    dataset_metadata : dict, optional
        Per-dataset metadata as stored by anemoi-training (``metadata["dataset"]``), for
        transforms that describe variables by their physical meaning.
    """

    output_dim: Optional[int] = None
    """Feature width per input time step, or ``None`` to keep one feature per variable."""

    def __init__(self, dataset_metadata: Optional[dict] = None) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, node_attributes_data: Tensor, variables: list[str]) -> Tensor:
        """Build the encoder input for one dataset.

        Parameters
        ----------
        x : Tensor
            Raw input, shaped ``(batch, time, ensemble, grid, vars)``.
        node_attributes_data : Tensor
            Node attributes of the dataset's grid, shaped ``(batch * ensemble * grid, n_attributes)``.
        variables : list[str]
            Names of the variables along the last axis of ``x``, in order.

        Returns
        -------
        Tensor
            Encoder input, shaped ``(batch * ensemble * grid, features)``, node attributes included.
        """

    @staticmethod
    def append_node_attributes(x_vars: Tensor, node_attributes_data: Tensor) -> Tensor:
        return torch.cat((x_vars, node_attributes_data), dim=-1)


class FlattenInputTransform(BaseInputTransform):
    """Feeds the raw variable values to the encoder unchanged.

    Has no learnable parameters, so a model without a configured ``input_transform``
    behaves as it did before this extension point existed.
    """

    def forward(self, x: Tensor, node_attributes_data: Tensor, variables: list[str]) -> Tensor:
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
        return self.append_node_attributes(x_vars, node_attributes_data)
