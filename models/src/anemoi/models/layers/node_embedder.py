# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Node embedders: per-dataset input projections for encoders that fuse several source datasets.

A node embedder is configured under ``model.encoders.<name>.node_embedder`` and used by the ``"sequential"``
dataset fusing strategy. It embeds every source dataset of an encoder into a common number of input channels,
``num_channels``, the number of input channels of the encoder's mapper. That lets source
datasets with different numbers of variables share one set of mapper weights.

A node embedder embeds node features only. The encoder edges (``<dataset> -> hidden``) of all source
datasets must still carry the same number of edge features, since the mapper's edge embedding
is shared too.

A node embedder is an ``nn.ModuleDict`` keyed by dataset name, so its parameters appear in the state
dict as ``encoder_src_projection.<encoder>.<dataset>.*``.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor
from torch import nn

from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


class LinearNodeEmbedder(nn.ModuleDict):
    """One linear projection per source dataset, onto a common number of input channels.

    Parameters
    ----------
    num_channels : int
        Common number of output channels; the input channels of the encoder mapper.
    in_channels : Mapping[str, int]
        Assembled encoder input channels per source dataset (variables, coordinates and node attributes).
    layer_kernels : DotDict, optional
        Layer implementations, e.g. ``layer_kernels.Linear``; defaults to ``torch.nn`` layers.
    """

    def __init__(
        self,
        num_channels: int,
        in_channels: Mapping[str, int],
        layer_kernels: DotDict | None = None,
    ) -> None:
        if isinstance(num_channels, bool) or not isinstance(num_channels, int) or num_channels <= 0:
            msg = f"LinearNodeEmbedder num_channels must be a positive integer, got {num_channels!r}."
            raise ValueError(msg)
        if not in_channels:
            msg = "LinearNodeEmbedder needs at least one source dataset in in_channels."
            raise ValueError(msg)

        linear = load_layer_kernels(layer_kernels).Linear
        super().__init__({name: linear(dim, num_channels) for name, dim in in_channels.items()})
        self.num_channels = num_channels

    def forward(self, x: Tensor, dataset_name: str) -> Tensor:
        """Project one source dataset onto the common input channels.

        Parameters
        ----------
        x : Tensor
            Assembled encoder input of ``dataset_name``, shape ``(..., in_channels[dataset_name])``.
        dataset_name : str
            Source dataset the input belongs to.

        Returns
        -------
        Tensor
            Projected input, shape ``(..., num_channels)``.
        """
        return self[dataset_name](x)
