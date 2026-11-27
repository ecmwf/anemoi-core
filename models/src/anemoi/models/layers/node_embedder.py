# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Node embedders: per-dataset input embeddings applied before an encoder's mapper.

Every encoder has a node embedder, configured under ``model.encoders.<name>.node_embedder``. It hands the
encoder's mapper the input of each source dataset with ``num_channels`` input channels:

- :class:`NoOpNodeEmbedder` passes the assembled input through unchanged. It belongs to the ``"none"``
  dataset fusing strategy, whose source datasets must already share one number of input channels.
- :class:`LinearNodeEmbedder` projects every source dataset onto a common number of input channels,
  ``num_channels``. The ``"sequential"`` strategy needs such a projecting node embedder; it lets source
  datasets with different numbers of variables share one set of mapper weights.

A node embedder embeds node features only. The encoder edges (``<dataset> -> hidden``) of all source
datasets must still carry the same number of edge features, since the mapper's edge embedding
is shared too.

The model keeps the node embedders in ``encoder_src_projection``, keyed by encoder name.
:class:`LinearNodeEmbedder` is an ``nn.ModuleDict`` keyed by dataset name, so its parameters appear in the
state dict as ``encoder_src_projection.<encoder>.<dataset>.*``; :class:`NoOpNodeEmbedder` has no parameters
and adds no state-dict keys.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor
from torch import nn

from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict

NOOP_NODE_EMBEDDER_TARGET = "anemoi.models.layers.node_embedder.NoOpNodeEmbedder"


def is_noop_node_embedder_config(node_embedder_config: Mapping | None) -> bool:
    """Whether a ``node_embedder`` config passes the input through unchanged.

    ``null`` counts as :class:`NoOpNodeEmbedder`, so configs written before every encoder had a node
    embedder keep working.
    """
    return node_embedder_config is None or node_embedder_config.get("_target_") == NOOP_NODE_EMBEDDER_TARGET


class NoOpNodeEmbedder(nn.Module):
    """Pass-through node embedder: the mapper takes each source dataset's assembled input as is.

    Used by the ``"none"`` dataset fusing strategy, much like ``NoOpProcessor`` stands in for a
    processor. All source datasets must have the same number of input channels, which becomes
    ``num_channels``.

    Parameters
    ----------
    in_channels : Mapping[str, int]
        Assembled encoder input channels per source dataset (variables, coordinates and node attributes).
    layer_kernels : DotDict, optional
        Accepted for a common node embedder interface and not used.
    """

    def __init__(self, in_channels: Mapping[str, int], layer_kernels: DotDict | None = None) -> None:
        del layer_kernels  # nothing to build
        if not in_channels:
            msg = "NoOpNodeEmbedder needs at least one source dataset in in_channels."
            raise ValueError(msg)
        channels = set(in_channels.values())
        if len(channels) > 1:
            msg = (
                "NoOpNodeEmbedder passes the input through unchanged, so all source datasets must have the same "
                f"number of input channels; got {dict(in_channels)}. Use a projecting node embedder such as "
                "LinearNodeEmbedder (with dataset_fusing_strategy 'sequential') for differing input channels."
            )
            raise ValueError(msg)
        super().__init__()
        self.num_channels = channels.pop()

    def forward(self, x: Tensor, dataset_name: str) -> Tensor:
        """Return the input of ``dataset_name`` unchanged.

        Parameters
        ----------
        x : Tensor
            Assembled encoder input of ``dataset_name``, shape ``(..., num_channels)``.
        dataset_name : str
            Source dataset the input belongs to.

        Returns
        -------
        Tensor
            ``x`` itself.
        """
        del dataset_name
        return x


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
