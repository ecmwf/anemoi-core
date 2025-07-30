# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
# from torch import nn
import einops
import torch
from hydra.utils import instantiate
from torch import nn
from torch_geometric.data import HeteroData

from anemoi.models.layers.utils import load_layer_kernels


class GraphNodeEmbedder(nn.Module):
    def __init__(self, num_input_channels: dict[str, int], out_channels: int, **kwargs):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.out_channels = out_channels

    def forward(self, graph: HeteroData, **kwargs) -> HeteroData:
        return graph


class NodeEmbedder(nn.Module):
    """Class to embed the node representations."""

    def __init__(
        self,
        config,
        node_dim: int,
        out_channels: int,
        num_input_channels: dict[str, int],
    ) -> None:
        super().__init__()
        self.embedders = nn.ModuleDict(
            {
                source: instantiate(config, in_features=in_channels + node_dim, out_features=out_channels)
                for source, in_channels in num_input_channels.items()
            }
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        return_indices: bool = False,
    ) -> torch.Tensor:
        data_embs, idx, i = [], {}, 0
        for data_name, t in x.items():
            data_embs.append(
                self.embedders[data_name](
                    einops.rearrange(t.unsqueeze(0).unsqueeze(0), "bs ens t grid vars -> (bs ens grid) (t vars)")
                )
            )

            if return_indices:
                num_nodes = t.shape[1]
                idx[data_name] = slice(i, i + num_nodes)
                i += num_nodes

        data_embs = torch.concat(data_embs, dim=0)
        return (data_embs, idx) if return_indices else data_embs


class NodeProjector(nn.Module):
    """Class to project the node representations."""

    def __init__(
        self,
        config: dict,
        in_features: int,
        num_output_channels: dict[str, int],
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {
                source: instantiate(
                    config,
                    _recursive_=False,
                    in_features=in_features,
                    out_features=out_channels,
                    layer_kernels=load_layer_kernels(config["layer_kernels"]),
                )
                for source, out_channels in num_output_channels.items()
            }
        )

    def forward(self, x: torch.Tensor, slices: dict[str, slice], dim: int = 0) -> dict[str, torch.Tensor]:
        """Projects the tensor into the different datasets/report types.

        Arguments
        ---------
        x : torch.Tensor
            Node embeddings of the shape (num_nodes, num_channels)
        slice : dict[str, slice]
            A mapping of the slices corresponding to each dataset/report type.
            For example,
                {"era": slice(0, 100), "synop": slice(100, num_nodes)}
            will maps the first 100 values to era values and the rest to synop observations.

        Returns
        -------
        dict[str, torch.Tensor]
            It returns a dict of each dataset/report type with tensors of shape (1, num_source_nodes, dim_source_nodes)
        """
        return {name: self.projectors[name](x[indices]) for name, indices in slices.items()}
