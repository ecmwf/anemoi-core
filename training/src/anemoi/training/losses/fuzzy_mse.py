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

import torch
from torch_geometric.data import HeteroData
import xarray as xr 
import numpy as np 

from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.graphs.nodes.builders.from_vectors import LatLonNodes
from anemoi.graphs.edges.builders.knn import KNNEdges


LOGGER = logging.getLogger(__name__)


class WeightedAreaRelatetSortedIntensityLoss(BaseWeightedLoss):
    """Node-weighted fuzzy MSE loss."""

    name = "arsil"

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted MSE Loss.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False

        """
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
        )
        # register some kwargs as properties
        for key in ["num_neighbors", "depth"]:
            if key in kwargs: setattr(self, key, kwargs[key]) 

        #INFO:mask currently hard coded 
        self.register_buffer("neighbor_index", self._generate_index_tensor(7), persistent=True)

        self.register_buffer("original_node_weights", node_weights, persistent=True)

    def _generate_index_tensor(self, refinmement_level: int):
        """
        generates index tensor with neighborhood indices 

        return
        ------ 
        z (Nnodes, Nneigh): torch.tensor (int), holds in a given line the neighbor indices of the corresponding node
        Nnodes: int, number of nodes in data domain
        Nneigh: int, number of neighbors in area around each node 
        generates index tensor that defines the neighborhood of each point 
        """
        # generate ValueError
        grid_filename = "/shared/data/fe1ai/ml-datasets/structure/01_aicon-graph-files/icon_grid_0026_R03B07_G.nc"
        grid_ds = xr.open_dataset(grid_filename) 
        mask = grid_ds["refinement_level_c"] <= refinmement_level
        lat, lon = np.rad2deg(grid_ds["clat"].values), np.rad2deg(grid_ds["clon"].values)
        lat, lon = lat[mask], lon[mask]
        mygraph = HeteroData()
        source_nodes = LatLonNodes(longitudes=lon, latitudes=lat, name = "source")
        mygraph = source_nodes.update_graph(mygraph)
        target_nodes = LatLonNodes(longitudes=lon, latitudes=lat, name = "target")
        mygraph = target_nodes.update_graph(mygraph)
        edges = KNNEdges(
            source_name = "source",
            target_name = "target",
            num_nearest_neighbours=self.num_neighbors,
            )
        mygraph = edges.update_graph(mygraph)
        edge_index = mygraph["source","to","target"].edge_index
        Nnodes = mygraph["source"].x.shape[0]
        # Sort edges by target node
        sorted_targets, perm = torch.sort(edge_index[1])
        sorted_sources = edge_index[0][perm]

        return sorted_sources.view(Nnodes, self.num_neighbors).int()

    def _reindex(self, x: torch.Tensor):
        """
        generates tensor that holds neighborhood info for each node 

        Parameters
        ----------
        x : torch.Tensor    
            Either pred or target shape (bs, ensemble, lat*lon, n_outputs)
            or (ensemble, lat*lon, n_outputs) or (lat*lon)
        Returns
        -------
        torch.Tensor with shape (bs, ensemble, lat*lon, number num_neighbors, n_outputs)
            or (ensemble, lat*lon, number num_neighbors, n_outputs)


        """
        # determine if batch dimension present
        # full_rank = True if len(x.shape)==4 else False
        # if no batch dimension create trivial 
        # if full_rank:
        x = x.unsqueeze(-2).expand(-1, -1, -1, self.num_neighbors, -1).clone()
        # else: 
        #     print(f"{x.shape=}")
        #     x = x.unsqueeze(1).expand(-1, self.num_neighbors)
        #     print(f"{x.shape=}")

  
        # expand node feature tensor to (bs, ens, lat*lon, num_neighbors ,n_outputs)    

        # expand index tensor to shape of 
        idx = self.neighbor_index.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        idx = idx.expand(x.shape[0], x.shape[1], -1, -1, x.shape[-1]).clone()
        result = torch.gather(x, 2, idx.to(torch.int64))
        result, _ = torch.sort(result, dim=-2)
        # if not full_rank: result = result.squeeze(0)
        # if no_batch: result = result.squeeze(0) 
        return result 

    def _reindex_weights(self, x: torch.Tensor):
        """
        generates tensor that holds neighborhood info for each node 

        Parameters
        ----------
        x : torch.Tensor    
            weight vector shape (lat*lon)
        Returns
        -------
        torch.Tensor wit shape (number of nodes, number num_neighbors)

        """
  
        # expand node feature tensor to (bs, ens, lat*lon, num_neighbors ,n_outputs)    
        x = x.unsqueeze(-1).expand(-1, self.num_neighbors).clone()

        # expand index tensor to shape of 
        idx = self.neighbor_index
        # idx = self.neighbor_index.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # idx = idx.expand(x.shape[0], x.shape[1], -1, -1, x.shape[-1])
        result = torch.gather(x, 0, idx.to(torch.int64))
        return result 

    def _aggregate(self,
                  x: torch.Tensor, 
                  num_neighbors:int = 4,
                  depth: int = 1,
                  sort: bool = False,
                  top_level: bool = True,
    ) -> torch.Tensor:
        """aggregates and sorts features in a local area 

        Parameters
        ----------
        x : torch.Tensor    
            Either pred or target shape (bs, ensemble, lat*lon, n_outputs)
            or (ensemble, lat*lon, n_outputs)
        num_neighbors : int 
            number of neighbors to be aggregated in one super-location
        depth : int 
            number of aggreatation steps -> depth of recursion
        sort : bool 
            debugging setting -> sort=False -> aggreatation has no effect
        top_level : bool
            Determines whether recursive routine is on outer most level

        Returns
        -------
        x: torch.Tensor 
            aggregated/reshaped pred or target
            shape : (bs, ensemble, lat*lon/num_neighbors, num_neighbors, n_outputs)
            or    : (ensemble, lat*lon/num_neighbors, num_neighbors, n_outputs)
        """
        # determine if batch dimension present
        no_batch = True if top_level and len(x.shape)==3 else False
        # if no batch dimension create trivial 
        if no_batch: x = x.unsqueeze(0)

        # go deeper in recursion?
        if depth > 1:
            x = self._aggregate(x, num_neighbors, depth=depth - 1, top_level=False)
            depth -= 1

        # aggretate via reshaping (4 rows always grouped spatially by construction)
        N = x.shape[2]
        latent = x.shape[3] if len(x.shape) > 4 else 1
        N_target = N // num_neighbors
        x = x.reshape(x.shape[0], x.shape[1], N_target, num_neighbors * latent, x.shape[-1])
        if sort: x, _ = torch.sort(x, dim=-2)

        # remove batch dimension if trivial
        if no_batch: x = x.squeeze(0)

        return x 

    def _aggregate_node_weights(self,
                                x: torch.Tensor,
                                num_neighbors: int = 4,
                                depth: int = 1,
    ) -> torch.Tensor:
        """applies spatial aggregation to node_weights

        Parameters
        ----------
        x : torch.Tensor    
            node_weights shape (lat*lon) or (lat*lon/N, N)
        num_neighbors : int 
            number of neighbors to be aggregated in one super-location
        depth : int 
            number of aggreatation steps -> depth of recursion
        sort : bool 
            debugging setting -> sort=False -> aggreatation has no effect

        Returns
        -------
        x: torch.Tensor 
            aggregated/reshaped pred or target
            shape : (lat*lon/Nnneigh, num_neighbors)

        """

        if depth > 1:
            x = self._aggregate_node_weights(x, num_neighbors, depth=depth - 1)
            depth -= 1
        N = x.shape[0]
        latent = x.shape[1] if len(x.shape) > 1 else 1
        N_target = N // num_neighbors
        x = x.reshape(N_target, num_neighbors * latent)
        return x

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
        depth: int = 1,
    ) -> torch.Tensor:
        """Calculates the lat-weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scalar_indices: tuple[int,...], optional
            Indices to subset the calculated scalar with, by default None
        without_scalars: list[str] | list[int] | None, optional
            list of scalars to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        depth: int, optional
            number of recursive coarsening steps in area related sorting
        Returns
        -------
        torch.Tensor
            Weighted ARSIL loss
        """

        pred = self._reindex(pred)
        target = self._reindex(target)
        # self.node_weights = self._reindex_weights(self.original_node_weights)
        self.node_weights = self.original_node_weights.unsqueeze(-1).expand(-1, self.num_neighbors).clone()

        out = torch.square(pred - target)
        out = self.scale(out, scalar_indices, without_scalars=without_scalars)
        return self.scale_by_node_weights(out, squash)
