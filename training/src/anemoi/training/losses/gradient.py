# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
import torch
from torch.distributed.distributed_c10d import ProcessGroup

#from anemoi.training.losses.base import FunctionalLoss
from anemoi.training.losses.base import BaseLoss
from torch_geometric.data import HeteroData
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

LOGGER = logging.getLogger(__name__)

class BaseGradientMetric(MessagePassing):

    def forward(
        self, x: torch.Tensor, edge_index: Adj, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """Computation of gradient metric at nodes

        Args:
            x (Tensor): has shape (..., N, c)
            edge_index (Tensor) : (2, E)
            edge_weight (Tensor) : (E, 1)

        Returns:
            Node gradient metric (Tensor): shape (..., N, c)
        """
        gradient = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        return gradient
    
    def metric(self, flux: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """Computation of the direction derivatives

        Args:
            x_j (Tensor): has shape (..., E, c)
            x_j (Tensor): has shape (..., E, c)
            edge_weight (Tensor) : (E, 1)

        Returns:
            edge_gradient_flux (Tensor): shape (..., E, c)
        """
        # Compute the difference
        difference_of_values = x_j - x_i

        # Compute the gradient normalized by the distance
        edge_gradient_flux = difference_of_values * edge_weight

        return self.metric(edge_gradient_flux)


class GradientMetricMaxAbs(BaseGradientMetric):
    def __init__(self):
        super().__init__(aggr="max")
        self.metric = torch.abs


class GradientMetricMeanAbs(BaseGradientMetric):
    def __init__(self):
        super().__init__(aggr="mean")
        self.metric = torch.abs


class GradientMetricMeanSquare(BaseGradientMetric):
    def __init__(self):
        super().__init__(aggr="mean")
        self.metric = torch.square


class BaseGradientLoss(BaseLoss):

    gradient_metric_layer:BaseGradientMetric

    def __init__(self, graph: HeteroData | str, graph_name: str, **kwargs):
        """_summary_

        Args:
            graph (HeteroData | str): _description_
            graph_name (str): _description_
        """
        super().__init__(**kwargs)

        LOGGER.info(f'Create instance of BaseGradientLoss from graph {graph} and \
                    graph_name {graph_name}')

        if isinstance(graph, str):
            graph = Path(graph)
            graph = torch.load(graph, weights_only=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph = graph.to(device)

        self.num_nodes = graph[graph_name]['x'].shape[0]       

        # graph data->data
        # ----------------
        graph = graph[graph_name, "to", graph_name]

        edge_index = graph["edge_index"]
        edge_weight = 1.0 / graph["edge_length"]

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        graph["edge_index"] = edge_index
        graph["edge_weight"] = edge_weight

        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: str = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group, by default None
        squash_mode : str, optional
            Reduction mode for the variable dimension, by default ``"avg"``
        **kwargs
            Additional keyword arguments

        Returns
        -------
        torch.Tensor
            Weighted loss
        """

        is_sharded = grid_shard_slice is not None

        difference = pred - target

        gradient_difference_metric = self.gradient_metric_layer(
            difference,
            edge_index = self.edge_index,
            edge_weight = self.edge_weight,
        )
        
        out = gradient_difference_metric
       
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(out, squash, group=group if is_sharded else None, squash_mode=squash_mode)    


class GradientMeanSquareLoss(BaseGradientLoss):
    """Gradient Absolute value Max loss."""

    name: str = "gradient-mean-square"
    gradient_metric_layer = GradientMetricMeanSquare()


class GradientMeanAbsLoss(BaseGradientLoss):
    """Gradient Mean absolute loss."""

    name: str = "gradient-mean-abs"
    gradient_metric_layer = GradientMetricMeanAbs()


class GradientMaxAbsLoss(BaseGradientLoss):
    """Gradient Absolute value Max loss."""

    name: str = "gradient-abs-max"
    gradient_metric_layer = GradientMetricMaxAbs()


class DiscreteSobolevH1Loss(BaseGradientLoss):
    """H1 sobolev norm from discrete differential operator (not physical gradient)

    compute square of H1 norm, defined as

    ||f||_H1^2 = ||f||_2^2 + metric( discrete_gradient(f) )

    """

    name: str = "discrete-sobolev-H1"
    gradient_metric_layer = GradientMetricMeanSquare()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: str = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Calculates the area-weighted scaled loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group, by default None
        squash_mode : str, optional
            Reduction mode for the variable dimension, by default ``"avg"``
        **kwargs
            Additional keyword arguments

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        is_sharded = grid_shard_slice is not None        

        # Computation of the gradient part        
        gradient_difference_metric = super().forward(pred=pred, 
                                 target=target, 
                                 squash=squash, 
                                 scaler_indices=scaler_indices,
                                 without_scalers=without_scalers,
                                 grid_shard_slice=grid_shard_slice,
                                 group=group,
                                 squash_mode=squash_mode,
                                 )
        

        # Computation of the mse part        
        mse = torch.square(pred - target)
        mse = self.scale(mse, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        mse = self.reduce(mse, squash, group=group if is_sharded else None, squash_mode=squash_mode)     

        return mse + gradient_difference_metric

