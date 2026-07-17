# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import Optional

import einops
from torch import Tensor
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import ProjectionGraphProvider
from anemoi.models.layers.sparse_projector import SparseProjector
from anemoi.models.preprocessing.spatial import SpatialPreprocessor

LOGGER = logging.getLogger(__name__)


class CrossGridProjector(SpatialPreprocessor):
    """Projects data from one grid onto another via a frozen sparse interpolation matrix.

    Intended for use in downscaling: reprojects low-resolution input onto the
    high-resolution grid before concatenating with native high-resolution inputs
    and passing to the encoder.

    The projection matrix is loaded from the graph at model construction time and
    stored as a non-trainable buffer so it is saved in the model checkpoint. This
    keeps ``anemoi-training`` free of any ``anemoi-graphs`` dependency.

    Handles the full 5-D batch tensor ``(batch, time, ensemble, grid_src, vars)``
    and preserves all time and ensemble slices — unlike ``InterpolationConnection``
    which selects a single timestep.

    Parameters
    ----------
    graph : HeteroData, optional
        Graph containing the source→target projection edges.
    edges_name : tuple[str, str, str], optional
        Edge type key ``(src_node_type, relation, dst_node_type)`` in *graph*.
    edge_weight_attribute : str, optional
        Edge attribute to use as interpolation weights.
    src_node_weight_attribute : str, optional
        Source-node attribute to multiply into edge weights.
    file_path : str | Path, optional
        Path to a pre-computed ``.npz`` sparse matrix (alternative to *graph*).
    row_normalize : bool
        If ``True`` each row of the projection matrix sums to 1 (recommended for
        conservative interpolation).
    autocast : bool
        Whether to run the sparse matmul under automatic mixed precision.
    """

    def __init__(
        self,
        graph: Optional[HeteroData] = None,
        edges_name: Optional[tuple[str, str, str]] = None,
        edge_weight_attribute: Optional[str] = None,
        src_node_weight_attribute: Optional[str] = None,
        file_path: Optional[str | Path] = None,
        row_normalize: bool = True,
        autocast: bool = False,
    ) -> None:
        super().__init__()

        self.provider = ProjectionGraphProvider(
            graph=graph,
            edges_name=edges_name,
            edge_weight_attribute=edge_weight_attribute,
            src_node_weight_attribute=src_node_weight_attribute,
            file_path=file_path,
            row_normalize=row_normalize,
        )
        self.projector = SparseProjector(autocast=autocast)

    def forward(self, x: Tensor, model_comm_group=None, grid_shard_sizes=None) -> Tensor:
        """Project all time and ensemble slices from source grid to target grid.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, time, ensemble, grid_src, vars)``.

        Returns
        -------
        Tensor
            Shape ``(batch, time, ensemble, grid_dst, vars)``.
        """
        batch, time, ensemble, _grid_src, _vars = x.shape

        # Flatten batch/time/ensemble into a single leading dim for the projector.
        x_flat = einops.rearrange(x, "b t e g v -> (b t e) g v")

        channel_shard_sizes = get_shard_sizes(x_flat, -1, model_comm_group)
        if grid_shard_sizes is not None:
            x_flat = all_to_all_transpose(x_flat, -1, channel_shard_sizes, -2, grid_shard_sizes, model_comm_group)

        x_proj = self.projector(x_flat, self.provider.get_edges(device=x.device))

        output_grid_shard_sizes = get_shard_sizes(x_proj, -2, model_comm_group)
        if grid_shard_sizes is not None:
            x_proj = all_to_all_transpose(
                x_proj, -2, output_grid_shard_sizes, -1, channel_shard_sizes, model_comm_group
            )
        return einops.rearrange(x_proj, "(b t e) g v -> b t e g v", b=batch, t=time, e=ensemble)

    def inverse(self, x: Tensor) -> Tensor:
        raise NotImplementedError("CrossGridProjector does not support inverse projection.")
