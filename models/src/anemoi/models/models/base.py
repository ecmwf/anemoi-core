# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiGraphModelBase(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()
        self._graph_data = graph_data
        self.data_indices = data_indices
        self.statistics = statistics
        self._truncation_data = truncation_data

        model_config = DotDict(model_config)
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden
        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        # build networks
        self._build_truncation(self._truncation_data)
        self._build_networks(model_config)
        self._build_boundings(model_config, self.data_indices, self.statistics)

    def _make_truncation_matrix(self, A, data_type=torch.float32):
        A_ = torch.sparse_coo_tensor(
            torch.tensor(np.vstack(A.nonzero()), dtype=torch.long),
            torch.tensor(A.data, dtype=data_type),
            size=A.shape,
        ).coalesce()
        return A_

    def _multiply_sparse(self, x, A):
        return torch.sparse.mm(A, x)

    def _truncate_fields(self, x, A, batch_size=None, auto_cast=False):
        if not batch_size:
            batch_size = x.shape[0]
        out = []
        with torch.amp.autocast(device_type="cuda", enabled=auto_cast):
            for i in range(batch_size):
                out.append(self._multiply_sparse(x[i, ...], A))
        return torch.stack(out)

    def _get_shard_shapes(self, x, dim=0, shard_shapes_dim=None, model_comm_group=None):
        if shard_shapes_dim is None:
            return get_shard_shapes(x, dim, model_comm_group)
        else:
            return apply_shard_shapes(x, dim, shard_shapes_dim)

    def _apply_truncation(self, x, grid_shard_shapes=None, model_comm_group=None):
        if self.A_down is not None or self.A_up is not None:
            if grid_shard_shapes is not None:
                shard_shapes = self._get_shard_shapes(x, 0, grid_shard_shapes, model_comm_group)
                # grid-sharded input: reshard to channel-shards to apply truncation
                x = shard_channels(x, shard_shapes, model_comm_group)  # we get the full sequence here

            # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
            # hence we check that they are on the correct device ; copy should only happen in the first forward run
            if self.A_down is not None:
                self.A_down = self.A_down.to(x.device)
                x = self._truncate_fields(x, self.A_down)  # to coarse resolution
            if self.A_up is not None:
                self.A_up = self.A_up.to(x.device)
                x = self._truncate_fields(x, self.A_up)  # back to high resolution

            if grid_shard_shapes is not None:
                # back to grid-sharding as before
                x = gather_channels(x, shard_shapes, model_comm_group)

        return x

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.model.input)
        self.num_output_channels = len(data_indices.model.output)
        self.num_input_channels_prognostic = len(data_indices.model.input.prognostic)
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic
        self.input_dim = (
            self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]
        )

    def _assert_matching_indices(self, data_indices: dict) -> None:
        assert len(self._internal_output_idx) == len(data_indices.model.output.full) - len(
            data_indices.model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the output indices excluding diagnostic variables "
            f"({len(data_indices.model.output.full) - len(data_indices.model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _assert_valid_sharding(
        self,
        batch_size: int,
        ensemble_size: int,
        in_out_sharded: bool,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> None:
        assert not (
            in_out_sharded and model_comm_group is None
        ), "If input is sharded, model_comm_group must be provided."

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

            assert (
                model_comm_group.size() == 1 or ensemble_size == 1
            ), "Ensemble size per device must be 1 when model is sharded across GPUs"

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        x_src_is_sharded : bool, optional
            Source data is sharded, by default False
        x_dst_is_sharded : bool, optional
            Destination data is sharded, by default False
        keep_x_dst_sharded : bool, optional
            Keep destination data sharded, by default False
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        kwargs = {
            "batch_size": batch_size,
            "shard_shapes": shard_shapes,
            "model_comm_group": model_comm_group,
            "x_src_is_sharded": x_src_is_sharded,
            "x_dst_is_sharded": x_dst_is_sharded,
            "keep_x_dst_sharded": keep_x_dst_sharded,
        }

        if isinstance(mapper, GraphTransformerBaseMapper) and mapper.shard_strategy == "edges":
            return mapper(data, **kwargs)

        return checkpoint(mapper, data, **kwargs, use_reentrant=use_reentrant)

    def _build_boundings(
        self,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
    ) -> None:
        """Builds the bounding functions for the model outputs."""
        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(
                    cfg,
                    name_to_index=data_indices.model.output.name_to_index,
                    statistics=statistics,
                    name_to_index_stats=data_indices.data.input.name_to_index,
                )
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

    def _build_truncation(self, truncation_data: dict) -> None:
        """Builds the truncation matrices for the model.

        Parameters
        ----------
        truncation_data : dict
            Truncation data containing down and up matrices
        """

        self.A_down, self.A_up = None, None
        if "down" in truncation_data:
            self.A_down = self._make_truncation_matrix(truncation_data["down"])
            LOGGER.info("Truncation: A_down %s", self.A_down.shape)
        if "up" in truncation_data:
            self.A_up = self._make_truncation_matrix(truncation_data["up"])
            LOGGER.info("Truncation: A_up %s", self.A_up.shape)

    @abstractmethod
    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the networks for the model."""
        pass

    @abstractmethod
    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        pass

    @abstractmethod
    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        pass

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : Tensor
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        Tensor
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        pass
