# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.layers.mapper import GraphTransformerBaseMapper
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
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

        self.input_dim = self._calculate_input_dim(model_config)
        self.input_dim_latent = self._calculate_input_dim_latent(model_config)

        # we can't register these as buffers because DDP does not support sparse tensors
        # these will be moved to the GPU when first used via sefl.interpolate_down/interpolate_up
        self.A_down, self.A_up = None, None
        if "down" in self._truncation_data:
            self.A_down = self._make_truncation_matrix(self._truncation_data["down"])
            LOGGER.info("Truncation: A_down %s", self.A_down.shape)
        if "up" in self._truncation_data:
            self.A_up = self._make_truncation_matrix(self._truncation_data["up"])
            LOGGER.info("Truncation: A_up %s", self.A_up.shape)

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.input_dim,
            in_channels_dst=self.input_dim_latent,
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.num_channels,
            in_channels_dst=self.input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_data],
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(
                    cfg,
                    name_to_index=self.data_indices.model.output.name_to_index,
                    statistics=self.statistics,
                    name_to_index_stats=self.data_indices.data.input.name_to_index,
                )
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

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

    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        x_skip = x[:, -1, ...]
        x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
        x_skip = self._apply_truncation(x_skip, grid_shard_shapes, model_comm_group)
        x_skip = einops.rearrange(x_skip, "(batch ensemble) grid vars -> batch ensemble grid vars", batch=batch_size)

        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=batch_size)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_data, 0, grid_shard_shapes, model_comm_group)
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def _calculate_input_dim(self, model_config):
        return self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]

    def _calculate_input_dim_latent(self, model_config):
        return self.node_attributes.attr_ndims[self._graph_name_hidden]

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.model.input)
        self.num_output_channels = len(data_indices.model.output)
        self.num_input_channels_prognostic = len(data_indices.model.input.prognostic)
        self._internal_input_idx = data_indices.model.input.prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

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
        **kwargs,
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
        mapper_args = {
            "batch_size": batch_size,
            "shard_shapes": shard_shapes,
            "model_comm_group": model_comm_group,
            "x_src_is_sharded": x_src_is_sharded,
            "x_dst_is_sharded": x_dst_is_sharded,
            "keep_x_dst_sharded": keep_x_dst_sharded,
            **kwargs,
        }
        if isinstance(mapper, GraphTransformerBaseMapper) and mapper.shard_strategy == "edges":
            return mapper(data, **mapper_args)  # finer grained checkpointing inside GTM with edge sharding
        return checkpoint(mapper, data, **mapper_args, use_reentrant=use_reentrant)

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
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        # Encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        # Processor
        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # Skip
        x_latent_proc = x_latent_proc + x_latent

        # Decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out

    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> Tensor:
        """Prediction step for the model.

        Base implementation applies pre-processing, performs a forward pass, and applies post-processing.
        Subclasses can override this for different behavior (e.g., sampling for diffusion models).

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data (before pre-processing)
        pre_processors : nn.Module,
            Pre-processing module
        post_processors : nn.Module,
            Post-processing module
        multi_step : int,
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        gather_out : bool
            Whether to gather output tensors across distributed processes
        **kwargs
            Additional arguments

        Returns
        -------
        Tensor
            Model output (after post-processing)
        """
        with torch.no_grad():
            batch = pre_processors(batch, in_place=False)
            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            # Dimensions are
            # batch, timesteps, grid, variables
            x = batch[:, 0:multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            # Handle distributed processing
            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            # Perform forward pass
            y_hat = self.forward(x, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes, **kwargs)

            # Apply post-processing
            y_hat = post_processors(y_hat, in_place=False)

            # Gather output if needed
            if gather_out and model_comm_group is not None:
                y_hat = gather_tensor(y_hat, -2, apply_shard_shapes(y_hat, -2, grid_shard_shapes), model_comm_group)

        return y_hat
