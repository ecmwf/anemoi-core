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
import torch
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecEnsInterp(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble
    and interpolator functionality."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:

        model_config = DotDict(model_config)
        self.num_target_forcings = (
            len(model_config.training.target_forcing.data) + model_config.training.target_forcing.time_fraction
        )
        self.input_times = len(model_config.training.explicit_times.input)

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )
        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
        )

        self.latent_skip = model_config.model.latent_skip
        self.grid_skip = model_config.model.grid_skip

    def _calculate_input_dim(self, model_config):
        return (
            self.input_times * self.num_input_channels
            + self.node_attributes.attr_ndims[self._graph_name_data]
            + self.num_target_forcings
        )

    def _assemble_input(self, x, target_forcing, bse, grid_shard_shapes=None, model_comm_group=None):

        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=bse)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_data, 0, grid_shard_shapes, model_comm_group)
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(target_forcing, "batch ensemble grid vars -> (batch ensemble grid) (vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        if self.grid_skip is not None:
            x_skip = x[:, self.grid_skip, ...]
            x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
            if self.A_down is not None or self.A_up is not None:
                x_skip = self._apply_truncation(x_skip, grid_shard_shapes, model_comm_group)
        else:
            x_skip = None

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, bse, dtype):
        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=batch_size).to(dtype=dtype).clone()

        # residual connection (just for the prognostic variables)
        if x_skip is not None:
            x_out[..., self._internal_output_idx] += einops.rearrange(
                x_skip[..., self._internal_input_idx],
                "(batch ensemble) grid var -> batch ensemble grid var",
                batch=batch_size,
            ).to(dtype=dtype)

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        self,
        x: torch.Tensor,
        *,
        target_forcing: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward operator.

        Args:
            x: torch.Tensor
                Input tensor, shape (bs, m, e, n, f)
            target_forcing: torch.Tensor
            model_comm_group: Optional[ProcessGroup], optional
                Model communication group
            grid_shard_shapes : list, optional
                Shard shapes of the grid, by default None
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor
        """
        batch_size, ensemble_size = x.shape[0], x.shape[2]
        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, target_forcing, bse, grid_shard_shapes, model_comm_group
        )
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=bse)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            noise_ref=x_hidden_latent,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=bse,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )

        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, bse, x.dtype)

        return x_out
