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

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec

LOGGER = logging.getLogger(__name__)


class AnemoiEnsModelEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble functionality."""

    def _calculate_input_dim(self):
        base_input_dim = super()._calculate_input_dim()
        return base_input_dim + self.num_input_channels_prognostic + 1

    def _build_networks(self, model_config):
        super()._build_networks(model_config)
        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
        )

    def _assemble_input(self, x, fcstep, bse, grid_shard_shapes=None, model_comm_group=None):
        x_skip = x[:, -1, :, :, self._internal_input_idx]
        x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
        x_skip = self.truncation(x_skip, grid_shard_shapes, model_comm_group)

        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=bse)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        x_data_latent = torch.cat(
            (x_data_latent, torch.ones(x_data_latent.shape[:-1], device=x_data_latent.device).unsqueeze(-1) * fcstep),
            dim=-1,
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, bse, dtype):
        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=batch_size).to(dtype=dtype).clone()

        # residual connection (just for the prognostic variables)
        x_out[..., self._internal_output_idx] += einops.rearrange(
            x_skip,
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
        fcstep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward operator.

        Args:
            x: torch.Tensor
                Input tensor, shape (bs, m, e, n, f)
            fcstep: int
                Forecast step
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

        fcstep = min(1, fcstep)

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, fcstep, bse, grid_shard_shapes, model_comm_group
        )
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=bse)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group=model_comm_group)

        x_data_latent, x_latent = self.encoder(
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

        x_latent_proc = x_latent_proc + x_latent

        x_out = self.decoder(
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
