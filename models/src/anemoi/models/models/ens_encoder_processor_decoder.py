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

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiEnsModelEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble functionality."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )
        model_config = DotDict(model_config)
        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
        )

    def _calculate_input_dim(self, model_config):
        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]
        input_dim += self.num_input_channels_prognostic
        input_dim += 1
        return input_dim

    def _assemble_input(self, x, fcstep, bse):
        x_skip = x[:, -1, :, :, self._internal_input_idx]
        x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")

        # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
        # hence we check that they are on the correct device ; copy should only happen in the first forward run
        # todo -> parallelize this
        if self.A_down is not None:
            self.A_down = self.A_down.to(x.device)
            x_skip = self._truncate_fields(x_skip, self.A_down)  # to coarse resolution
        if self.A_up is not None:
            self.A_up = self.A_up.to(x.device)
            x_skip = self._truncate_fields(x_skip, self.A_up)  # back to high resolution

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars"),
                self.node_attributes(self._graph_name_data, batch_size=bse),
            ),
            dim=-1,  # feature dimension
        )
        x_data_latent = torch.cat(
            (x_data_latent, torch.ones(x_data_latent.shape[:-1], device=x_data_latent.device).unsqueeze(-1) * fcstep),
            dim=-1,
        )

        return x_data_latent, x_skip

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
        self, x: torch.Tensor, *, fcstep: int, model_comm_group: Optional[ProcessGroup] = None, **kwargs
    ) -> torch.Tensor:
        """Forward operator.

        Args:
            x: torch.Tensor
                Input tensor, shape (bs, m, e, n, f)
            fcstep: int
                Forecast step
            model_comm_group: Optional[ProcessGroup], optional
                Model communication group
        Returns:
            Output tensor
        """
        batch_size, ensemble_size = x.shape[0], x.shape[2]
        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged

        fcstep = min(1, fcstep)

        x_data_latent, x_skip = self._assemble_input(x, fcstep, bse)
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=bse)

        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
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

        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, bse, x.dtype)

        return x_out
