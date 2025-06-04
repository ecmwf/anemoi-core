# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import einops
import torch
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDecInterpolator(AnemoiModelEncProcDec):
    """Message passing interpolating graph neural network."""

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
        config : DotDict
            Job configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
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

        self.latent_skip = model_config.model.latent_skip
        self.grid_skip = model_config.model.grid_skip

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        super()._calculate_shapes_and_indices(data_indices)
        self.input_dim = (
            self.input_times * self.num_input_channels
            + self.node_attributes.attr_ndims[self._graph_name_data]
            + self.num_target_forcings
        )

    def _assemble_input(self, x, target_forcing, batch_size):
        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(target_forcing, "batch ensemble grid vars -> (batch ensemble grid) (vars)"),
                self.node_attributes(self._graph_name_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        if self.grid_skip is not None:
            x_skip = x[:, self.grid_skip, ...]
            if self.A_down is not None or self.A_up is not None:
                x_skip = einops.rearrange(x_skip, "batch ensemble grid vars -> (batch ensemble) grid vars")
                # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
                # hence we check that they are on the correct device ; copy should only happen in the first forward run
                if self.A_down is not None:
                    self.A_down = self.A_down.to(x_skip.device)
                    x_skip = self._truncate_fields(x_skip, self.A_down)  # to coarse resolution
                if self.A_up is not None:
                    self.A_up = self.A_up.to(x_skip.device)
                    x_skip = self._truncate_fields(x_skip, self.A_up)  # back to high resolution
                x_skip = einops.rearrange(
                    x_skip, "(batch ensemble) grid vars -> batch ensemble grid vars", batch=batch_size
                )
        else:
            x_skip = None

        return x_data_latent, x_skip

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
        if x_skip is not None:
            x_out[..., self._internal_output_idx] += x_skip[..., self._internal_input_idx]

        for bounding in self.boundings:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        self, x: Tensor, *, target_forcing: torch.Tensor, model_comm_group: Optional[ProcessGroup] = None, **kwargs
    ) -> Tensor:
        batch_size = x.shape[0]
        ensemble_size = x.shape[2]

        x_data_latent, x_skip = self._assemble_input(x, target_forcing, batch_size)
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out
