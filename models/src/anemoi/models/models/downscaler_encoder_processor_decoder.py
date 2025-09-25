# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from typing import Callable
from typing import Optional
from typing import Union

import einops
import torch
from hydra.utils import instantiate
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.diffusion_encoder_processor_decoder import (
    AnemoiDiffusionModelEncProcDec,
    AnemoiDiffusionTendModelEncProcDec,
)
from anemoi.models.samplers import diffusion_samplers
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiDownscalingModelEncProcDec(AnemoiDiffusionTendModelEncProcDec):
    """Downscaling Model."""

    def compute_residuals(
        self,
        y: torch.Tensor,
        x_in_interp_to_hres: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residuals between high-res target and interpolated low-res input.

        Parameters
        ----------

        y : torch.Tensor
            The high-resolution target tensor with shape (bs, ens, latlon, nvar)
        x_in_interp_to_hres : torch.Tensor
            The interpolated low-resolution input tensor with shape (bs, ens, latlon, n

        Returns
        -------
        torch.Tensor
            The residuals tensor output from model.
        """
        residuals = (
            x_in_interp_to_hres[..., self.data_indices.data.output.full]
            - y[..., self.data_indices.data.output.full]
        )

        # to deal with residuals or direct prediction, see compute_tendency
        # in diffusion_encoder_processor_decoder.py
        return residuals

    def _interpolate_to_high_res(
        self, x, grid_shard_shapes=None, model_comm_group=None
    ):

        if grid_shard_shapes is not None:
            shard_shapes = self._get_shard_shapes(
                x, 0, grid_shard_shapes, model_comm_group
            )
            # grid-sharded input: reshard to channel-shards to apply truncation
            x = shard_channels(
                x, shard_shapes, model_comm_group
            )  # we get the full sequence here

        # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
        # hence we check that they are on the correct device ; copy should only happen in the first forward run
        if self.A_down is not None:

            self.A_down = self.A_down.to(x.device)
            x = self._truncate_fields(x, self.A_down)  # back to high resolution
        else:
            raise ValueError("A_up not defined at model level.")

        if grid_shard_shapes is not None:
            # back to grid-sharding as before
            x = gather_channels(x, shard_shapes, model_comm_group)

        return x

    def apply_interpolate_to_high_res(
        self, x: torch.Tensor, grid_shard_shapes: list, model_comm_group: ProcessGroup
    ) -> torch.Tensor:
        """Apply interpolate to high res to the low res input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (bs, ens, latlon, nvar)

        Returns
        -------
        torch.Tensor
            Truncated tensor with same shape as input
        """
        bs, ens, _, _ = x.shape
        x_trunc = einops.rearrange(x, "bs ens latlon nvar -> (bs ens) latlon nvar")

        x_trunc = self._interpolate_to_high_res(
            x_trunc, grid_shard_shapes, model_comm_group
        )
        return einops.rearrange(
            x_trunc, "(bs ens) latlon nvar -> bs ens latlon nvar", bs=bs, ens=ens
        )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_lres_channels = len(data_indices.model.input[0])
        self.num_input_hres_channels = len(data_indices.model.input[1])
        self.num_output_channels = len(data_indices.model.output)
        self._internal_input_lres_idx = data_indices.model.input[0].prognostic
        self._internal_input_hres_idx = data_indices.model.input[1].prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

    def _calculate_input_dim(self, model_config):
        return (
            self.num_input_lres_channels
            + self.num_input_hres_channels
            + self.num_output_channels
            + +self.node_attributes.attr_ndims[self._graph_name_data]
        )  # input_lres + input_hres + noised targets + nodes_attributes

    def _assemble_input(
        self,
        x_in_lres_interp_hres,
        x_in_hres,
        y_noised,
        bse,
        grid_shard_shapes=None,
        model_comm_group=None,
    ):
        node_attributes_data = self.node_attributes(
            self._graph_name_data, batch_size=bse
        )
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(
                node_attributes_data, 0, grid_shard_shapes, model_comm_group
            )
            node_attributes_data = shard_tensor(
                node_attributes_data, 0, shard_shapes_nodes, model_comm_group
            )

        # combine noised target, input state, noise conditioning and add data positional info (lat/lon)

        x_data_latent = torch.cat(
            (
                einops.rearrange(
                    x_in_lres_interp_hres,
                    "batch time ensemble grid vars -> (batch ensemble grid) (time  vars)",
                ),
                einops.rearrange(
                    x_in_hres,
                    "batch  time ensemble grid vars -> (batch ensemble grid) (time  vars)",
                ),
                einops.rearrange(
                    y_noised,
                    "batch  time ensemble grid vars -> (batch ensemble grid) (time  vars)",
                ),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = self._get_shard_shapes(
            x_data_latent, 0, grid_shard_shapes, model_comm_group
        )

        return x_data_latent, None, shard_shapes_data

    def _assert_matching_indices(self, data_indices: dict) -> None:
        pass

    def forward(
        self,
        x_in_lres_interp_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:

        batch_size, ensemble_size = (
            x_in_lres_interp_hres.shape[0],
            x_in_lres_interp_hres.shape[2],
        )
        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(
            batch_size, ensemble_size, in_out_sharded, model_comm_group
        )

        # prepare noise conditionings
        c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(sigma)
        shape_c_data = get_shard_shapes(c_data, 0, model_comm_group)
        shape_c_hidden = get_shard_shapes(c_hidden, 0, model_comm_group)

        c_data = shard_tensor(c_data, 0, shape_c_data, model_comm_group)
        c_hidden = shard_tensor(c_hidden, 0, shape_c_hidden, model_comm_group)

        fwd_mapper_kwargs = {"cond": (c_data, c_hidden)}
        processor_kwargs = {"cond": c_hidden}
        bwd_mapper_kwargs = {"cond": (c_hidden, c_data)}

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x_in_lres_interp_hres,
            x_in_hres,
            y_noised,
            bse,
            grid_shard_shapes,
            model_comm_group,
        )
        x_hidden_latent = self.node_attributes(
            self._graph_name_hidden, batch_size=batch_size
        )
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
            **fwd_mapper_kwargs,
        )

        x_latent_proc = self.processor(
            x=x_latent,
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
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
            **bwd_mapper_kwargs,
        )

        x_out = self._assemble_output(
            x_out, x_skip, batch_size, ensemble_size, x_in_lres_interp_hres.dtype
        )

        return x_out

    def fwd_with_preconditioning(
        self,
        x_in_lres_interp_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-conditioning of EDM diffusion model."""
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(sigma, self.sigma_data)
        pred = self(
            x_in_lres_interp_hres,
            x_in_hres,
            (c_in * y_noised),
            c_noise,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
        )  # calls forward ...
        D_x = c_skip * y_noised + c_out * pred

        return D_x
