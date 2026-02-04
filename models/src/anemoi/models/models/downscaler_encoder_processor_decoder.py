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
import time

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

    def compute_residuals_advanced(
        self,
        y: torch.Tensor,
        x_in_interp_to_hres: torch.Tensor,
        pre_processors_state: Callable,
        list_indices_direct_prediction: list,
    ) -> torch.Tensor:
        """Compute the tendency from two states.

        Parameters
        ----------
        y : torch.Tensor
            The high-resolution target tensor with shape (bs, ens, latlon, nvar)
        x_in_interp_to_hres : torch.Tensor
            The interpolated low-resolution input tensor with shape (bs, ens, latlon, nvar)
        pre_processors_state : callable
            Function to pre-process the state variables.
        list_indices_direct_prediction : list
            List of indices for direct prediction (not computed as residuals).

        Returns
        -------
        torch.Tensor
            The residuals tensor output from model.
        """

        inverse_indices = [
            i
            for i in self.data_indices.data.output.full
            if i not in set(list_indices_direct_prediction)
        ]

        mask = y.new_zeros(y.shape[-1])  # dtype/device matches y
        mask[inverse_indices] = 1

        # residuals = y for direct channels, and y - x for inverse channels
        residuals = (
            y[..., self.data_indices.data.output.full]
            - x_in_interp_to_hres[..., self.data_indices.data.output.full] * mask
        )

        norm_target = pre_processors_state(residuals, dataset="output", in_place=False)
        return norm_target

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
            The interpolated low-resolution input tensor with shape (bs, ens, latlon, nvar)

        Returns
        -------
        torch.Tensor
            The residuals tensor output from model.
        """
        residuals = (
            y[..., self.data_indices.data.output.full]
            - x_in_interp_to_hres[..., self.data_indices.data.output.full]
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
        c_noise: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        start_init = time.time()
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
        c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(c_noise)
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

        # time_encoder = time.time()
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
        # print("time in encoder", time.time() - time_encoder)
        # time_processor = time.time()
        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=bse,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )
        # print("time in processor", time.time() - time_processor)

        x_latent_proc = x_latent_proc + x_latent
        # time_decoder = time.time()

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
        # print("time in decoder", time.time() - time_decoder)

        x_out = self._assemble_output(
            x_out, x_skip, batch_size, ensemble_size, x_in_lres_interp_hres.dtype
        )
        # print("time in model forward step", time.time() - start_init)
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

    def _before_sampling(
        self,
        x_in: torch.Tensor,
        x_in_hres: torch.Tensor,
        pre_processors: nn.Module,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor, ...]], Optional[list]]:
        """Prepare batch before sampling.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch after pre-processing
        pre_processors : nn.Module
            Pre-processing module (already applied)
        multi_step : int
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        **kwargs
            Additional parameters for subclasses

        Returns
        -------
        tuple[Union[torch.Tensor, tuple[torch.Tensor, ...]], Optional[list]]
            Prepared input tensor(s) and grid shard shapes.
            Can return a single tensor or tuple of tensors for sampling input.
        """

        lres_grid_shard_shapes = None
        if model_comm_group is not None:
            lres_shard_shapes = get_shard_shapes(x_in, -2, model_comm_group)
            x = shard_tensor(x, -2, lres_shard_shapes, model_comm_group)
            lres_grid_shard_shapes = [shape[-2] for shape in lres_shard_shapes]
        hres_grid_shard_shapes = None
        if model_comm_group is not None:
            hres_shard_shapes = get_shard_shapes(x_in_hres, -2, model_comm_group)
            x_in_hres = shard_tensor(x_in_hres, -2, hres_shard_shapes, model_comm_group)
            hres_grid_shard_shapes = [shape[-2] for shape in hres_shard_shapes]

        x_in_interp_to_hres = self.apply_interpolate_to_high_res(
            x_in[:, 0, ...],
            grid_shard_shapes=lres_grid_shard_shapes,
            model_comm_group=model_comm_group,
        )[:, None, ...]

        x_in_interp_to_hres = pre_processors(
            x_in_interp_to_hres, dataset="input_lres", in_place=False
        )
        x_in_hres = pre_processors(x_in_hres, dataset="input_hres", in_place=False)

        return (x_in_interp_to_hres, x_in_hres), hres_grid_shard_shapes

    def predict_step(
        self,
        x_in_lres: torch.Tensor,
        x_in_hres: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        pre_processors_tendencies: Optional[nn.Module] = None,
        post_processors_tendencies: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Prediction step for flow/diffusion models - performs sampling.

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
        noise_scheduler_params : Optional[dict]
            Dictionary of noise scheduler parameters (schedule_type, sigma_max, sigma_min, rho, num_steps, etc.)
            These will override the default values from inference_defaults
        sampler_params : Optional[dict]
            Dictionary of sampler parameters (sampler, S_churn, S_min, S_max, S_noise, etc.)
            These will override the default values from inference_defaults
        pre_processors_tendencies : Optional[nn.Module]
            Pre-processing module for tendencies (used by subclasses)
        post_processors_tendencies : Optional[nn.Module]
            Post-processing module for tendencies (used by subclasses)
        **kwargs
            Additional sampling parameters

        Returns
        -------
        torch.Tensor
            Sampled output (after post-processing)
        """

        noise_scheduler_params = {
            "schedule_type": "karras",
            # "sigma_max": 88,
            "sigma_max": 100000,
            "sigma_min": 0.03,
            "rho": 7.0,
            "num_steps": 80,
        }

        sampler_params = {
            "sampler": "heun",
            "S_churn": 2.5,
            "S_min": 0.75,
            # "S_max": 88,
            "S_max": 100000,
            "S_noise": 1.05,
        }

        print("noise_scheduler_params:", noise_scheduler_params)
        print("sampler_params:", sampler_params)

        with torch.no_grad():

            if len(x_in_lres.shape) == 4:
                x_in_lres = x_in_lres[:, None, ...]
            if len(x_in_hres.shape) == 4:
                x_in_hres = x_in_hres[:, None, ...]
            assert (
                len(x_in_lres.shape) == 5
            ), f"The input tensor has an incorrect shape: expected a 5-dimensional tensor, got {x_in_lres.shape}!"

            assert (
                len(x_in_hres.shape) == 5
            ), f"The input tensor has an incorrect shape: expected a 5-dimensional tensor, got {x_in_hres.shape}!"

            # Before sampling hook
            before_sampling_data, grid_shard_shapes = self._before_sampling(
                x_in_lres,
                x_in_hres,
                pre_processors,
                multi_step,
                model_comm_group,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies,
                **kwargs,
            )

            x_in_interp_to_hres = before_sampling_data[0]
            x_in_hres = before_sampling_data[1]

            out = self.sample(
                x_in_interp_to_hres,
                x_in_hres,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                noise_scheduler_params=noise_scheduler_params,
                sampler_params=sampler_params,
                **kwargs,
            ).to(x_in_interp_to_hres.dtype)

            # After sampling hook
            out = self._after_sampling(
                out,
                post_processors,
                before_sampling_data,
                model_comm_group,
                grid_shard_shapes,
                gather_out,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies,
                **kwargs,
            )

        return out

    def sample(
        self,
        x_in_interp_to_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Parameters
        ----------
        x : torch.Tensor
            Input conditioning data with shape (batch, time, ensemble, grid, vars)
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : Optional[list]
            Grid shard shapes for distributed processing
        noise_scheduler_params : Optional[dict]
            Dictionary of noise scheduler parameters (schedule_type, num_steps, sigma_max, etc.) to override defaults
        sampler_params : Optional[dict]
            Dictionary of sampler parameters (sampler, S_churn, S_min, etc.) to override defaults
        **kwargs
            Additional sampler-specific arguments

        Returns
        -------
        torch.Tensor
            Sampled output with shape (batch, ensemble, grid, vars)
        """

        # Start with inference defaults
        noise_scheduler_config = dict(self.inference_defaults.noise_scheduler)

        # Override config with provided noise scheduler parameters
        if noise_scheduler_params is not None:
            noise_scheduler_config.update(noise_scheduler_params)

        warnings.warn(f"noise_scheduler_config: {noise_scheduler_config}")
        print(f"noise_scheduler_config: {noise_scheduler_config}")

        # Remove schedule_type (used for class selection, not constructor)
        actual_schedule_type = noise_scheduler_config.pop("schedule_type")

        if actual_schedule_type not in diffusion_samplers.NOISE_SCHEDULERS:
            raise ValueError(f"Unknown schedule type: {actual_schedule_type}")

        scheduler_cls = diffusion_samplers.NOISE_SCHEDULERS[actual_schedule_type]
        scheduler = scheduler_cls(**noise_scheduler_config)
        sigmas = scheduler.get_schedule(x_in_interp_to_hres.device, torch.float64)

        # Initialize output with noise
        batch_size, ensemble_size, grid_size = (
            x_in_interp_to_hres.shape[0],
            x_in_interp_to_hres.shape[2],
            x_in_interp_to_hres.shape[-2],
        )
        time_size = 1
        shape = (
            batch_size,
            time_size,
            ensemble_size,
            grid_size,
            self.num_output_channels,
        )
        y_init = (
            torch.randn(shape, device=x_in_interp_to_hres.device, dtype=sigmas.dtype)
            * sigmas[0]
        )

        print("sigmas", sigmas)

        # Build diffusion sampler config dict from all inference defaults
        diffusion_sampler_config = dict(self.inference_defaults.diffusion_sampler)

        # Override config with provided sampler parameters
        if sampler_params is not None:
            diffusion_sampler_config.update(sampler_params)

        warnings.warn(f"diffusion_sampler_config: {diffusion_sampler_config}")

        # Remove sampler name (used for class selection, not constructor)
        actual_sampler = diffusion_sampler_config.pop("sampler")

        if actual_sampler not in diffusion_samplers.DIFFUSION_SAMPLERS:
            raise ValueError(f"Unknown sampler: {actual_sampler}")

        sampler_cls = diffusion_samplers.DIFFUSION_SAMPLERS[actual_sampler]
        sampler_instance = sampler_cls(dtype=sigmas.dtype, **diffusion_sampler_config)

        return sampler_instance.sample(
            x_in_interp_to_hres,
            x_in_hres,
            y_init,
            sigmas,
            self.fwd_with_preconditioning,
            grid_shard_shapes=grid_shard_shapes,
            model_comm_group=model_comm_group,
        )

    def add_interp_to_state(
        self,
        state_inp: torch.Tensor,
        residuals: torch.Tensor,
        post_processors_state: Callable,
        post_processors_tendencies: Callable,
        output_pre_processor: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Add the tendency to the state.

        Parameters
        ----------
        state_inp : torch.Tensor
            The normalized input state tensor with full input variables.
        residuals : torch.Tensor
            The normalized residuals tensor output from model.
        post_processors_state : callable
            Function to post-process the state variables.
        post_processors_tendencies : callable
            Function to post-process the tendency variables.
            Not used for dowsncaling but kept for compatibility
        output_pre_processor : Optional[Callable], optional
            Function to pre-process the output state. If provided,
            the output state will be pre-processed before returning.
            If None, the output state is returned directly. Default is None.

        Returns
        -------
        torch.Tensor
            the de-normalised state
        """
        state_outp = post_processors_state(residuals, dataset="output", in_place=False)

        state_outp += post_processors_state(
            state_inp,
            dataset="input_lres",
            in_place=False,
        )
        return state_outp

    def correction_bug_to_delete(self, out, y_pred_residuals, post_processors_state):
        """Temporary fix to correct a bug in the addition of residuals to state"""
        out = out - 2 * post_processors_state(
            y_pred_residuals,
            dataset="output",
            in_place=False,
        )
        return out

    def _after_sampling(
        self,
        out: torch.Tensor,
        post_processors: nn.Module,
        before_sampling_data: Union[torch.Tensor, tuple[torch.Tensor, ...]],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        gather_out: bool = True,
        post_processors_tendencies: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process sampled tendency to get state prediction.

        Override to convert tendency to state using x_t0.
        """
        if isinstance(before_sampling_data, tuple) and len(before_sampling_data) >= 2:
            x_in_interp = before_sampling_data[0]
        else:
            raise ValueError("Expected before_sampling_data to contain x_in_interp")

        # Convert tendency to state
        residuals = out.clone()
        out = self.add_interp_to_state(
            x_in_interp,
            out,
            post_processors,
            post_processors_tendencies,
        )
        """
        out = self.correction_bug_to_delete(
            out, residuals, post_processors
        )  # to be deleted
        """

        # Gather if needed
        if gather_out and model_comm_group is not None:
            out = gather_tensor(
                out,
                -2,
                apply_shard_shapes(out, -2, grid_shard_shapes),
                model_comm_group,
            )

        return out
