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
from anemoi.models.models.base import BaseGraphModel
from anemoi.models.samplers import diffusion_samplers
from anemoi.utils.config import DotDict


LOGGER = logging.getLogger(__name__)


class AnemoiDownscalingModelEncProcDec(AnemoiDiffusionModelEncProcDec):
    """Downscaling Diffusion Model."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

    def _assemble_input(
        self,
        x_lres: torch.Tensor,
        x_hres_forcings: torch.Tensor,
        y_noised: torch.Tensor,
        bse: int,
        grid_shard_shapes: dict | None = None,
        model_comm_group=None,
        dataset_name="hres",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """Assemble inputs for downscaling: concatenate lres (upsampled) + hres_forcings.

        Parameters
        ----------
        x_lres : torch.Tensor
            Low-resolution input, already upsampled to hres grid, shape (batch, time, ensemble, grid, vars)
        x_hres_forcings : torch.Tensor
            High-resolution forcings, shape (batch, time, ensemble, grid, vars)
        y_noised : torch.Tensor
            Noised target, shape (batch, ensemble, grid, vars)
        bse : int
            Batch size * ensemble size
        grid_shard_shapes : dict | None
            Shard shapes for distributed processing
        model_comm_group : ProcessGroup
            Communication group
        dataset_name : str
            Name of the output dataset (default "hres")

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]
            Assembled input tensor, skip connection tensor, shard shapes
        """
        assert dataset_name is not None, "dataset_name must be provided."

        # Get node attributes for the data nodes
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=bse)
        grid_shard_shapes_data = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

        # Compute skip connection (for residual prediction)
        x_skip = self.residual[dataset_name](x_lres, grid_shard_shapes_data, model_comm_group)

        # Shard node attributes if grid sharding is enabled
        if grid_shard_shapes_data is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes_data, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # Reshape inputs: combine batch and ensemble dimensions
        # x_lres: low-res input (already upsampled to hres)
        x_lres_reshaped = einops.rearrange(x_lres, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")

        # x_hres_forcings: high-res forcings
        x_hres_forcings_reshaped = einops.rearrange(
            x_hres_forcings, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"
        )

        # y_noised: noised target
        y_noised_reshaped = einops.rearrange(y_noised, "batch ensemble grid vars -> (batch ensemble grid) vars")

        # Concatenate all inputs along feature dimension:
        # [lres upsampled, hres forcings, noised target, node attributes (lat/lon)]
        x_data_latent = torch.cat(
            (
                x_lres_reshaped,
                x_hres_forcings_reshaped,
                y_noised_reshaped,
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )

        # Get shard shapes for the assembled data
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes_data, model_comm_group=model_comm_group
        )

        return x_data_latent, x_skip, shard_shapes_data

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
            i for i in self.data_indices.data.output.full if i not in set(list_indices_direct_prediction)
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
            y[..., self.data_indices.data.output.full] - x_in_interp_to_hres[..., self.data_indices.data.output.full]
        )

        # to deal with residuals or direct prediction, see compute_tendency
        # in diffusion_encoder_processor_decoder.py
        return residuals

    def _interpolate_to_high_res(self, x, grid_shard_shapes=None, model_comm_group=None):

        if grid_shard_shapes is not None:
            shard_shapes = self._get_shard_shapes(x, 0, grid_shard_shapes, model_comm_group)
            # grid-sharded input: reshard to channel-shards to apply truncation
            x = shard_channels(x, shard_shapes, model_comm_group)  # we get the full sequence here

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

        x_trunc = self._interpolate_to_high_res(x_trunc, grid_shard_shapes, model_comm_group)
        return einops.rearrange(x_trunc, "(bs ens) latlon nvar -> bs ens latlon nvar", bs=bs, ens=ens)

    def _before_sampling(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict]:
        """Prepare batch before sampling (prediction/inference mode).

        During prediction, lres comes at low resolution and needs upsampling.
        During training, lres is already upsampled in the training code.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batch dictionary with keys "lres", "hres_forcings", "hres"
            Each tensor has shape (batch, timesteps, grid, variables)
        pre_processors : dict[str, nn.Module]
            Dictionary of pre-processing modules per dataset
        multi_step : int
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        **kwargs
            Additional parameters

        Returns
        -------
        tuple[tuple[torch.Tensor, torch.Tensor], dict]
            Tuple of (x_lres_upsampled, x_hres_forcings) and grid shard shapes dict
        """
        # Extract inputs from batch
        x_lres = batch.get("lres")  # Low-res input
        x_hres_forcings = batch.get("hres_forcings")  # High-res forcings

        if x_lres is None or x_hres_forcings is None:
            raise ValueError("Batch must contain 'lres' and 'hres_forcings' keys")

        # Add dummy ensemble dimension as 3rd index if not present
        # Expected shape: (batch, timesteps, ensemble, grid, variables)
        if x_lres.ndim == 4:
            x_lres = x_lres[:, 0:multi_step, None, ...]
        if x_hres_forcings.ndim == 4:
            x_hres_forcings = x_hres_forcings[:, 0:multi_step, None, ...]

        # Grid sharding setup
        grid_shard_shapes = {}

        # Shard lres input
        lres_grid_shard_shapes = None
        if model_comm_group is not None:
            lres_shard_shapes = get_shard_shapes(x_lres, -2, model_comm_group=model_comm_group)
            lres_grid_shard_shapes = [shape[-2] for shape in lres_shard_shapes]
            x_lres = shard_tensor(x_lres, -2, lres_shard_shapes, model_comm_group)

        # Upsample lres to hres grid (only during prediction - training already has it upsampled)
        x_lres_upsampled = self.apply_interpolate_to_high_res(
            x_lres[:, :, 0, ...],  # Remove ensemble dim temporarily for interpolation
            grid_shard_shapes=lres_grid_shard_shapes,
            model_comm_group=model_comm_group,
        )
        # Add ensemble dimension back
        x_lres_upsampled = x_lres_upsampled[:, :, None, ...]

        # Now both should be at hres grid - setup hres grid sharding
        hres_grid_shard_shapes = None
        if model_comm_group is not None:
            hres_shard_shapes = get_shard_shapes(x_hres_forcings, -2, model_comm_group=model_comm_group)
            hres_grid_shard_shapes = [shape[-2] for shape in hres_shard_shapes]
            x_hres_forcings = shard_tensor(x_hres_forcings, -2, hres_shard_shapes, model_comm_group)

            # Also shard the upsampled lres to match hres sharding
            x_lres_upsampled_shard_shapes = get_shard_shapes(x_lres_upsampled, -2, model_comm_group=model_comm_group)
            x_lres_upsampled = shard_tensor(x_lres_upsampled, -2, x_lres_upsampled_shard_shapes, model_comm_group)

        # Apply preprocessing
        x_lres_upsampled = pre_processors["lres"](x_lres_upsampled, in_place=False)
        x_hres_forcings = pre_processors["hres_forcings"](x_hres_forcings, in_place=False)

        # Store hres grid shard shapes for all datasets
        grid_shard_shapes["lres"] = hres_grid_shard_shapes
        grid_shard_shapes["hres_forcings"] = hres_grid_shard_shapes
        grid_shard_shapes["hres"] = hres_grid_shard_shapes

        return (x_lres_upsampled, x_hres_forcings), grid_shard_shapes

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        post_processors: dict[str, nn.Module],
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
        batch : dict[str, torch.Tensor]
            Input batched data dictionary (before pre-processing) with keys "lres", "hres_forcings"
        pre_processors : dict[str, nn.Module]
            Dictionary of pre-processing modules per dataset
        post_processors : dict[str, nn.Module]
            Dictionary of post-processing modules per dataset
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

            # Before sampling hook - will upsample lres and preprocess both inputs
            before_sampling_data, grid_shard_shapes = self._before_sampling(
                batch,
                pre_processors,
                multi_step,
                model_comm_group,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies,
                **kwargs,
            )

            x_lres_upsampled = before_sampling_data[0]
            x_hres_forcings = before_sampling_data[1]

            out = self.sample(
                x_lres_upsampled,
                x_hres_forcings,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                noise_scheduler_params=noise_scheduler_params,
                sampler_params=sampler_params,
                **kwargs,
            ).to(x_lres_upsampled.dtype)

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
        y_init = torch.randn(shape, device=x_in_interp_to_hres.device, dtype=sigmas.dtype) * sigmas[0]

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

    def _get_preconditioning(self, sigma: dict[str, torch.Tensor], sigma_data: torch.Tensor) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Compute preconditioning factors."""
        c_skip, c_out, c_in, c_noise = {}, {}, {}, {}
        for dataset_name, sigma_i in sigma.items():
            c_skip[dataset_name] = sigma_data**2 / (sigma_i**2 + sigma_data**2)
            c_out[dataset_name] = sigma_i * sigma_data / (sigma_i**2 + sigma_data**2) ** 0.5
            c_in[dataset_name] = 1.0 / (sigma_data**2 + sigma_i**2) ** 0.5
            c_noise[dataset_name] = sigma_i.log() / 4.0

        return c_skip, c_out, c_in, c_noise

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.multi_step,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes

            rel_date_indices = md_dict["metadata_inference"][dataset]["timesteps"]["relative_date_indices_training"]
            input_rel_date_indices = rel_date_indices[:-1]
            output_rel_date_indices = rel_date_indices[-1]
            md_dict["metadata_inference"][dataset]["timesteps"]["input_relative_date_indices"] = input_rel_date_indices
            md_dict["metadata_inference"][dataset]["timesteps"][
                "output_relative_date_indices"
            ] = output_rel_date_indices
