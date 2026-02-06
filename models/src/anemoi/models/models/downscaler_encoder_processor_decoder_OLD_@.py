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
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models.diffusion_encoder_processor_decoder import AnemoiDiffusionModelEncProcDec
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

    def _build_residual(self, residual_config):
        """Build residual connections with per-dataset configs.

        Overrides parent to properly handle dictionary of per-dataset residual configs.
        """
        from hydra.utils import instantiate
        import torch.nn as nn

        if isinstance(residual_config, dict) and "_target_" not in residual_config:
            # Per-dataset configs provided
            self.residual = nn.ModuleDict()
            for dataset_name in self._graph_data.keys():
                if dataset_name in residual_config:
                    self.residual[dataset_name] = instantiate(
                        residual_config[dataset_name], graph=self._graph_data[dataset_name]
                    )
                else:
                    # Use default SkipConnection if not specified
                    from anemoi.models.layers.residual import SkipConnection

                    self.residual[dataset_name] = SkipConnection(step=-1)
        else:
            # Single config for all datasets - use parent implementation
            super()._build_residual(residual_config)

    def _calculate_input_dim(self, dataset_name: str) -> int:
        """Calculate input dimension for downscaler.

        For downscaler, the encoder input concatenates:
        - x_in_lres (upsampled): multi_step * num_channels_in_lres
        - x_in_hres (forcings): multi_step * num_channels_in_hres
        - y_noised (target): multi_step * num_channels_out_hres
        - node_attributes (lat/lon etc)
        """
        num_channels_in_lres = len(self.data_indices["in_lres"].model.input)
        num_channels_in_hres = len(self.data_indices["in_hres"].model.input)
        num_channels_out_hres = len(self.data_indices["out_hres"].model.output)

        input_dim = (
            self.multi_step * num_channels_in_lres
            + self.multi_step * num_channels_in_hres
            + self.multi_step * num_channels_out_hres
            + self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
        )
        return input_dim

    def forward(
        self,
        x_in_lres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict[str, list]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for downscaling with two separate inputs.

        Parameters
        ----------
        x_in_lres : torch.Tensor
            Low-resolution input (already upsampled to hres during training)
        x_in_hres : torch.Tensor
            High-resolution forcings
        y_noised : torch.Tensor
            Noised target
        sigma : torch.Tensor
            Noise level
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : Optional[dict[str, list]]
            Grid shard shapes for distributed processing
        **kwargs
            Additional arguments

        Returns
        -------
        torch.Tensor
            Model prediction
        """
        # Multi-dataset case - use "out_hres" as the output dataset name
        dataset_name = "out_hres"

        # Extract and validate batch & ensemble sizes
        batch_size = x_in_lres.shape[0]
        ensemble_size = x_in_lres.shape[2]

        bse = batch_size * ensemble_size
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # Prepare noise conditioning
        c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(
            sigma, dataset_name=dataset_name, edge_conditioning=False
        )
        shape_c_data = get_shard_shapes(c_data, 0, model_comm_group=model_comm_group)
        shape_c_hidden = get_shard_shapes(c_hidden, 0, model_comm_group=model_comm_group)

        c_data = shard_tensor(c_data, 0, shape_c_data, model_comm_group)
        c_hidden = shard_tensor(c_hidden, 0, shape_c_hidden, model_comm_group)

        fwd_mapper_kwargs = {"cond": (c_data, c_hidden)}
        processor_kwargs = {"cond": c_hidden}
        bwd_mapper_kwargs = {"cond": (c_hidden, c_data)}

        # Assemble input with two separate inputs
        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x_in_lres, x_in_hres, y_noised, bse, grid_shard_shapes, model_comm_group, dataset_name
        )

        x_hidden_latent = self.node_attributes[dataset_name](self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group=model_comm_group)

        encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
            dataset_name
        ].get_edges(
            batch_size=bse,
            model_comm_group=model_comm_group,
        )

        x_data_latent, x_latent = self.encoder[dataset_name](
            (x_data_latent, x_hidden_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            edge_attr=encoder_edge_attr,
            edge_index=encoder_edge_index,
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
            edge_shard_shapes=enc_edge_shard_shapes,
            **fwd_mapper_kwargs,
        )

        # Processor
        processor_edge_attr, processor_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=bse,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=bse,
            shard_shapes=shard_shapes_hidden,
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
            **processor_kwargs,
        )

        # Decoder
        decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
            dataset_name
        ].get_edges(
            batch_size=bse,
            model_comm_group=model_comm_group,
        )

        x_out = self.decoder[dataset_name](
            (x_latent_proc, x_data_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            edge_attr=decoder_edge_attr,
            edge_index=decoder_edge_index,
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,
            x_dst_is_sharded=in_out_sharded,
            edge_shard_shapes=dec_edge_shard_shapes,
            **bwd_mapper_kwargs,
        )

        # Assemble output
        dtype = x_in_lres.dtype
        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, dtype)

        return x_out

    def fwd_with_preconditioning(
        self,
        x_in_lres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict[str, list]] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-conditioning for downscaling.

        Parameters
        ----------
        x_in_lres : torch.Tensor
            Low-resolution input (already upsampled to hres during training)
        x_in_hres : torch.Tensor
            High-resolution forcings
        y_noised : torch.Tensor
            Noised target
        sigma : torch.Tensor
            Noise level
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : Optional[dict[str, list]]
            Grid shard shapes for distributed processing

        Returns
        -------
        torch.Tensor
            Preconditioned model prediction
        """
        # Compute preconditioning factors
        # For downscaler, we use a dict with single dataset "out_hres"
        sigma_dict = {"out_hres": sigma}
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(sigma_dict, self.sigma_data)

        # Apply preconditioning to noised input
        y_noised_precond = c_in["out_hres"] * y_noised

        # Forward pass
        pred = self(
            x_in_lres,
            x_in_hres,
            y_noised_precond,
            c_noise["out_hres"],
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
        )

        # Apply output preconditioning
        D_x = c_skip["out_hres"] * y_noised + c_out["out_hres"] * pred

        return D_x

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        """Assemble output with time dimension preserved.

        Overrides parent method to reshape with time dimension.
        Output format: (batch, time, ensemble, grid, vars)
        """
        import einops

        # Reshape from flat to structured format
        x_out = einops.rearrange(
            x_out,
            "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
            batch=batch_size,
            ensemble=ensemble_size,
            time=1,  # Currently single-step, but supports future multi-step
        ).to(dtype=dtype)

        return x_out

    def _assemble_input(
        self,
        x_in_lres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        bse: int,
        grid_shard_shapes: dict | None = None,
        model_comm_group=None,
        dataset_name="out_hres",
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """Assemble inputs for downscaling: concatenate in_lres (upsampled) + in_hres.

        Parameters
        ----------
        x_in_lres : torch.Tensor
            Low-resolution input, already upsampled to hres grid, shape (batch, time, ensemble, grid, vars)
        x_in_hres : torch.Tensor
            High-resolution forcings, shape (batch, time, ensemble, grid, vars)
        y_noised : torch.Tensor
            Noised target, shape (batch, time, ensemble, grid, vars)
        bse : int
            Batch size * ensemble size
        grid_shard_shapes : dict | None
            Shard shapes for distributed processing
        model_comm_group : ProcessGroup
            Communication group
        dataset_name : str
            Name of the output dataset (default "out_hres")

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
        x_skip = self.residual[dataset_name](x_in_lres, grid_shard_shapes_data, model_comm_group)

        # Shard node attributes if grid sharding is enabled
        if grid_shard_shapes_data is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes_data, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # Reshape inputs: combine batch and ensemble dimensions
        # x_in_lres: low-res input (already upsampled to hres)
        x_in_lres_reshaped = einops.rearrange(
            x_in_lres, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"
        )

        # x_in_hres: high-res forcings
        x_in_hres_reshaped = einops.rearrange(
            x_in_hres, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"
        )

        # y_noised: noised target (with time dimension)
        y_noised_reshaped = einops.rearrange(
            y_noised, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"
        )

        # Concatenate all inputs along feature dimension:
        # [in_lres upsampled, in_hres forcings, noised target, node attributes (lat/lon)]
        x_data_latent = torch.cat(
            (
                x_in_lres_reshaped,
                x_in_hres_reshaped,
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

        During prediction, in_lres comes at low resolution and needs upsampling.
        During training, in_lres is already upsampled in the training code.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batch dictionary with keys "in_lres", "in_hres", "out_hres"
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
            Tuple of (x_in_lres_upsampled, x_in_hres) and grid shard shapes dict
        """
        # Extract inputs from batch
        x_in_lres = batch.get("in_lres")  # Low-res input
        x_in_hres = batch.get("in_hres")  # High-res forcings

        if x_in_lres is None or x_in_hres is None:
            raise ValueError("Batch must contain 'in_lres' and 'in_hres' keys")

        # Add dummy ensemble dimension as 3rd index if not present
        # Expected shape: (batch, timesteps, ensemble, grid, variables)
        if x_in_lres.ndim == 4:
            x_in_lres = x_in_lres[:, 0:multi_step, None, ...]
        if x_in_hres.ndim == 4:
            x_in_hres = x_in_hres[:, 0:multi_step, None, ...]

        # ============================================================================
        # GRID SHARDING SETUP
        # ============================================================================
        # Grid sharding is based on PHYSICAL GRID RESOLUTION (lres vs hres)
        # NOT on dataset roles (input vs output)
        #
        # Flow:
        # 1. in_lres starts at lres resolution -> gets upsampled -> becomes hres
        # 2. in_hres starts at hres resolution -> stays hres
        # 3. Result: ALL data is at hres resolution after upsampling
        # ============================================================================

        # Upsample in_lres from lres -> hres using residual connection
        # Follow the same pattern as training: residual removes ensemble, we add it back
        x_in_lres_upsampled = self.residual["in_lres"](
            x_in_lres,  # (batch, multistep, ensemble, grid, features)
            grid_shard_shapes=None,
            model_comm_group=model_comm_group,
        )[
            :, :, None, :, :
        ]  # Add ensemble back: (batch, time, ensemble=1, grid, features)

        # Apply preprocessing (dataset-specific normalization)
        x_in_lres_upsampled = pre_processors["in_lres"](x_in_lres_upsampled, in_place=False)
        x_in_hres = pre_processors["in_hres"](x_in_hres, in_place=False)

        # Build grid_shard_shapes dict for model (all None for single-device inference)
        grid_shard_shapes = {
            "in_lres": None,
            "in_hres": None,
            "out_hres": None,
        }

        return (x_in_lres_upsampled, x_in_hres), grid_shard_shapes

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
            Input batched data dictionary (before pre-processing) with keys "in_lres", "in_hres"
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

            # Before sampling hook - will upsample in_lres and preprocess both inputs
            before_sampling_data, grid_shard_shapes = self._before_sampling(
                batch,
                pre_processors,
                multi_step,
                model_comm_group,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies,
                **kwargs,
            )

            x_in_lres_upsampled = before_sampling_data[0]
            x_in_hres = before_sampling_data[1]

            out = self.sample(
                x_in_lres_upsampled,
                x_in_hres,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                noise_scheduler_params=noise_scheduler_params,
                sampler_params=sampler_params,
                **kwargs,
            ).to(x_in_lres_upsampled.dtype)

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
        x_in_lres_upsampled: torch.Tensor,
        x_in_hres: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict] = None,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Parameters
        ----------
        x_in_lres_upsampled : torch.Tensor
            Low-res input (upsampled), shape (batch, time, ensemble, grid, vars)
        x_in_hres : torch.Tensor
            High-res forcings, shape (batch, time, ensemble, grid, vars)
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : Optional[dict]
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
        sigmas = scheduler.get_schedule(x_in_lres_upsampled.device, torch.float64)

        # Initialize output with noise
        print(f"DEBUG: x_in_lres_upsampled.shape = {x_in_lres_upsampled.shape}")
        batch_size = x_in_lres_upsampled.shape[0]
        ensemble_size = x_in_lres_upsampled.shape[2]
        grid_size = x_in_lres_upsampled.shape[-2]
        time_size = 1

        print(
            f"DEBUG: batch_size={batch_size}, time_size={time_size}, ensemble_size={ensemble_size}, grid_size={grid_size}, num_output_channels={self.num_output_channels}"
        )

        shape = (
            batch_size,
            time_size,
            ensemble_size,
            grid_size,
            self.num_output_channels,
        )
        print(f"DEBUG: shape = {shape}")
        print(f"DEBUG: shape types = {[type(s) for s in shape]}")

        y_init = torch.randn(shape, device=x_in_lres_upsampled.device, dtype=sigmas.dtype) * sigmas[0]

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
            x_in_lres_upsampled,
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
            Not used for downscaling but kept for compatibility
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
            dataset="input_in_lres",
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
        grid_shard_shapes: Optional[dict] = None,
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
                apply_shard_shapes(out, -2, grid_shard_shapes["out_hres"]),
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
