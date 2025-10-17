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

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models.base import BaseGraphModel
from anemoi.models.samplers import diffusion_samplers
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiDiffusionModelEncProcDec(BaseGraphModel):
    """Diffusion Model."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        truncation_data: dict,
    ) -> None:

        model_config_local = DotDict(model_config)

        diffusion_config = model_config_local.model.model.diffusion
        self.noise_channels = diffusion_config.noise_channels
        self.noise_cond_dim = diffusion_config.noise_cond_dim
        self.sigma_data = diffusion_config.sigma_data
        self.sigma_max = diffusion_config.sigma_max
        self.sigma_min = diffusion_config.sigma_min
        self.inference_defaults = diffusion_config.inference_defaults

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )

        self.noise_embedder = instantiate(diffusion_config.noise_embedder)
        self.noise_cond_mlp = self._create_noise_conditioning_mlp()

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""

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

    def _calculate_input_dim(self):
        base_input_dim = super()._calculate_input_dim()
        return base_input_dim + self.num_output_channels  # input + noised targets

    def _create_noise_conditioning_mlp(self) -> nn.Sequential:
        mlp = nn.Sequential()
        mlp.add_module("linear1_no_gradscaling", nn.Linear(self.noise_channels, self.noise_channels))
        mlp.add_module("activation", nn.SiLU())
        mlp.add_module("linear2_no_gradscaling", nn.Linear(self.noise_channels, self.noise_cond_dim))
        return mlp

    def _assemble_input(self, x, y_noised, bse, grid_shard_shapes=None, model_comm_group=None):
        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=bse)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # combine noised target, input state, noise conditioning and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(y_noised, "batch ensemble grid vars -> (batch ensemble grid) vars"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, None, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        x_out = einops.rearrange(
            x_out,
            "(batch ensemble grid) vars -> batch ensemble grid vars",
            batch=batch_size,
            ensemble=ensemble_size,
        ).to(dtype=dtype)

        return x_out

    def _make_noise_emb(self, noise_emb: torch.Tensor, repeat: int) -> torch.Tensor:
        out = einops.repeat(
            noise_emb, "batch ensemble noise_level vars -> batch ensemble (repeat noise_level) vars", repeat=repeat
        )
        out = einops.rearrange(out, "batch ensemble grid vars -> (batch ensemble grid) vars")
        return out

    def _generate_noise_conditioning(self, sigma: torch.Tensor, edge_conditioning: bool = False) -> torch.Tensor:
        noise_cond = self.noise_embedder(sigma)
        noise_cond = self.noise_cond_mlp(noise_cond)

        c_data = self._make_noise_emb(
            noise_cond,
            repeat=self.node_attributes.num_nodes[self._graph_name_data],
        )
        c_hidden = self._make_noise_emb(noise_cond, repeat=self.node_attributes.num_nodes[self._graph_name_hidden])

        if edge_conditioning:  # this is currently not used but could be useful for edge conditioning of GNN
            c_data_to_hidden = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)]["edge_length"].shape[0],
            )
            c_hidden_to_data = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)]["edge_length"].shape[0],
            )
            c_hidden_to_hidden = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)]["edge_length"].shape[
                    0
                ],
            )
        else:
            c_data_to_hidden = None
            c_hidden_to_data = None
            c_hidden_to_hidden = None

        return c_data, c_hidden, c_data_to_hidden, c_hidden_to_data, c_hidden_to_hidden

    def forward(
        self,
        x: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:

        batch_size, ensemble_size = x.shape[0], x.shape[2]
        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # prepare noise conditionings
        c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(sigma)
        shape_c_data = get_shard_shapes(c_data, 0, model_comm_group=model_comm_group)
        shape_c_hidden = get_shard_shapes(c_hidden, 0, model_comm_group=model_comm_group)

        c_data = shard_tensor(c_data, 0, shape_c_data, model_comm_group)
        c_hidden = shard_tensor(c_hidden, 0, shape_c_hidden, model_comm_group)

        fwd_mapper_kwargs = {"cond": (c_data, c_hidden)}
        processor_kwargs = {"cond": c_hidden}
        bwd_mapper_kwargs = {"cond": (c_hidden, c_data)}

        x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, y_noised, bse, grid_shard_shapes, model_comm_group
        )
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group=model_comm_group)

        x_data_latent, x_latent = self.encoder(
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

        x_out = self.decoder(
            (x_latent_proc, x_data_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
            **bwd_mapper_kwargs,
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x.dtype)

        return x_out

    def fwd_with_preconditioning(
        self,
        x: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-conditioning of EDM diffusion model."""
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(sigma, self.sigma_data)
        pred = self(
            x, (c_in * y_noised), c_noise, model_comm_group=model_comm_group, grid_shard_shapes=grid_shard_shapes
        )  # calls forward ...
        D_x = c_skip * y_noised + c_out * pred

        return D_x

    def _get_preconditioning(
        self, sigma: torch.Tensor, sigma_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute preconditioning factors."""
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1.0 / (sigma_data**2 + sigma**2) ** 0.5
        c_noise = sigma.log() / 4.0

        return c_skip, c_out, c_in, c_noise

    def _before_sampling(
        self,
        batch: torch.Tensor,
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
        # Dimensions are batch, timesteps, grid, variables
        x = batch[:, 0:multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

        grid_shard_shapes = None
        if model_comm_group is not None:
            shard_shapes = get_shard_shapes(x, -2, model_comm_group=model_comm_group)
            grid_shard_shapes = [shape[-2] for shape in shard_shapes]
            x = shard_tensor(x, -2, shard_shapes, model_comm_group)
        x = pre_processors(x, in_place=False)

        return (x,), grid_shard_shapes

    def _after_sampling(
        self,
        out: torch.Tensor,
        post_processors: nn.Module,
        before_sampling_data: Union[torch.Tensor, tuple[torch.Tensor, ...]],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Process sampled output.

        Parameters
        ----------
        out : torch.Tensor
            Sampled output tensor
        post_processors : nn.Module
            Post-processing module
        before_sampling_data : Union[torch.Tensor, tuple[torch.Tensor, ...]]
            Data returned from _before_sampling (can be used by subclasses)
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : Optional[list]
            Grid shard shapes for gathering
        gather_out : bool
            Whether to gather output
        **kwargs
            Additional parameters for subclasses

        Returns
        -------
        torch.Tensor
            Post-processed output
        """
        out = post_processors(out, in_place=False)

        if gather_out and model_comm_group is not None:
            out = gather_tensor(
                out, -2, apply_shard_shapes(out, -2, shard_shapes_dim=grid_shard_shapes), model_comm_group
            )

        return out

    def predict_step(
        self,
        batch: torch.Tensor,
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
        with torch.no_grad():

            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"

            # Before sampling hook
            before_sampling_data, grid_shard_shapes = self._before_sampling(
                batch,
                pre_processors,
                multi_step,
                model_comm_group,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies,
                **kwargs,
            )

            x = before_sampling_data[0]

            out = self.sample(
                x,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                noise_scheduler_params=noise_scheduler_params,
                sampler_params=sampler_params,
                **kwargs,
            ).to(x.dtype)

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
        x: torch.Tensor,
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

        # Remove schedule_type (used for class selection, not constructor)
        actual_schedule_type = noise_scheduler_config.pop("schedule_type")

        if actual_schedule_type not in diffusion_samplers.NOISE_SCHEDULERS:
            raise ValueError(f"Unknown schedule type: {actual_schedule_type}")

        scheduler_cls = diffusion_samplers.NOISE_SCHEDULERS[actual_schedule_type]
        scheduler = scheduler_cls(**noise_scheduler_config)
        sigmas = scheduler.get_schedule(x.device, torch.float64)

        # Initialize output with noise
        batch_size, ensemble_size, grid_size = x.shape[0], x.shape[2], x.shape[-2]
        shape = (batch_size, ensemble_size, grid_size, self.num_output_channels)
        y_init = torch.randn(shape, device=x.device, dtype=sigmas.dtype) * sigmas[0]

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
            x,
            y_init,
            sigmas,
            self.fwd_with_preconditioning,
            model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
        )


class AnemoiDiffusionTendModelEncProcDec(AnemoiDiffusionModelEncProcDec):
    """Diffusion model for tendency prediction."""

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

    def _calculate_input_dim(self):
        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]
        input_dim += self.num_output_channels  # noised targets
        input_dim += len(self.data_indices.model.input.prognostic)  # truncated input state
        return input_dim

    def _assemble_input(self, x, y_noised, bse, grid_shard_shapes=None, model_comm_group=None):
        x_trunc = x[:, -1, :, :, self._internal_input_idx]
        x_trunc = einops.rearrange(x_trunc, "batch ensemble grid vars -> (batch ensemble) grid vars")
        x_trunc = self.truncation(x_trunc, grid_shard_shapes, model_comm_group)

        # Get node attributes
        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=bse)

        # Shard node attributes if grid sharding is enabled
        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # combine noised target, input state, noise conditioning and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(y_noised, "batch ensemble grid vars -> (batch ensemble grid) vars"),
                node_attributes_data,
                einops.rearrange(x_trunc, "bse grid vars -> (bse grid) vars"),
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, None, shard_shapes_data

    def compute_tendency(
        self,
        x_t1: torch.Tensor,
        x_t0: torch.Tensor,
        pre_processors_state: Callable,
        pre_processors_tendencies: Callable,
        input_post_processor: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Compute the tendency from two states.

        Parameters
        ----------
        x_t1 : torch.Tensor
            The state at time t1 with full input variables.
        x_t0 : torch.Tensor
            The state at time t0 with full input variables.
        pre_processors_state : callable
            Function to pre-process the state variables.
        pre_processors_tendencies : callable
            Function to pre-process the tendency variables.
        input_post_processor : Optional[Callable], optional
            Function to post-process the input state variables. If provided,
            the input states will be post-processed before computing the tendency.
            If None, the input states are used directly. Default is None.

        Returns
        -------
        torch.Tensor
            The normalized tendency tensor output from model.
        """

        if input_post_processor is not None:
            x_t1 = input_post_processor(
                x_t1[..., self.data_indices.data.output.full],
                in_place=False,
                data_index=self.data_indices.data.output.full,
            )
            x_t0 = input_post_processor(
                x_t0[..., self.data_indices.data.output.full],
                in_place=False,
                data_index=self.data_indices.data.output.full,
            )
        else:
            x_t1 = x_t1[..., self.data_indices.data.output.full]
            x_t0 = x_t0[..., self.data_indices.data.output.full]

        tendency = pre_processors_tendencies(
            x_t1 - x_t0,
            in_place=False,
            data_index=self.data_indices.data.output.full,
        )
        # diagnostic variables are taken from x_t1, normalised as full fields:
        tendency[..., self.data_indices.model.output.diagnostic] = pre_processors_state(
            x_t1[..., self.data_indices.model.output.diagnostic],
            in_place=False,
            data_index=self.data_indices.data.output.diagnostic,
        )

        return tendency

    def add_tendency_to_state(
        self,
        state_inp: torch.Tensor,
        tendency: torch.Tensor,
        post_processors_state: Callable,
        post_processors_tendencies: Callable,
        output_pre_processor: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Add the tendency to the state.

        Parameters
        ----------
        state_inp : torch.Tensor
            The normalized input state tensor with full input variables.
        tendency : torch.Tensor
            The normalized tendency tensor output from model.
        post_processors_state : callable
            Function to post-process the state variables.
        post_processors_tendencies : callable
            Function to post-process the tendency variables.
        output_pre_processor : Optional[Callable], optional
            Function to pre-process the output state. If provided,
            the output state will be pre-processed before returning.
            If None, the output state is returned directly. Default is None.

        Returns
        -------
        torch.Tensor
            the de-normalised state
        """
        state_outp = post_processors_tendencies(tendency, in_place=False, data_index=self.data_indices.data.output.full)

        state_outp[..., self.data_indices.model.output.diagnostic] = post_processors_state(
            tendency[..., self.data_indices.model.output.diagnostic],
            in_place=False,
            data_index=self.data_indices.data.output.diagnostic,
        )

        state_outp[..., self.data_indices.model.output.prognostic] += post_processors_state(
            state_inp[..., self.data_indices.model.input.prognostic],
            in_place=False,
            data_index=self.data_indices.data.input.prognostic,
        )

        if output_pre_processor is not None:
            state_outp = output_pre_processor(
                state_outp,
                in_place=False,
                data_index=self.data_indices.data.output.full,
            )

        return state_outp

    def _before_sampling(
        self,
        batch: torch.Tensor,
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
        # Dimensions are batch, timesteps, grid, variables
        x = batch[:, 0:multi_step, None, ...]  # add dummy ensemble dimension as 3rd index
        x_t0 = batch[:, -1, None, ...]  # add dummy ensemble dimension

        grid_shard_shapes = None
        if model_comm_group is not None:
            shard_shapes = get_shard_shapes(x, -2, model_comm_group=model_comm_group)
            grid_shard_shapes = [shape[-2] for shape in shard_shapes]
            x = shard_tensor(x, -2, shard_shapes, model_comm_group)
            shard_shapes = get_shard_shapes(x_t0, -2, model_comm_group=model_comm_group)
            x_t0 = shard_tensor(x_t0, -2, shard_shapes, model_comm_group)

        x = pre_processors(x, in_place=False)
        x_t0 = pre_processors(x_t0, in_place=False)

        return (x, x_t0), grid_shard_shapes

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
        # Extract x_t0 from before_sampling_data
        if isinstance(before_sampling_data, tuple) and len(before_sampling_data) >= 2:
            x_t0 = before_sampling_data[1]
        else:
            raise ValueError("Expected before_sampling_data to contain x_t0")

        # truncate x_t0 if needed
        x_t0 = self.apply_reference_state_truncation(x_t0, grid_shard_shapes, model_comm_group)

        # Convert tendency to state
        out = self.add_tendency_to_state(
            x_t0,
            out,
            post_processors,
            post_processors_tendencies,
        )

        # Gather if needed
        if gather_out and model_comm_group is not None:
            out = gather_tensor(
                out, -2, apply_shard_shapes(out, -2, shard_shapes_dim=grid_shard_shapes), model_comm_group
            )

        return out

    def apply_reference_state_truncation(
        self, x: torch.Tensor, grid_shard_shapes: list, model_comm_group: ProcessGroup
    ) -> torch.Tensor:
        """Apply reference state truncation to the input tensor.

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
        x_trunc = self.truncation(x_trunc, grid_shard_shapes, model_comm_group)
        return einops.rearrange(x_trunc, "(bs ens) latlon nvar -> bs ens latlon nvar", bs=bs, ens=ens)
