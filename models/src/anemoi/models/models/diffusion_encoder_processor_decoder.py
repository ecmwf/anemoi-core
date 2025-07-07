# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Callable
from typing import Optional

import einops
import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.diffusion import SinusoidalEmbeddings
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.samplers import diffusion_samplers
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiDiffusionModelEncProcDec(AnemoiModelEncProcDec):
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

        # model_config can be either a dict (when instantiated directly, tests?) or DotDict (from Hydra instantiation)
        model_config_local = DotDict(model_config) if isinstance(model_config, dict) else model_config

        diffusion_config = model_config_local.model.model.diffusion
        self.noise_channels = diffusion_config.noise.noise_channels
        self.noise_cond_dim = diffusion_config.noise.noise_cond_dim
        self.sigma_max = diffusion_config.noise.sigma_max
        self.sigma_min = diffusion_config.noise.sigma_min
        self.sigma_data = diffusion_config.noise.sigma_data

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )

        self.get_noise_schedule = diffusion_samplers.get_noise_schedule

        self.noise_embedder = SinusoidalEmbeddings(num_channels=self.noise_channels, max_period=1000)

        self.noise_cond_mlp = nn.Sequential()
        self.noise_cond_mlp.add_module("linear1_no_gradscaling", nn.Linear(self.noise_channels, self.noise_channels))
        self.noise_cond_mlp.add_module("activation", nn.SiLU())
        self.noise_cond_mlp.add_module("linear2_no_gradscaling", nn.Linear(self.noise_channels, self.noise_cond_dim))

    def _calculate_input_dim(self, model_config):
        base_input_dim = super()._calculate_input_dim(model_config)
        return base_input_dim + self.num_output_channels + self.noise_cond_dim

    def _calculate_input_dim_latent(self, model_config):
        base_input_dim = super()._calculate_input_dim_latent(model_config)
        return base_input_dim + self.noise_cond_dim

    def _assemble_input(self, x, y_noised, c_data, c_hidden, bse, grid_shard_shapes=None, model_comm_group=None):
        # Get node attributes
        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=bse)

        # Shard node attributes if grid sharding is enabled
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_data, 0, grid_shard_shapes, model_comm_group)
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # combine noised target, input state, noise conditioning and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(y_noised, "batch ensemble grid vars -> (batch ensemble grid) vars"),
                c_data,
                node_attributes_data,
                # einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars"),
            ),
            dim=-1,  # feature dimension
        )

        # Get shard shapes for data
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        x_hidden_latent = torch.cat(
            (
                self.node_attributes(self._graph_name_hidden, batch_size=bse),
                c_hidden,
            ),
            dim=-1,
        )
        return x_data_latent, x_hidden_latent, None, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, bse, dtype):
        x_out = einops.rearrange(x_out, "(bse n) f -> bse n f", bse=bse)
        x_out = einops.rearrange(x_out, "(bs e) n f -> bs e n f", bs=batch_size).to(dtype=dtype)

        assert x_skip is None, "Residual connection not implemented for flow model"

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

        if edge_conditioning:
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

        assert not (
            in_out_sharded and (grid_shard_shapes is None or model_comm_group is None)
        ), "If input is sharded, grid_shard_shapes and model_comm_group must be provided."

        c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(sigma)

        x_data_latent, x_hidden_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, y_noised, c_data, c_hidden, bse, grid_shard_shapes, model_comm_group
        )

        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        # Shard conditioning tensors based on data shard shapes if grid sharding is enabled
        if grid_shard_shapes is not None:
            shape_c_data = shard_shapes_data
            shape_c_hidden = shard_shapes_hidden
        else:
            shape_c_data = get_shard_shapes(c_data, 0, model_comm_group)
            shape_c_hidden = get_shard_shapes(c_hidden, 0, model_comm_group)

        c_data = shard_tensor(c_data, 0, shape_c_data, model_comm_group)
        c_hidden = shard_tensor(c_hidden, 0, shape_c_hidden, model_comm_group)

        fwd_mapper_kwargs = {"cond": (c_data, c_hidden)}
        processor_kwargs = {"cond": c_hidden}
        bwd_mapper_kwargs = {"cond": (c_hidden, c_data)}

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

        x_out = self._assemble_output(x_out, x_skip, batch_size, bse, x.dtype)

        return x_out

    def fwd_with_preconditioning(
        self,
        x: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
    ) -> torch.Tensor:
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

    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
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
        **kwargs
            Sampling parameters (sampler, num_steps, schedule_type, sigma_max, sigma_min, rho, etc.)

        Returns
        -------
        torch.Tensor
            Sampled output (after post-processing)
        """
        batch = pre_processors(batch, in_place=False)

        with torch.no_grad():
            assert (
                len(batch.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"

            # Dimensions are batch, timesteps, grid, variables
            x = batch[:, 0:multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)

            sigma_max = kwargs.get("sigma_max", self.sigma_max)
            sigma_min = kwargs.get("sigma_min", self.sigma_min)
            rho = kwargs.get("rho", 7.0)
            num_steps = kwargs.get("num_steps", 20)
            sampler = kwargs.get("sampler", "heun")
            schedule_type = kwargs.get("schedule_type", "karras")

            out = self.sample(
                x,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                rho=rho,
                num_steps=num_steps,
                schedule_type=schedule_type,
                sampler=sampler,
                **kwargs,
            ).to(x.dtype)

            # Apply post-processing
            out = post_processors(out, in_place=False)

            # Gather output if needed
            if gather_out and model_comm_group is not None:
                out = gather_tensor(out, -2, apply_shard_shapes(out, -2, grid_shard_shapes), model_comm_group)

        return out

    def sample(
        self,
        x: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        sampler: str = "heun",
        num_steps: int = 50,
        schedule_type: str = "karras",
        rho: float = 7.0,
        sigma_max: float = None,
        sigma_min: float = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from the diffusion model.

        Parameters
        ----------
        x : torch.Tensor
            Input conditioning data with shape (batch, time, ensemble, grid, vars)
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        sampler : str
            Sampling method: "heun" or "dpmpp_2m"
        num_steps : int
            Number of sampling steps
        schedule_type : str
            Type of noise schedule
        rho : float
            Time discretization parameter
        sigma_max : float
            Maximum noise level
        sigma_min : float
            Minimum noise level
        **kwargs
            Additional sampler-specific arguments

        Returns
        -------
        torch.Tensor
            Sampled output with shape (batch, ensemble, grid, vars)
        """

        # Generate noise schedule
        assert sigma_max is not None and sigma_min is not None, "sigma_max and sigma_min are required."
        sigmas = self.get_noise_schedule(
            num_steps=num_steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            schedule_type=schedule_type,
            rho=rho,
            device=x.device,
            dtype_compute=torch.float64,
        )

        # Initialize output with noise
        batch_size, ensemble_size, grid_size = x.shape[0], x.shape[2], x.shape[-2]
        shape = (batch_size, ensemble_size, grid_size, self.num_output_channels)
        y_init = torch.randn(shape, device=x.device, dtype=sigmas.dtype) * sigmas[0]

        if sampler == "heun":
            return diffusion_samplers.edm_heun_sampler(
                x,
                y_init,
                sigmas,
                self.fwd_with_preconditioning,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                S_churn=kwargs.get("S_churn", 0.0),
                S_min=kwargs.get("S_min", 0.0),
                S_max=kwargs.get("S_max", float("inf")),
                S_noise=kwargs.get("S_noise", 1.0),
                dtype=sigmas.dtype,
            )
        elif sampler == "dpmpp_2m":
            return diffusion_samplers.dpmpp_2m_sampler(
                x,
                y_init.to(x.dtype),
                sigmas.to(x.dtype),
                self.fwd_with_preconditioning,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}")


class AnemoiDiffusionTendModelEncProcDec(AnemoiDiffusionModelEncProcDec):
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

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            truncation_data=truncation_data,
        )

    def _calculate_input_dim(self, model_config):
        input_dim = self.multi_step * self.num_input_channels + self.node_attributes.attr_ndims[self._graph_name_data]
        input_dim += self.num_output_channels  # noised targets
        input_dim += len(self.data_indices.model.input.prognostic)  # truncated input state
        input_dim += self.noise_cond_dim
        return input_dim

    def _assemble_input(self, x, y_noised, c_data, c_hidden, bse, grid_shard_shapes=None, model_comm_group=None):
        x_trunc = x[:, -1, :, :, self._internal_input_idx]
        x_trunc = einops.rearrange(x_trunc, "batch ensemble grid vars -> (batch ensemble) grid vars")
        x_trunc = self._apply_truncation(x_trunc, grid_shard_shapes, model_comm_group)

        # Get node attributes
        node_attributes_data = self.node_attributes(self._graph_name_data, batch_size=bse)

        # Shard node attributes if grid sharding is enabled
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_data, 0, grid_shard_shapes, model_comm_group)
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # combine noised target, input state, noise conditioning and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(y_noised, "batch ensemble grid vars -> (batch ensemble grid) vars"),
                c_data,  # SL TODO: remove this ... compare to what was in old model
                node_attributes_data,
                einops.rearrange(x_trunc, "bse grid vars -> (bse grid) vars"),
            ),
            dim=-1,  # feature dimension
        )

        # Get shard shapes for data
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        x_hidden_latent = torch.cat(
            (
                self.node_attributes(self._graph_name_hidden, batch_size=bse),
                c_hidden,  # SL TODO: remove this ... compare to what was in old model
            ),
            dim=-1,
        )
        return x_data_latent, x_hidden_latent, None, shard_shapes_data

    def add_tendency_to_state(
        self,
        state_inp: torch.Tensor,
        tendency: torch.Tensor,
        post_processors_state: Callable,
        post_processors_tendencies: Callable,
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

        return state_outp

    def predict_step(
        self,
        batch: torch.Tensor,
        pre_processors: nn.Module,
        post_processors: nn.Module,
        pre_processors_tendencies: nn.Module,
        post_processors_tendencies: nn.Module,
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Prediction step for tendency diffusion models.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data (before pre-processing)
        pre_processors : nn.Module
            Pre-processing module for states
        post_processors : nn.Module
            Post-processing module for states
        pre_processors_tendencies : nn.Module
            Pre-processing module for tendencies
        post_processors_tendencies : nn.Module
            Post-processing module for tendencies
        multi_step : int
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        gather_out : bool
            Whether to gather output tensors across distributed processes
        **kwargs
            Sampling parameters

        Returns
        -------
        torch.Tensor
            Predicted state (after post-processing)
        """
        # Apply pre-processing to get normalized state
        batch_normalized = pre_processors(batch, in_place=False)

        with torch.no_grad():
            assert (
                len(batch_normalized.shape) == 4
            ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch_normalized.shape}!"

            # Get reference state, with ensemble dimension
            x_t0 = batch_normalized[:, -1, None, ...]

            # Prepare input for diffusion model with ensemble dimension
            x = batch_normalized[:, 0:multi_step, None, ...]

            # Handle distributed processing
            grid_shard_shapes = None
            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group)
                grid_shard_shapes = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)
                x_t0 = shard_tensor(x_t0, -2, shard_shapes, model_comm_group)

            # Sample tendency, SL TODO: make this configurable for inference
            sigma_max = 90
            sigma_min = 0.03
            rho = 7.0
            num_steps = 20
            sampler = "heun"

            # Sample normalized tendency
            tendency_normalized = self.sample(
                x,
                model_comm_group,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                rho=rho,
                num_steps=num_steps,
                schedule_type="karras",
                sampler=sampler,
                **kwargs,
            ).to(x_t0.dtype)

            # Add tendency to reference state
            state_output = self.add_tendency_to_state(
                x_t0,
                tendency_normalized,
                post_processors,
                post_processors_tendencies,
            )

            # Gather output if needed
            if gather_out and model_comm_group is not None:
                state_output = gather_tensor(
                    state_output, -2, apply_shard_shapes(state_output, -2, grid_shard_shapes), model_comm_group
                )

        return state_output

    def compute_tendency(
        self,
        x_t1: torch.Tensor,
        x_t0: torch.Tensor,
        pre_processors_state: Callable,
        pre_processors_tendencies: Callable,
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

        Returns
        -------
        torch.Tensor
            The normalized tendency tensor output from model.
        """
        tendency = pre_processors_tendencies(
            x_t1[..., self.data_indices.data.output.full] - x_t0[..., self.data_indices.data.output.full],
            in_place=False,
            data_index=self.data_indices.data.output.full,
        )
        # diagnostic variables are taken from x_t1, normalised as full fields:
        tendency[..., self.data_indices.model.output.diagnostic] = pre_processors_state(
            x_t1[..., self.data_indices.data.output.diagnostic],
            in_place=False,
            data_index=self.data_indices.data.output.diagnostic,
        )
        return tendency
