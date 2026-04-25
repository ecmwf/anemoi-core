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

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.diffusion_encoder_processor_decoder import (
    AnemoiDiffusionModelEncProcDec,
    AnemoiDiffusionTendModelEncProcDec,
)
from anemoi.models.samplers import diffusion_samplers
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def _match_tensor_channels(input_name_to_index: dict, output_name_to_index: dict) -> torch.Tensor:
    common_channels = set(input_name_to_index.keys()) & set(output_name_to_index.keys())
    channel_mapping = []
    for channel_name in output_name_to_index.keys():
        if channel_name in common_channels:
            channel_mapping.append(input_name_to_index[channel_name])
    return torch.tensor(channel_mapping)


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

    @torch.compile()
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

    @torch.compile()
    def _interpolate_to_high_res(self, x, grid_shard_sizes=None, model_comm_group=None):

        if grid_shard_sizes is not None:
            shard_sizes = grid_shard_sizes
            channel_sizes = get_shard_sizes(x, -1, model_comm_group)
            # grid-sharded input: reshard to channel-shards to apply truncation
            x = all_to_all_transpose(x, dim_split=-1, split_sizes=channel_sizes, dim_concat=0, concat_sizes=shard_sizes, mgroup=model_comm_group)

        # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
        # hence we check that they are on the correct device ; copy should only happen in the first forward run
        if self.A_down is not None:

            self.A_down = self.A_down.to(x.device)
            x = self._truncate_fields(x, self.A_down)  # back to high resolution
        else:
            raise ValueError("A_up not defined at model level.")

        if grid_shard_sizes is not None:
            # back to grid-sharding as before
            x = all_to_all_transpose(x, dim_split=0, split_sizes=shard_sizes, dim_concat=-1, concat_sizes=channel_sizes, mgroup=model_comm_group)

        return x

    def apply_interpolate_to_high_res(
        self, x: torch.Tensor, grid_shard_sizes: list, model_comm_group: ProcessGroup
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

        x_trunc = self._interpolate_to_high_res(x_trunc, grid_shard_sizes, model_comm_group)
        return einops.rearrange(x_trunc, "(bs ens) latlon nvar -> bs ens latlon nvar", bs=bs, ens=ens)

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_lres_channels = len(data_indices.model.input[0])
        self.num_input_hres_channels = len(data_indices.model.input[1])
        self.num_output_channels = len(data_indices.model.output)
        self._internal_input_lres_idx = data_indices.model.input[0].prognostic
        self._internal_input_hres_idx = data_indices.model.input[1].prognostic
        self._internal_output_idx = data_indices.model.output.prognostic

    def _calculate_input_dim(self, dataset_name: str) -> int:
        return (
            self.num_input_lres_channels
            + self.num_input_hres_channels
            + self.num_output_channels
            + self.node_attributes[dataset_name].attr_ndims[self._graph_name_data]
        )  # input_lres + input_hres + noised targets + nodes_attributes

    def _assemble_input(
        self,
        x_in_lres_interp_hres,
        x_in_hres,
        y_noised,
        bse,
        grid_shard_sizes=None,
        model_comm_group=None,
    ):
        dataset_name = next(iter(self._graph_data.keys()))
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=bse)
        if grid_shard_sizes is not None:
            shard_sizes_nodes = grid_shard_sizes
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_sizes_nodes, model_comm_group)

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
        shard_sizes_data = grid_shard_sizes if grid_shard_sizes is not None else get_shard_sizes(x_data_latent, 0, model_comm_group)

        return x_data_latent, None, shard_sizes_data

    def _assert_matching_indices(self, data_indices: dict) -> None:
        pass

    def forward(
        self,
        x_in_lres_interp_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        c_noise: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, ensemble_size = (
            x_in_lres_interp_hres.shape[0],
            x_in_lres_interp_hres.shape[2],
        )
        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = grid_shard_sizes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # prepare noise conditionings
        c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(c_noise)
        shape_c_data = get_shard_sizes(c_data, 0, model_comm_group)
        shape_c_hidden = get_shard_sizes(c_hidden, 0, model_comm_group)

        c_data = shard_tensor(c_data, 0, shape_c_data, model_comm_group)
        c_hidden = shard_tensor(c_hidden, 0, shape_c_hidden, model_comm_group)

        fwd_mapper_kwargs = {"cond": (c_data, c_hidden)}
        processor_kwargs = {"cond": c_hidden}
        bwd_mapper_kwargs = {"cond": (c_hidden, c_data)}

        dataset_name = next(iter(self._graph_data.keys()))

        x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
            x_in_lres_interp_hres,
            x_in_hres,
            y_noised,
            bse,
            grid_shard_sizes,
            model_comm_group,
        )
        x_hidden_latent = self.node_attributes[dataset_name](self._graph_name_hidden, batch_size=batch_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        # Encoder
        encoder_edge_attr, encoder_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[
            dataset_name
        ].get_edges(batch_size=bse, model_comm_group=model_comm_group)

        enc_shard_info = BipartiteGraphShardInfo(
            src_nodes=shard_sizes_data if in_out_sharded else None,
            dst_nodes=shard_sizes_hidden,
            edges=enc_edge_shard_sizes,
        )

        x_data_latent, x_latent = self.encoder[dataset_name](
            (x_data_latent, x_hidden_latent),
            batch_size=bse,
            shard_info=enc_shard_info,
            edge_attr=encoder_edge_attr,
            edge_index=encoder_edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=True,
            **fwd_mapper_kwargs,
        )

        # Processor
        processor_edge_attr, processor_edge_index, proc_edge_shard_sizes = self.processor_graph_provider.get_edges(
            batch_size=bse, model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=bse,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )

        x_latent_proc = x_latent_proc + x_latent

        # Decoder
        decoder_edge_attr, decoder_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[
            dataset_name
        ].get_edges(batch_size=bse, model_comm_group=model_comm_group)

        dec_shard_info = BipartiteGraphShardInfo(
            src_nodes=shard_sizes_hidden,
            dst_nodes=shard_sizes_data if in_out_sharded else None,
            edges=dec_edge_shard_sizes,
        )

        x_out = self.decoder[dataset_name](
            (x_latent_proc, x_data_latent),
            batch_size=bse,
            shard_info=dec_shard_info,
            edge_attr=decoder_edge_attr,
            edge_index=decoder_edge_index,
            model_comm_group=model_comm_group,
            keep_x_dst_sharded=in_out_sharded,
            **bwd_mapper_kwargs,
        )

        x_out = self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x_in_lres_interp_hres.dtype)
        # print("time in model forward step", time.time() - start_init)
        return x_out

    def fwd_with_preconditioning(
        self,
        x_in_lres_interp_hres: torch.Tensor,
        x_in_hres: torch.Tensor,
        y_noised: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-conditioning of EDM diffusion model."""
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(sigma, self.sigma_data)
        pred = self(
            x_in_lres_interp_hres,
            x_in_hres,
            (c_in * y_noised),
            c_noise,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
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

        lres_grid_shard_sizes = None
        if model_comm_group is not None:
            lres_shard_sizes = get_shard_sizes(x_in, -2, model_comm_group)
            x_in = shard_tensor(x_in, -2, lres_shard_sizes, model_comm_group)
            lres_grid_shard_sizes = lres_shard_sizes
        hres_grid_shard_sizes = None
        if model_comm_group is not None:
            hres_shard_sizes = get_shard_sizes(x_in_hres, -2, model_comm_group)
            x_in_hres = shard_tensor(x_in_hres, -2, hres_shard_sizes, model_comm_group)
            hres_grid_shard_sizes = hres_shard_sizes

        x_in_interp_to_hres_raw = self.apply_interpolate_to_high_res(
            x_in[:, 0, ...],
            grid_shard_sizes=lres_grid_shard_sizes,
            model_comm_group=model_comm_group,
        )[:, None, ...]

        for i in range(min(6, x_in_interp_to_hres_raw.shape[-1])):
            LOGGER.info(
                "Interpolated tensor statistics for index %d: mean=%f, std=%f",
                i,
                x_in_interp_to_hres_raw[..., i].mean().item(),
                x_in_interp_to_hres_raw[..., i].std().item(),
            )

        x_in_interp_to_hres = x_in_interp_to_hres_raw
        x_in_interp_to_hres = pre_processors(x_in_interp_to_hres, dataset="input_lres", in_place=False)
        x_in_hres = pre_processors(x_in_hres, dataset="input_hres", in_place=False)

        return (x_in_interp_to_hres, x_in_hres, x_in_interp_to_hres_raw), hres_grid_shard_sizes

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

        LOGGER.info("Starting predict_step in downscaling model.", kwargs)

        num_steps = kwargs.get("num_steps", kwargs.get("nsteps", 80))

        noise_scheduler_params = dict(noise_scheduler_params or {})
        noise_scheduler_params.update(
            {
                "schedule_type": kwargs.get(
                    "schedule_type",
                    noise_scheduler_params.get("schedule_type", "karras"),
                ),
                "sigma_max": kwargs.get(
                    "sigma_max",
                    noise_scheduler_params.get("sigma_max", 100000),
                ),
                "sigma_min": kwargs.get(
                    "sigma_min",
                    noise_scheduler_params.get("sigma_min", 0.03),
                ),
                "rho": kwargs.get(
                    "rho",
                    noise_scheduler_params.get("rho", 7.0),
                ),
                "num_steps": num_steps,
            }
        )
        for piecewise_key in (
            "sigma_transition",
            "high_schedule_type",
            "low_schedule_type",
            "num_steps_high",
            "num_steps_low",
        ):
            if piecewise_key in kwargs:
                noise_scheduler_params[piecewise_key] = kwargs[piecewise_key]

        sampler_params = dict(sampler_params or {})
        sampler_params.update(
            {
                "sampler": kwargs.get(
                    "sampler",
                    sampler_params.get("sampler", "heun"),
                ),
                "S_churn": kwargs.get(
                    "S_churn",
                    sampler_params.get("S_churn", 2.5),
                ),
                "S_min": kwargs.get(
                    "S_min",
                    sampler_params.get("S_min", 0.75),
                ),
                "S_max": kwargs.get(
                    "S_max",
                    sampler_params.get("S_max", 100000),
                ),
                "S_noise": kwargs.get(
                    "S_noise",
                    sampler_params.get("S_noise", 1.05),
                ),
            }
        )

        LOGGER.info("noise_scheduler_params: %s", noise_scheduler_params)
        LOGGER.info("sampler_params: %s", sampler_params)

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
            before_sampling_data, grid_shard_sizes = self._before_sampling(
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
                grid_shard_sizes=grid_shard_sizes,
                noise_scheduler_params=noise_scheduler_params,
                sampler_params=sampler_params,
                **kwargs,
            )

            # After sampling hook
            out = self._after_sampling(
                out,
                post_processors,
                before_sampling_data,
                model_comm_group,
                grid_shard_sizes,
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
        grid_shard_sizes: Optional[list] = None,
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
        grid_shard_sizes : Optional[list]
            Grid shard sizes for distributed processing
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
        LOGGER.info(f"noise_scheduler_config: {noise_scheduler_config}")

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
        step_callback = kwargs.get("step_callback")
        capture_init_state = bool(kwargs.get("capture_init_state", False))
        seed = kwargs.get("seed")
        noise_generator = None
        if seed is not None:
            noise_generator = torch.Generator(device=x_in_interp_to_hres.device.type)
            noise_generator.manual_seed(int(seed))
        print("sample, sigmas", sigmas)
        y_init = torch.randn(
            shape,
            device=x_in_interp_to_hres.device,
            generator=noise_generator,
        ) * sigmas[0]
        print("sample, y val", y_init.mean(dim=-1).std())
        if step_callback is not None and capture_init_state:
            step_callback(-1, y_init)
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
        print("sample y_init", y_init.mean(dim=-1).std())
        return sampler_instance.sample(
            x_in_interp_to_hres,
            x_in_hres,
            y_init,
            sigmas,
            self.fwd_with_preconditioning,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            **kwargs,
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
            The raw interpolated input state tensor with full input variables.
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
        denorm_residuals = post_processors_state(residuals, dataset="output", in_place=False)

        for i in range(min(6, denorm_residuals.shape[-1])):
            LOGGER.info(
                "Residual tensor statistics for index %d: mean=%f, std=%f",
                i,
                denorm_residuals[..., i].mean().item(),
                denorm_residuals[..., i].std().item(),
            )

        if not hasattr(self, "_output_input_matching_indices"):
            output_name_to_index = {
                key: value
                for key, value in self.data_indices.data.output.name_to_index.items()
                if value in self.data_indices.data.output.full
            }
            self._output_input_matching_indices = _match_tensor_channels(
                self.data_indices.data.input[0].name_to_index,
                output_name_to_index,
            )

        matching_channel_indices = self._output_input_matching_indices.to(state_inp.device)
        denorm_state_inp = state_inp[..., matching_channel_indices]
        for i in range(min(6, denorm_state_inp.shape[-1])):
            LOGGER.info(
                "Input tensor statistics for index %d: mean=%f, std=%f",
                i,
                denorm_state_inp[..., i].mean().item(),
                denorm_state_inp[..., i].std().item(),
            )

        denorm_output = denorm_residuals + denorm_state_inp
        for i in range(min(6, denorm_output.shape[-1])):
            LOGGER.info(
                "Output tensor statistics for index %d: mean=%f, std=%f",
                i,
                denorm_output[..., i].mean().item(),
                denorm_output[..., i].std().item(),
            )
        return denorm_output

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
        grid_shard_sizes: Optional[list] = None,
        gather_out: bool = True,
        post_processors_tendencies: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process sampled tendency to get state prediction.

        Override to convert tendency to state using x_t0.
        """
        if isinstance(before_sampling_data, tuple) and len(before_sampling_data) >= 3:
            x_in_interp_raw = before_sampling_data[2]
        else:
            raise ValueError("Expected before_sampling_data to contain x_in_interp")

        # Convert tendency to state
        out = self.add_interp_to_state(
            x_in_interp_raw,
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
                grid_shard_sizes,
                model_comm_group,
            )

        return out
