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
from anemoi.models.layers.graph_provider import create_graph_provider
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
        )

        self.noise_embedder = instantiate(diffusion_config.noise_embedder)
        self.noise_cond_mlp = self._create_noise_conditioning_mlp()

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""
        # Encoder data -> hidden
        self.encoder_graph_provider = torch.nn.ModuleDict()
        self.encoder = torch.nn.ModuleDict()
        for dataset_name in self._graph_data.keys():
            # Create graph providers
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[dataset_name][(self._graph_name_data, "to", self._graph_name_hidden)],
                edge_attributes=model_config.model.encoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
                dst_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_hidden],
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )

            self.encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.input_dim[dataset_name],
                in_channels_dst=self.node_attributes[dataset_name].attr_ndims[self._graph_name_hidden],
                hidden_dim=self.num_channels,
                edge_dim=self.encoder_graph_provider[dataset_name].edge_dim,
            )

        # Processor hidden -> hidden (shared across all datasets)
        first_dataset_name = next(iter(self._graph_data.keys()))
        processor_graph = self._graph_data[first_dataset_name][(self._graph_name_hidden, "to", self._graph_name_hidden)]
        processor_grid_size = self.node_attributes[first_dataset_name].num_nodes[self._graph_name_hidden]

        # Processor hidden -> hidden
        self.processor_graph_provider = create_graph_provider(
            graph=processor_graph,
            edge_attributes=model_config.model.processor.get("sub_graph_edge_attributes"),
            src_size=processor_grid_size,
            dst_size=processor_grid_size,
            trainable_size=model_config.model.processor.get("trainable_size", 0),
        )

        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.num_channels,
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        # Decoder hidden -> data
        self.decoder_graph_provider = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()
        for dataset_name in self._graph_data.keys():
            self.decoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[dataset_name][(self._graph_name_hidden, "to", self._graph_name_data)],
                edge_attributes=model_config.model.decoder.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_hidden],
                dst_size=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
                trainable_size=model_config.model.decoder.get("trainable_size", 0),
            )
            self.decoder[dataset_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.num_channels,
                in_channels_dst=self.input_dim[dataset_name],
                hidden_dim=self.num_channels,
                out_channels_dst=self.num_output_channels[dataset_name],
                edge_dim=self.decoder_graph_provider[dataset_name].edge_dim,
            )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        input_dim = super()._calculate_input_dim(dataset_name)
        input_dim += self.num_output_channels[dataset_name]  # input + noised targets
        return input_dim

    def _create_noise_conditioning_mlp(self) -> nn.Sequential:
        mlp = nn.Sequential()
        mlp.add_module("linear1_no_gradscaling", nn.Linear(self.noise_channels, self.noise_channels))
        mlp.add_module("activation", nn.SiLU())
        mlp.add_module("linear2_no_gradscaling", nn.Linear(self.noise_channels, self.noise_cond_dim))
        return mlp

    def _assemble_input(self, x, y_noised, bse, grid_shard_shapes=None, model_comm_group=None, dataset_name=None):
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=bse)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

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

    def _generate_noise_conditioning(
        self, sigma: torch.Tensor, dataset_name: str, edge_conditioning: bool = False
    ) -> torch.Tensor:
        noise_cond = self.noise_embedder(sigma)
        noise_cond = self.noise_cond_mlp(noise_cond)

        c_data = self._make_noise_emb(
            noise_cond,
            repeat=self.node_attributes[dataset_name].num_nodes[self._graph_name_data],
        )
        c_hidden = self._make_noise_emb(
            noise_cond, repeat=self.node_attributes[dataset_name].num_nodes[self._graph_name_hidden]
        )

        if edge_conditioning:  # this is currently not used but could be useful for edge conditioning of GNN
            c_data_to_hidden = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[dataset_name][(self._graph_name_data, "to", self._graph_name_hidden)][
                    "edge_length"
                ].shape[0],
            )
            c_hidden_to_data = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[dataset_name][(self._graph_name_hidden, "to", self._graph_name_data)][
                    "edge_length"
                ].shape[0],
            )
            c_hidden_to_hidden = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[dataset_name][(self._graph_name_hidden, "to", self._graph_name_hidden)][
                    "edge_length"
                ].shape[0],
            )
        else:
            c_data_to_hidden = None
            c_hidden_to_data = None
            c_hidden_to_hidden = None

        return c_data, c_hidden, c_data_to_hidden, c_hidden_to_data, c_hidden_to_hidden

    def forward(
        self,
        x: dict[str, torch.Tensor],
        y_noised: dict[str, torch.Tensor],
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[dict[str, list]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Multi-dataset case
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # prepare noise conditionings
        c_data, c_hidden, shape_c_data, shape_c_hidden = {}, {}, {}, {}
        fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = {}, {}, {}
        for dataset_name in x:
            c_data[dataset_name], c_hidden[dataset_name], _, _, _ = self._generate_noise_conditioning(
                sigma[dataset_name], dataset_name=dataset_name, edge_conditioning=False
            )
            shape_c_data[dataset_name] = get_shard_shapes(c_data[dataset_name], 0, model_comm_group=model_comm_group)
            shape_c_hidden[dataset_name] = get_shard_shapes(
                c_hidden[dataset_name], 0, model_comm_group=model_comm_group
            )

            c_data[dataset_name] = shard_tensor(c_data[dataset_name], 0, shape_c_data[dataset_name], model_comm_group)
            c_hidden[dataset_name] = shard_tensor(
                c_hidden[dataset_name], 0, shape_c_hidden[dataset_name], model_comm_group
            )

            fwd_mapper_kwargs[dataset_name] = {"cond": (c_data[dataset_name], c_hidden[dataset_name])}
            processor_kwargs[dataset_name] = {"cond": c_hidden[dataset_name]}
            bwd_mapper_kwargs[dataset_name] = {"cond": (c_hidden[dataset_name], c_data[dataset_name])}

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}
        shard_shapes_hidden_dict = {}
        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
                x[dataset_name], y_noised[dataset_name], bse, grid_shard_shapes, model_comm_group, dataset_name
            )
            x_skip_dict[dataset_name] = x_skip
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            x_hidden_latent = self.node_attributes[dataset_name](self._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden_dict[dataset_name] = get_shard_shapes(
                x_hidden_latent, 0, model_comm_group=model_comm_group
            )

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=bse,
                model_comm_group=model_comm_group,
            )

            x_data_latent, dataset_latents[dataset_name] = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latent),
                batch_size=bse,
                shard_shapes=(shard_shapes_data_dict[dataset_name], shard_shapes_hidden_dict[dataset_name]),
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                edge_shard_shapes=enc_edge_shard_shapes,
                **fwd_mapper_kwargs[dataset_name],
            )
            x_data_latent_dict[dataset_name] = x_data_latent

        x_latent = sum(dataset_latents.values())

        # Processor
        shard_shapes_hidden = shard_shapes_hidden_dict[dataset_names[0]]
        assert all(
            shard_shape == shard_shapes_hidden for shard_shape in shard_shapes_hidden_dict.values()
        ), "All datasets must have the same shard shapes for the hidden graph."
        proc_kwargs = processor_kwargs[dataset_names[0]]
        assert all(
            proc_kwargs == processor_kwargs[dataset_name] for dataset_name in dataset_names
        ), "All datasets must have the same processor kwargs."

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
            **proc_kwargs,
        )

        # Processor skip connection
        x_latent_proc = x_latent_proc + x_latent

        # Decoder
        x_out_dict = {}
        for dataset_name in dataset_names:
            # Compute decoder edges using updated latent representation
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=bse,
                model_comm_group=model_comm_group,
            )

            x_out = self.decoder[dataset_name](
                (x_latent_proc, x_data_latent_dict[dataset_name]),
                batch_size=bse,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data_dict[dataset_name]),
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,  # x_latent always comes sharded
                x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
                edge_shard_shapes=dec_edge_shard_shapes,
                **bwd_mapper_kwargs[dataset_name],
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out, x_skip_dict[dataset_name], batch_size, ensemble_size, x_out.dtype
            )

        return x_out_dict

    def fwd_with_preconditioning(
        self,
        x: dict[str, torch.Tensor],
        y_noised: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with pre-conditioning of EDM diffusion model."""
        c_skip, c_out, c_in, c_noise = self._get_preconditioning(sigma, self.sigma_data)
        pred = self(
            x,
            {key: c_in[key] * y_noised[key] for key in y_noised.keys()},
            c_noise,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
        )  # calls forward ...
        D_x = {key: c_skip[key] * y_noised[key] + c_out[key] * pred[key] for key in y_noised.keys()}

        return D_x

    def _get_preconditioning(
        self, sigma: dict[str, torch.Tensor], sigma_data: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute preconditioning factors."""
        c_skip, c_out, c_in, c_noise = {}, {}, {}, {}
        for dataset_name, sigma_i in sigma.items():
            c_skip[dataset_name] = sigma_data**2 / (sigma_i**2 + sigma_data**2)
            c_out[dataset_name] = sigma_i * sigma_data / (sigma_i**2 + sigma_data**2) ** 0.5
            c_in[dataset_name] = 1.0 / (sigma_data**2 + sigma_i**2) ** 0.5
            c_noise[dataset_name] = sigma_i.log() / 4.0

        return c_skip, c_out, c_in, c_noise

    def _before_sampling(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> tuple[tuple[dict[str, torch.Tensor]], dict[str, Optional[list]]]:
        """Prepare batch before sampling.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batch after pre-processing
        pre_processors : dict[str, nn.Module]
            Pre-processing module (already applied)
        multi_step : int
            Number of input timesteps
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        **kwargs
            Additional parameters for subclasses

        Returns
        -------
        tuple[tuple[dict[str, torch.Tensor]], dict[str, Optional[list]]]
            Prepared input tensor(s) and grid shard shapes.
            Can return a single tensor or tuple of tensors for sampling input.
        """
        xs = {}
        grid_shard_shapes = dict.fromkeys(batch, None)

        for dataset_name, x in batch.items():
            # Dimensions are batch, timesteps, grid, variables
            x = x[:, 0:multi_step, None, ...]  # add dummy ensemble dimension as 3rd index

            if model_comm_group is not None:
                shard_shapes = get_shard_shapes(x, -2, model_comm_group=model_comm_group)
                grid_shard_shapes[dataset_name] = [shape[-2] for shape in shard_shapes]
                x = shard_tensor(x, -2, shard_shapes, model_comm_group)
            x = pre_processors[dataset_name](x, in_place=False)

            xs[dataset_name] = x

        return (xs,), grid_shard_shapes

    def _after_sampling(
        self,
        out: dict[str, torch.Tensor],
        post_processors: dict[str, nn.Module],
        before_sampling_data: dict[str, Union[torch.Tensor, tuple[torch.Tensor, ...]]],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Process sampled output.

        Parameters
        ----------
        out : dict[str, torch.Tensor]
            Sampled output tensor
        post_processors : dict[str, nn.Module]
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
        for dataset_name in out.keys():
            out[dataset_name] = post_processors[dataset_name](out[dataset_name], in_place=False)

            if gather_out and model_comm_group is not None:
                out[dataset_name] = gather_tensor(
                    out[dataset_name],
                    -2,
                    apply_shard_shapes(out[dataset_name], -2, shard_shapes_dim=grid_shard_shapes[dataset_name]),
                    model_comm_group,
                )

        return out

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
        pre_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        post_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Prediction step for flow/diffusion models - performs sampling.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batched data (before pre-processing)
        pre_processors : dict[str, nn.Module],
            Pre-processing module
        post_processors : dict[str, nn.Module],
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
        pre_processors_tendencies : Optional[dict[str, nn.Module]]
            Pre-processing module for tendencies (used by subclasses)
        post_processors_tendencies : Optional[dict[str, nn.Module]]
            Post-processing module for tendencies (used by subclasses)
        **kwargs
            Additional sampling parameters

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled output (after post-processing)
        """
        with torch.no_grad():

            assert isinstance(batch, dict), "Input batch must be a dictionary!"
            for dataset_name, dataset_tensor in batch.values():
                assert (
                    len(dataset_tensor.shape) == 4
                ), f'The input tensor "{dataset_name}" has an incorrect shape: expected a 4-dimensional tensor, got {dataset_tensor.shape}!'

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
            x_dtype = next(iter(batch.values())).dtype

            out = self.sample(
                x,
                model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                noise_scheduler_params=noise_scheduler_params,
                sampler_params=sampler_params,
                **kwargs,
            ).to(x_dtype)

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
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        noise_scheduler_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Sample from the diffusion model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input conditioning data with shape (batch, time, ensemble, grid, vars)
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training
        grid_shard_shapes : dict[str, Optional[list]]
            Grid shard shapes for distributed processing
        noise_scheduler_params : Optional[dict]
            Dictionary of noise scheduler parameters (schedule_type, num_steps, sigma_max, etc.) to override defaults
        sampler_params : Optional[dict]
            Dictionary of sampler parameters (sampler, S_churn, S_min, etc.) to override defaults
        **kwargs
            Additional sampler-specific arguments

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled output with shape (batch, ensemble, grid, vars)
        """
        x_device = next(iter(x.values())).device

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

        sigmas, y_init = {}, {}
        for dataset_name, x_data in x.items():
            sigmas[dataset_name] = scheduler.get_schedule(x_device, torch.float64)

            # Initialize output with noise
            batch_size, ensemble_size, grid_size = x_data.shape[0], x_data.shape[2], x_data.shape[-2]
            shape = (batch_size, ensemble_size, grid_size, self.num_output_channels)
            y_init[dataset_name] = torch.randn(shape, device=x_data.device, dtype=sigmas.dtype) * sigmas[0]

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
        sampler_instance = sampler_cls(dtype=next(iter(sigmas.values())).dtype, **diffusion_sampler_config)

        # The same scheduler has to be used for all datasets
        sigmas_ref = next(iter(sigmas.values()))
        for dataset_name, sigmas_i in sigmas.items():
            if not torch.allclose(sigmas_i, sigmas_ref):
                warnings.warn(
                    f"Sigma schedules differ between datasets! Dataset '{dataset_name}' has a different schedule."
                )

        return sampler_instance.sample(
            x,
            y_init,
            sigmas_ref,
            self.fwd_with_preconditioning,
            model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
        )

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


class AnemoiDiffusionTendModelEncProcDec(AnemoiDiffusionModelEncProcDec):
    """Diffusion model for tendency prediction."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        model_config_local = DotDict(model_config)

        self.condition_on_residual = model_config_local.model.condition_on_residual
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        input_dim = super()._calculate_input_dim(dataset_name)
        if self.condition_on_residual:
            input_dim += len(self.data_indices[dataset_name].model.input.prognostic)  # truncated input state
        return input_dim

    def _assemble_input(
        self,
        x: torch.Tensor,
        y_noised: torch.Tensor,
        bse: int,
        grid_shard_shapes: dict | None = None,
        model_comm_group=None,
        dataset_name=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=bse)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

        x_skip = self.residual[dataset_name](x, grid_shard_shapes, model_comm_group)

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
            ),
            dim=-1,  # feature dimension
        )
        if self.condition_on_residual:
            x_data_latent = torch.cat(
                (x_data_latent, einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars")), dim=-1
            )

        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, x_skip, shard_shapes_data

    def compute_tendency(
        self,
        x_t1: dict[str, torch.Tensor],
        x_t0: dict[str, torch.Tensor],
        pre_processors_state: dict[str, Callable],
        pre_processors_tendencies: dict[str, Callable],
        input_post_processor: Optional[Callable] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the tendency from two states.

        Parameters
        ----------
        x_t1 : torch.Tensor
            The state at time t1 with full input variables.
        x_t0 : torch.Tensor
            The state at time t0 with prognostic input variables.
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
        tendencies = {}

        assert set(x_t1.keys()) == set(x_t0.keys()), "x_t1 and x_t0 must have the same dataset keys."

        for dataset_name in x_t1.keys():
            if input_post_processor[dataset_name] is not None:
                x_t1[dataset_name] = input_post_processor[dataset_name](
                    x_t1[dataset_name], in_place=False, data_index=self.data_indices[dataset_name].data.output.full
                )
                x_t0[dataset_name] = input_post_processor[dataset_name](
                    x_t0[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.output.prognostic,
                )

            tendency = x_t1[dataset_name].clone()
            tendency[..., self.data_indices[dataset_name].model.output.prognostic] = pre_processors_tendencies[
                dataset_name
            ](
                x_t1[dataset_name][..., self.data_indices[dataset_name].model.output.prognostic] - x_t0[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.prognostic,
            )
            # diagnostic variables are taken from x_t1, normalised as full fields:
            tendency[..., self.data_indices[dataset_name].model.output.diagnostic] = pre_processors_state[dataset_name](
                x_t1[dataset_name][..., self.data_indices[dataset_name].model.output.diagnostic],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.diagnostic,
            )
            tendencies[dataset_name] = tendency

        return tendencies

    def add_tendency_to_state(
        self,
        state_inp: dict[str, torch.Tensor],
        tendency: dict[str, torch.Tensor],
        post_processors_state: dict[str, Callable],
        post_processors_tendencies: dict[str, Callable],
        output_pre_processor: dict[str, Optional[Callable]] = None,
    ) -> dict[str, torch.Tensor]:
        """Add the tendency to the state.

        Parameters
        ----------
        state_inp : dict[str, torch.Tensor]
            The normalized input state tensor with prognostic input variables.
        tendency : dict[str, torch.Tensor]
            The normalized tendency tensor output from model.
        post_processors_state : dict[str, Callable]
            Function to post-process the state variables.
        post_processors_tendencies : dict[str, Callable]
            Function to post-process the tendency variables.
        output_pre_processor : Optional[Callable], optional
            Function to pre-process the output state. If provided,
            the output state will be pre-processed before returning.
            If None, the output state is returned directly. Default is None.

        Returns
        -------
        dict[str, torch.Tensor]
            the de-normalised state
        """
        state_outp = {}

        for dataset_name in tendency.keys():
            state_outp[dataset_name] = post_processors_tendencies[dataset_name](
                tendency[dataset_name], in_place=False, data_index=self.data_indices[dataset_name].data.output.full
            )

            state_outp[dataset_name][
                ..., self.data_indices[dataset_name].model.output.diagnostic
            ] = post_processors_state[dataset_name](
                tendency[dataset_name][..., self.data_indices[dataset_name].model.output.diagnostic],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.diagnostic,
            )

            state_outp[dataset_name][
                ..., self.data_indices[dataset_name].model.output.prognostic
            ] += post_processors_state[dataset_name](
                state_inp[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.input.prognostic,
            )

            if output_pre_processor[dataset_name] is not None:
                state_outp[dataset_name] = output_pre_processor[dataset_name](
                    state_outp[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.output.full,
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
        self, x: dict[str, torch.Tensor], grid_shard_shapes: dict[str, list], model_comm_group: ProcessGroup
    ) -> dict[str, torch.Tensor]:
        """Apply reference state truncation to the input tensor.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input tensor with shape {dataset_name: (bs, ens, latlon, nvar)}

        Returns
        -------
        dict[str, torch.Tensor]
            Truncated tensor with same shape as input
        """
        x_skips = {}

        for dataset_name, in_x in x.items():
            x_skip = self.residual[dataset_name](in_x, grid_shard_shapes[dataset_name], model_comm_group)
            # x_skip.shape: (bs, ens, latlon, nvar)
            x_skips[dataset_name] = x_skip[..., self.data_indices[dataset_name].model.input.prognostic]

        return x_skips
