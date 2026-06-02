# (C) Copyright 2025 Anemoi contributors.
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
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.transport import EdmSettings
from anemoi.models.transport import NoiseConditioningSettings
from anemoi.models.transport import StochasticInterpolantSettings
from anemoi.models.transport import TransportSourceBuilder
from anemoi.models.transport import TransportSourceRequest
from anemoi.models.transport import get_transport_model_objective
from anemoi.models.transport import reference_state_sampling_source
from anemoi.models.transport import sampling_source_specs
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)

SamplingData = tuple[dict[str, torch.Tensor], ...]


class AnemoiTransportModelEncProcDec(AnemoiModelEncProcDec):
    """Encoder-processor-decoder model conditioned on diffusion noise level or bridge time."""

    def __init__(
        self,
        *,
        model_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        n_step_input: int,
        n_step_output: int,
        graph_data: HeteroData,
    ) -> None:

        model_config = DotDict(model_config)

        transport_params = model_config.model.model.transport
        self.noise_conditioning = NoiseConditioningSettings.from_config(transport_params)
        self.edm = EdmSettings.from_config(transport_params)
        self.stochastic_interpolant = StochasticInterpolantSettings.from_config(transport_params)
        self.transport_source = TransportSourceBuilder.from_config(transport_params)
        self.training_condition = dict(transport_params.get("training_condition", {}))
        self.noise_channels = self.noise_conditioning.channels
        self.noise_cond_dim = self.noise_conditioning.cond_dim
        self.inference_defaults = transport_params.get("inference_defaults", {})
        self.transport_model_objective = get_transport_model_objective(transport_params.objective)

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
        )

        self.noise_embedder = instantiate(transport_params.noise_embedder)
        self.noise_cond_mlp = self._create_noise_conditioning_mlp()

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        output_dim = super()._calculate_output_dim(dataset_name)
        input_dim = base_input_dim + output_dim  # input history plus corrupted target
        return input_dim

    def _create_noise_conditioning_mlp(self) -> nn.Sequential:
        mlp = nn.Sequential()
        mlp.add_module("linear1_no_gradscaling", nn.Linear(self.noise_channels, self.noise_channels))
        mlp.add_module("activation", nn.SiLU())
        mlp.add_module("linear2_no_gradscaling", nn.Linear(self.noise_channels, self.noise_cond_dim))
        return mlp

    def _assemble_input(
        self,
        x: torch.Tensor,
        y_noised: torch.Tensor,
        bse: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, None, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=bse)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

        # Combine input history, corrupted target, and node position features
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(y_noised, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )

        return x_data_latent, None, grid_shard_sizes

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        x_out = einops.rearrange(
            x_out,
            "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
            batch=batch_size,
            ensemble=ensemble_size,
            time=self.n_step_output,
        ).to(dtype=dtype)

        return x_out

    def _make_noise_emb(self, noise_emb: torch.Tensor, repeat: int) -> torch.Tensor:
        assert noise_emb.ndim in (4, 5), "noise_emb must be 4D or 5D."
        if noise_emb.ndim == 4:
            noise_emb = noise_emb.unsqueeze(3)
        out = einops.repeat(
            noise_emb,
            "batch time ensemble noise_level vars -> batch time ensemble (repeat noise_level) vars",
            repeat=repeat,
        )
        out = einops.rearrange(out, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
        return out

    def _embed_noise_conditioning(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.noise_cond_mlp(self.noise_embedder(sigma))

    def _assert_condition_shapes(self, condition: dict[str, torch.Tensor]) -> tuple[int, int]:
        dataset_names = list(condition)
        condition_ref = condition[dataset_names[0]]
        assert condition_ref.ndim == 5, "Expected condition to have shape (batch, 1, ensemble, 1, 1)."
        batch_size, _, ensemble_size = condition_ref.shape[:3]
        for dataset_name in dataset_names:
            condition_shape = condition[dataset_name].shape
            assert (
                len(condition_shape) == 5
            ), f"Expected condition to have shape (batch, 1, ensemble, 1, 1) for '{dataset_name}'."
            assert (
                condition_shape[1] == condition_shape[3] == condition_shape[4] == 1
            ), f"Expected condition to have shape (batch, 1, ensemble, 1, 1) for '{dataset_name}'."
            assert (
                condition_shape[0] == batch_size and condition_shape[2] == ensemble_size
            ), "Batch or ensemble dimension mismatch across datasets for conditioned inputs."
        return batch_size, ensemble_size

    def _generate_noise_conditioning(
        self,
        noise_cond: torch.Tensor,
        dataset_name: str,
        edge_conditioning: bool = False,
    ) -> torch.Tensor:

        c_data = self._make_noise_emb(noise_cond, repeat=self.node_attributes.num_nodes[dataset_name])
        c_hidden = self._make_noise_emb(noise_cond, repeat=self.node_attributes.num_nodes[self._graph_name_hidden])

        if edge_conditioning:  # currently unused, but available if graph edges need conditioning later
            c_data_to_hidden = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[(dataset_name, "to", self._graph_name_hidden)]["edge_length"].shape[0],
            )
            c_hidden_to_data = self._make_noise_emb(
                noise_cond,
                repeat=self._graph_data[(self._graph_name_hidden, "to", dataset_name)]["edge_length"].shape[0],
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

    def _build_conditioning_kwargs(
        self,
        x: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[dict[str, dict], dict[str, torch.Tensor], dict[str, dict]]:
        self._assert_condition_shapes(condition)
        dataset_names = list(x.keys())

        # Transport assumes one noise level or bridge time per sample and
        # ensemble member, shared across datasets. The training objectives build
        # the condition that way, so we can read it from the first dataset,
        # embed it once, and repeat it over each dataset's graph nodes below.
        condition_base = condition[dataset_names[0]][:, 0, :, 0]
        noise_cond_base = self._embed_noise_conditioning(condition_base)

        fwd_mapper_kwargs, bwd_mapper_kwargs = {}, {}
        for dataset_name in x:
            # The same transport noise/time embedding is shared across all output steps.
            noise_cond = noise_cond_base[:, None, :, None, :]
            c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(
                noise_cond, dataset_name=dataset_name, edge_conditioning=False
            )
            c_data_shard_sizes = get_shard_sizes(c_data, 0, model_comm_group=model_comm_group)
            c_hidden_shard_sizes = get_shard_sizes(c_hidden, 0, model_comm_group=model_comm_group)
            c_data = shard_tensor(c_data, 0, c_data_shard_sizes, model_comm_group)
            c_hidden = shard_tensor(c_hidden, 0, c_hidden_shard_sizes, model_comm_group)

            fwd_mapper_kwargs[dataset_name] = {"cond": (c_data, c_hidden)}
            bwd_mapper_kwargs[dataset_name] = {"cond": (c_hidden, c_data)}

        processor_kwargs = {"cond": c_hidden}
        return fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs

    def forward(
        self,
        x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.transport_model_objective.forward(
            self,
            x,
            conditioned_target,
            condition,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            **kwargs,
        )

    def _forward_transport_network(
        self,
        x: dict[str, torch.Tensor],
        conditioned_target: dict[str, torch.Tensor],
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Multi-dataset case
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Embed the current noise level or bridge time and pass it to the conditional layers.
        fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = self._build_conditioning_kwargs(
            x, condition, model_comm_group=model_comm_group
        )

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict: dict[str, torch.Tensor | None] = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group=model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)
        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                x[dataset_name],
                conditioned_target[dataset_name],
                bse,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_sizes_data_dict[dataset_name] = shard_sizes_data

            (
                encoder_edge_attr,
                encoder_edge_index,
                enc_edge_shard_sizes,
            ) = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=bse,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            x_data_latent, dataset_latents[dataset_name] = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latent),
                batch_size=bse,
                shard_info=enc_shard_info,
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                **fwd_mapper_kwargs[dataset_name],
            )
            x_data_latent_dict[dataset_name] = x_data_latent

        x_latent = sum(dataset_latents.values())

        # Processor
        (
            processor_edge_attr,
            processor_edge_index,
            proc_edge_shard_sizes,
        ) = self.processor_graph_provider.get_edges(
            batch_size=bse,
            model_comm_group=model_comm_group,
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

        if self.latent_skip:
            # Processor skip connection
            x_latent = x_latent_proc + x_latent

        # Decoder
        x_out_dict = {}
        for dataset_name in dataset_names:
            # Compute decoder edges using updated latent representation
            (
                decoder_edge_attr,
                decoder_edge_index,
                dec_edge_shard_sizes,
            ) = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=bse,
                model_comm_group=model_comm_group,
            )

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            x_out = self.decoder[dataset_name](
                (x_latent, x_data_latent_dict[dataset_name]),
                batch_size=bse,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],
                **bwd_mapper_kwargs[dataset_name],
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out, x_skip_dict[dataset_name], batch_size, ensemble_size, x[dataset_name].dtype
            )

        return x_out_dict

    def _before_sampling(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        n_step_input: int,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> tuple[SamplingData, DatasetShardSizes | None]:
        """Prepare batch before sampling.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batch after pre-processing.
        pre_processors : dict[str, nn.Module]
            Pre-processing module (already applied).
        n_step_input : int
            Number of input timesteps.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        **kwargs
            Additional parameters for subclasses.

        Returns
        -------
        tuple[SamplingData, DatasetShardSizes]
            Prepared input tensor(s) and per-dataset grid shard sizes.
            Can return a single tensor or tuple of tensors for sampling input.
        """
        xs = {}
        grid_shard_sizes: DatasetShardSizes | None = None
        if model_comm_group is not None:
            grid_shard_sizes = {}

        for dataset_name, x in batch.items():
            # Dimensions are batch, timesteps, grid, variables
            x = x[:, 0:n_step_input, None, ...]  # add dummy ensemble dimension as 3rd index

            if model_comm_group is not None:
                shard_sizes = get_shard_sizes(x, -2, model_comm_group=model_comm_group)
                assert grid_shard_sizes is not None
                grid_shard_sizes[dataset_name] = shard_sizes
                x = shard_tensor(x, -2, shard_sizes, model_comm_group)
            x = pre_processors[dataset_name](x, in_place=False)

            xs[dataset_name] = x

        return (xs,), grid_shard_sizes

    def _after_sampling(
        self,
        out: dict[str, torch.Tensor],
        post_processors: dict[str, nn.Module],
        before_sampling_data: SamplingData,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Post-process sampled output and gather shards when needed.

        Parameters
        ----------
        out : dict[str, torch.Tensor]
            Sampled output tensor.
        post_processors : dict[str, nn.Module]
            Post-processing module.
        before_sampling_data : SamplingData
            Data returned from _before_sampling (can be used by subclasses).
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset grid shard sizes for gathering. ``None`` means the
            corresponding dataset is replicated, not sharded.
        gather_out : bool
            Whether to gather output.
        **kwargs
            Additional parameters for subclasses.

        Returns
        -------
        torch.Tensor
            Post-processed output.
        """
        for dataset_name in out.keys():
            out[dataset_name] = post_processors[dataset_name](out[dataset_name], in_place=False)

            if gather_out and model_comm_group is not None:
                assert grid_shard_sizes is not None
                out[dataset_name] = gather_tensor(
                    out[dataset_name],
                    -2,
                    grid_shard_sizes[dataset_name],
                    model_comm_group,
                )

        return out

    def build_sampling_source(
        self,
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        default_kind: str = "gaussian",
    ) -> dict[str, torch.Tensor]:
        """Build the starting/source field used by transport sampling."""
        request = TransportSourceRequest(
            specs=sampling_source_specs(
                x,
                n_step_output=self.n_step_output,
                num_output_channels=self.num_output_channels,
                grid_shard_sizes=grid_shard_sizes,
            ),
            default_kind=default_kind,
            custom_source_factories={
                "reference_state": lambda: reference_state_sampling_source(
                    x,
                    data_indices=self.data_indices,
                    n_step_output=self.n_step_output,
                ),
            },
            model_comm_group=model_comm_group,
            error_context="state prediction",
        )
        return self.transport_source.build(request)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        post_processors: dict[str, nn.Module],
        n_step_input: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        pre_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        post_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run inference by sampling from the selected transport objective.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batched data (before pre-processing).
        pre_processors : dict[str, nn.Module]
            Pre-processing module.
        post_processors : dict[str, nn.Module]
            Post-processing module.
        n_step_input : int
            Number of input timesteps.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        gather_out : bool
            Whether to gather output tensors across distributed processes.
        schedule_params : Optional[dict]
            Dictionary of sampling schedule parameters (schedule_type, num_steps, etc.)
            These will override the default values from inference_defaults.
        sampler_params : Optional[dict]
            Dictionary of sampler parameters (sampler, S_churn, S_min, S_max, S_noise, etc.)
            These will override the default values from inference_defaults.
        pre_processors_tendencies : Optional[dict[str, nn.Module]]
            Pre-processing module for tendencies (used by subclasses).
        post_processors_tendencies : Optional[dict[str, nn.Module]]
            Post-processing module for tendencies (used by subclasses).
        **kwargs
            Additional sampling parameters.

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled output (after post-processing).
        """
        with torch.no_grad():

            assert isinstance(batch, dict), "Input batch must be a dictionary!"
            for dataset_name, dataset_tensor in batch.items():
                assert (
                    len(dataset_tensor.shape) == 4
                ), f'The input tensor "{dataset_name}" has an incorrect shape: expected a 4-dimensional tensor, got {dataset_tensor.shape}!'

            # Before sampling hook
            before_sampling_data, grid_shard_sizes = self._before_sampling(
                batch,
                pre_processors,
                n_step_input,
                model_comm_group,
                pre_processors_tendencies=pre_processors_tendencies,
                post_processors_tendencies=post_processors_tendencies,
                **kwargs,
            )

            x = before_sampling_data[0]

            out = self.sample(
                x,
                model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
                schedule_params=schedule_params,
                sampler_params=sampler_params,
                **kwargs,
            )
            for dataset_name in out:
                out[dataset_name] = out[dataset_name].to(batch[dataset_name].dtype)

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
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run the sampler selected by the transport objective."""
        return self.transport_model_objective.sample(
            self,
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            schedule_params=schedule_params,
            sampler_params=sampler_params,
            **kwargs,
        )

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.n_step_input,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes


class AnemoiTransportTendModelEncProcDec(AnemoiTransportModelEncProcDec):
    """Transport model that predicts tendencies and converts them back to state fields."""

    def __init__(
        self,
        *,
        model_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        n_step_input: int,
        n_step_output: int,
        graph_data: HeteroData,
    ) -> None:
        model_config = DotDict(model_config)

        self.condition_on_residual = model_config.model.condition_on_residual
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
            graph_data=graph_data,
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        input_dim = super()._calculate_input_dim(dataset_name)
        if self.condition_on_residual:
            input_dim += len(self.data_indices[dataset_name].model.input.prognostic) * self.n_step_output
        return input_dim

    @staticmethod
    def _apply_imputer_inverse(
        post_processors: dict[str, nn.Module],
        dataset_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        processors = post_processors[dataset_name]
        if not hasattr(processors, "processors"):
            return x
        for processor in processors.processors.values():
            if getattr(processor, "supports_skip_imputation", False):
                x = processor(x, in_place=False, inverse=True, skip_imputation=False)
        return x

    def _assemble_input(
        self,
        x: torch.Tensor,
        y_noised: torch.Tensor,
        bse: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=bse)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        x_skip = self.residual[dataset_name](x, grid_shard_sizes, model_comm_group, n_step_output=self.n_step_output)[
            ..., self._internal_input_idx[dataset_name]
        ]
        assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
        x_skip = einops.rearrange(x_skip, "batch time ensemble grid vars -> (batch ensemble) grid (time vars)")

        # Shard node attributes if grid sharding is enabled
        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

        # Combine input history, corrupted target, and node position features
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(y_noised, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        if self.condition_on_residual:
            x_data_latent = torch.cat(
                (x_data_latent, einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars")), dim=-1
            )

        return x_data_latent, x_skip, grid_shard_sizes

    def compute_tendency(
        self,
        x_t1: dict[str, torch.Tensor],
        x_t0: dict[str, torch.Tensor],
        pre_processors_state: dict[str, Callable],
        pre_processors_tendencies: dict[str, Callable],
        input_post_processor: dict[str, Callable | None] | None = None,
        skip_imputation: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute the tendency from two states.

        Parameters
        ----------
        x_t1 : torch.Tensor
            Target state.
        x_t0 : torch.Tensor
            Reference state.
        pre_processors_state : callable
            Function to pre-process the state variables.
        pre_processors_tendencies : callable
            Function to pre-process the tendency variables.
        input_post_processor : Optional[Callable], optional
            Function to post-process the input state variables. If provided,
            the input states will be post-processed before computing the tendency.
            If None, the input states are used directly. Default is None.
        skip_imputation : bool, optional
            When True, skip imputation in the state/tendency processors and input_post_processor.
            Defaults to False.

        Returns
        -------
        torch.Tensor
            Normalized tendency fields in the model-output variable order.
        """
        tendencies = {}

        assert set(x_t1.keys()) == set(x_t0.keys()), "x_t1 and x_t0 must have the same dataset keys."

        for dataset_name in x_t1.keys():
            input_post_proc = input_post_processor[dataset_name] if input_post_processor is not None else None
            if input_post_proc is not None:
                x_t1[dataset_name] = input_post_proc(
                    x_t1[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.output.full,
                    skip_imputation=skip_imputation,
                )
                x_t0[dataset_name] = input_post_proc(
                    x_t0[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.input.prognostic,
                    skip_imputation=skip_imputation,
                )

            tendency = x_t1[dataset_name].clone()
            tendency[..., self.data_indices[dataset_name].model.output.prognostic] = pre_processors_tendencies[
                dataset_name
            ](
                x_t1[dataset_name][..., self.data_indices[dataset_name].model.output.prognostic] - x_t0[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.prognostic,
                skip_imputation=skip_imputation,
            )
            # Diagnostic variables are kept as normalized full fields from x_t1.
            tendency[..., self.data_indices[dataset_name].model.output.diagnostic] = pre_processors_state[dataset_name](
                x_t1[dataset_name][..., self.data_indices[dataset_name].model.output.diagnostic],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.diagnostic,
                skip_imputation=skip_imputation,
            )
            tendencies[dataset_name] = tendency

        return tendencies

    def add_tendency_to_state(
        self,
        state_inp: dict[str, torch.Tensor],
        tendency: dict[str, torch.Tensor],
        post_processors_state: dict[str, Callable],
        post_processors_tendencies: dict[str, Callable],
        output_pre_processor: dict[str, Callable | None] | None = None,
        skip_imputation: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convert predicted tendencies back to state fields.

        Parameters
        ----------
        state_inp : dict[str, torch.Tensor]
            Normalized reference state with prognostic input variables.
        tendency : dict[str, torch.Tensor]
            Normalized tendency fields predicted by the model.
        post_processors_state : dict[str, Callable]
            Function to post-process the state variables.
        post_processors_tendencies : dict[str, Callable]
            Function to post-process the tendency variables.
        output_pre_processor : Optional[Callable], optional
            Function to pre-process the output state. If provided,
            the output state will be pre-processed before returning.
            If None, the output state is returned directly. Default is None.
        skip_imputation : bool, optional
            When True, skip imputation in the state/tendency processors.
            Defaults to False.

        Returns
        -------
        dict[str, torch.Tensor]
            De-normalized state fields.
        """
        state_outp = {}

        for dataset_name in tendency.keys():
            state_outp[dataset_name] = post_processors_tendencies[dataset_name](
                tendency[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.full,
                skip_imputation=skip_imputation,
            )

            state_outp[dataset_name][
                ..., self.data_indices[dataset_name].model.output.diagnostic
            ] = post_processors_state[dataset_name](
                tendency[dataset_name][..., self.data_indices[dataset_name].model.output.diagnostic],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.output.diagnostic,
                skip_imputation=skip_imputation,
            )

            state_outp[dataset_name][
                ..., self.data_indices[dataset_name].model.output.prognostic
            ] += post_processors_state[dataset_name](
                state_inp[dataset_name],
                in_place=False,
                data_index=self.data_indices[dataset_name].data.input.prognostic,
                skip_imputation=skip_imputation,
            )

            output_pre_proc = output_pre_processor[dataset_name] if output_pre_processor is not None else None
            if output_pre_proc is not None:
                state_outp[dataset_name] = output_pre_proc(
                    state_outp[dataset_name],
                    in_place=False,
                    data_index=self.data_indices[dataset_name].data.output.full,
                    skip_imputation=skip_imputation,
                )

        return state_outp

    def _before_sampling(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        n_step_input: int,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> tuple[SamplingData, DatasetShardSizes | None]:
        """Prepare batch before sampling.

        Returns (xs, x_t0s) and grid shard sizes per dataset.
        """
        xs = {}
        x_t0s = {}
        grid_shard_sizes: DatasetShardSizes | None = None
        if model_comm_group is not None:
            grid_shard_sizes = {}

        for dataset_name, x in batch.items():
            # Dimensions are batch, timesteps, grid, variables
            x_in = x[:, 0:n_step_input, None, ...]  # add dummy ensemble dimension as 3rd index
            x_t0 = x[:, -1:, None, ...]  # keep time dim and add dummy ensemble dimension

            if model_comm_group is not None:
                shard_sizes = get_shard_sizes(x_in, -2, model_comm_group=model_comm_group)
                assert grid_shard_sizes is not None
                grid_shard_sizes[dataset_name] = shard_sizes
                x_in = shard_tensor(x_in, -2, shard_sizes, model_comm_group)
                shard_sizes = get_shard_sizes(x_t0, -2, model_comm_group=model_comm_group)
                x_t0 = shard_tensor(x_t0, -2, shard_sizes, model_comm_group)

            x_in = pre_processors[dataset_name](x_in, in_place=False)
            x_t0 = pre_processors[dataset_name](x_t0, in_place=False)

            xs[dataset_name] = x_in
            x_t0s[dataset_name] = x_t0

        return (xs, x_t0s), grid_shard_sizes

    def build_sampling_source(
        self,
        x: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        default_kind: str = "gaussian",
    ) -> dict[str, torch.Tensor]:
        """Build the starting/source field for tendency-space transport sampling."""
        # Tendency prediction can use Gaussian noise, zeros, or the latest
        # input state projected to the variables the model predicts.
        request = TransportSourceRequest(
            specs=sampling_source_specs(
                x,
                n_step_output=self.n_step_output,
                num_output_channels=self.num_output_channels,
                grid_shard_sizes=grid_shard_sizes,
            ),
            default_kind=default_kind,
            custom_source_factories={
                "reference_state": lambda: reference_state_sampling_source(
                    x,
                    data_indices=self.data_indices,
                    n_step_output=self.n_step_output,
                ),
            },
            model_comm_group=model_comm_group,
            error_context="tendency prediction",
        )
        return self.transport_source.build(request)

    def _after_sampling(
        self,
        out: dict[str, torch.Tensor],
        post_processors: dict[str, nn.Module],
        before_sampling_data: SamplingData,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        gather_out: bool = True,
        post_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Convert sampled tendencies into state predictions."""
        if isinstance(before_sampling_data, tuple) and len(before_sampling_data) >= 2:
            x_t0s = before_sampling_data[1]
        else:
            raise ValueError("Expected before_sampling_data to contain x_t0s")

        x_t0s = self.apply_reference_state_truncation(x_t0s, grid_shard_sizes, model_comm_group)
        x_refs = {}
        for dataset_name, ref in x_t0s.items():
            assert ref.ndim == 5, f"Expected 5D reference state for '{dataset_name}', got {ref.ndim}D."
            x_refs[dataset_name] = ref[:, -1]
        assert post_processors_tendencies is not None, "Per-step tendency processors must be provided."

        for dataset_name, out_dataset in out.items():
            post_tend = post_processors_tendencies[dataset_name]
            assert post_tend is not None, "Tendency processors must be provided per dataset."
            if not isinstance(post_tend, StepwiseProcessors):
                # Single-output tendency models may still provide one flat
                # Processors object. Treat it as the only output-step processor.
                # Multi-output models need an explicit processor for each lead time.
                assert (
                    self.n_step_output == 1
                ), "Per-step tendency processors must be provided for multiple output steps."
                post_tend = [post_tend]
            assert (
                len(post_tend) == out_dataset.shape[1]
            ), "Per-step tendency processors must match the number of output steps."

            states = []
            for step, post_proc in enumerate(post_tend):
                out_step = out_dataset[:, step : step + 1]
                state_step = self.add_tendency_to_state(
                    {dataset_name: x_refs[dataset_name].unsqueeze(1)},
                    {dataset_name: out_step},
                    {dataset_name: post_processors[dataset_name]},
                    {dataset_name: post_proc},
                    skip_imputation=True,
                )[dataset_name]
                states.append(state_step)

            out_dataset = torch.cat(states, dim=1)
            out_dataset = self._apply_imputer_inverse(post_processors, dataset_name, out_dataset)
            if gather_out and model_comm_group is not None:
                assert grid_shard_sizes is not None
                out_dataset = gather_tensor(
                    out_dataset,
                    -2,
                    grid_shard_sizes[dataset_name],
                    model_comm_group,
                )
            out[dataset_name] = out_dataset

        return out

    def apply_reference_state_truncation(
        self,
        x: dict[str, torch.Tensor],
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: Optional[ProcessGroup],
    ) -> dict[str, torch.Tensor]:
        """Project the latest input state to the variables needed as the tendency reference.

        The tendency model predicts changes relative to this reference state.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input tensor with shape {dataset_name: (batch, time, ensemble, grid, variables)}.
        grid_shard_sizes : DatasetShardSizes
            Per-dataset grid shard sizes used when the model grid is sharded.
        model_comm_group : ProcessGroup
            Communication group used by model-parallel grid sharding.

        Returns
        -------
        dict[str, torch.Tensor]
            Reference states containing the prognostic input variables.
        """
        x_skips = {}

        for dataset_name, in_x in x.items():
            grid_shard_sizes_i = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None
            x_skip = self.residual[dataset_name](
                in_x, grid_shard_sizes_i, model_comm_group, n_step_output=self.n_step_output
            )
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            # Keep only prognostic input variables, matching the tendency reference state.
            x_skips[dataset_name] = x_skip[..., self.data_indices[dataset_name].model.input.prognostic]

        return x_skips
