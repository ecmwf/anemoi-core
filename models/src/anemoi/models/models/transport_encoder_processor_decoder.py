# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import TYPE_CHECKING
from typing import Callable
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data.batch import STATIC_COORDS_META_KEY
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.encoder_processor_decoder import latlons_to_sincos
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.models.transport import EdmSettings
from anemoi.models.transport import NoiseConditioningSettings
from anemoi.models.transport import StochasticInterpolantSettings
from anemoi.models.transport import TransportSourceBuilder
from anemoi.models.transport import TransportSourceRequest
from anemoi.models.transport import get_transport_model_objective
from anemoi.models.transport import reference_state_sampling_source
from anemoi.models.transport import sampling_source_specs
from anemoi.models.transport.data_helpers import Data
from anemoi.models.transport.data_helpers import data_device
from anemoi.models.transport.data_helpers import map_data
from anemoi.utils.config import DotDict

if TYPE_CHECKING:
    from anemoi.models.data.views import SourceView

LOGGER = logging.getLogger(__name__)

SamplingData = tuple[Batch, ...]


def _cat_feature_data(a: Data, b: Data) -> Data:
    """Concatenate two per-dataset payloads along the variable axis."""
    if isinstance(a, list):
        return [torch.cat([a_sample, b_sample.to(a_sample.dtype)], dim=-1) for a_sample, b_sample in zip(a, b)]
    return torch.cat([a, b.to(a.dtype)], dim=-1)


class AnemoiTransportModelEncProcDec(AnemoiModelEncProcDec):
    """Encoder-processor-decoder model conditioned on diffusion noise level or bridge time."""

    def __init__(
        self,
        *,
        model_config: DictConfig,
        model_graph_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        is_dataset_static: dict[str, bool],
        n_step_input: int,
        n_step_output: int,
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
            model_graph_config=model_graph_config,
            data_indices=data_indices,
            statistics=statistics,
            is_dataset_static=is_dataset_static,
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

    def _calculate_target_dim(self, dataset_name: str) -> int:
        target_dim = super()._calculate_target_dim(dataset_name)
        if not self.use_encoder_data_output[dataset_name]:
            # In addition to the deterministic decoder inputs (output-time
            # decoding forcings plus coordinates), the transport decoder
            # consumes the corrupted target values at the target locations.
            target_dim += super()._calculate_output_dim(dataset_name)
        return target_dim

    def _create_noise_conditioning_mlp(self) -> nn.Sequential:
        mlp = nn.Sequential()
        mlp.add_module("linear1_no_gradscaling", nn.Linear(self.noise_channels, self.noise_channels))
        mlp.add_module("activation", nn.SiLU())
        mlp.add_module("linear2_no_gradscaling", nn.Linear(self.noise_channels, self.noise_cond_dim))
        return mlp

    def _assemble_input(
        self,
        x: "SourceView",
        y_noised: "SourceView",
        bse: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        x_features = x.flatten()
        y_noised_features = y_noised.flatten()
        grid_shard_sizes = x_features.shard_sizes
        same_coordinates = torch.equal(x_features.coordinates, y_noised_features.coordinates)
        if same_coordinates:
            data_coords = x_features.coordinates
            x_input_features = x_features.data
        else:
            if not isinstance(x.data, list) or not isinstance(y_noised.data, list):
                msg = "Input and conditioned target coordinates must match for dense transport data."
                raise AssertionError(msg)
            data_coords = y_noised_features.coordinates
            x_input_features = torch.zeros(
                y_noised_features.data.shape[0],
                x_features.data.shape[-1],
                device=y_noised_features.data.device,
                dtype=y_noised_features.data.dtype,
            )

        inputs = [
            x_input_features,
            y_noised_features.data,
            latlons_to_sincos(data_coords),
        ]

        if dataset_name in self.node_attributes:
            node_attributes_data = self.node_attributes(dataset_name, batch_size=bse).to(y_noised_features.data.device)
            if node_attributes_data.shape[0] != y_noised_features.data.shape[0]:
                msg = (
                    "Trainable node attributes are not implemented for dynamic sparse transport nodes. "
                    f"Dataset '{dataset_name}' has {y_noised_features.data.shape[0]} target nodes, "
                    f"but static node attributes provide {node_attributes_data.shape[0]} rows."
                )
                raise NotImplementedError(msg)
            if grid_shard_sizes is not None:
                node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)
            inputs.append(node_attributes_data)

        x_data_latent = torch.cat(inputs, dim=-1)

        return data_coords, x_data_latent, None, grid_shard_sizes

    def _assemble_output(self, x_out, x_skip, target: "SourceView", dtype: torch.dtype, dataset_name: str):
        del x_skip
        pred = target.unflatten(x_out.to(dtype=dtype))
        pred = self.boundings[dataset_name](pred)

        return pred

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

    def _make_noise_emb_for_view(self, noise_emb: torch.Tensor, view: "SourceView") -> torch.Tensor:
        """Repeat noise embeddings over the actual flattened nodes in a source view."""
        if not isinstance(view.data, list):
            grid_size = view.data.shape[view.layout.axis("grid", ndim=view.data.ndim)]
            return self._make_noise_emb(noise_emb, repeat=grid_size)

        noise_base = noise_emb[:, 0, :, 0, :]
        chunks = []
        for sample_index, sample in enumerate(view.data):
            if sample.ndim != 2:
                msg = "Sparse transport conditioning currently expects per-sample data shaped " "(nodes, variables)."
                raise NotImplementedError(msg)
            if noise_base.shape[1] != 1:
                msg = "Sparse observation transport without an ensemble axis requires ensemble_size == 1."
                raise NotImplementedError(msg)
            num_nodes = sample.shape[view.layout.axis("grid", ndim=sample.ndim)]
            chunks.append(noise_base[sample_index, 0].expand(num_nodes, -1).to(sample.device))
        return torch.cat(chunks, dim=0)

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
        data_view: Optional["SourceView"] = None,
        edge_conditioning: bool = False,
    ) -> torch.Tensor:

        if data_view is None:
            c_data = self._make_noise_emb(noise_cond, repeat=self.node_attributes.num_nodes[dataset_name])
        else:
            c_data = self._make_noise_emb_for_view(noise_cond, data_view)
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
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> tuple[dict[str, dict], dict[str, torch.Tensor], dict[str, dict]]:
        self._assert_condition_shapes(condition)
        dataset_names = list(conditioned_target.keys())

        # Transport assumes one noise level or bridge time per sample and
        # ensemble member, shared across datasets. The training objectives build
        # the condition that way, so we can read it from the first dataset,
        # embed it once, and repeat it over each dataset's graph nodes below.
        condition_base = condition[dataset_names[0]][:, 0, :, 0]
        noise_cond_base = self._embed_noise_conditioning(condition_base)

        fwd_mapper_kwargs, bwd_mapper_kwargs = {}, {}
        for dataset_name in dataset_names:
            # The same transport noise/time embedding is shared across all output steps.
            noise_cond = noise_cond_base[:, None, :, None, :]
            c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(
                noise_cond,
                dataset_name=dataset_name,
                data_view=conditioned_target[dataset_name],
                edge_conditioning=False,
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
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        target_forcing: Optional[Batch] = None,
        **kwargs,
    ) -> Batch:
        return self.transport_model_objective.forward(
            self,
            x,
            conditioned_target,
            condition,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            target_forcing=target_forcing,
            **kwargs,
        )

    def _forward_transport_network(
        self,
        batch: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        target_forcing: Optional[Batch] = None,
        **kwargs,
    ) -> Batch:
        # Multi-dataset case
        dataset_names = list(batch.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(batch, 0)
        ensemble_size = self._get_consistent_dim(batch, 2)

        bse = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = self._resolve_in_out_sharded(batch)
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Embed the current noise level or bridge time and pass it to the conditional layers.
        fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs = self._build_conditioning_kwargs(
            conditioned_target, condition, model_comm_group=model_comm_group
        )

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict: dict[str, torch.Tensor | None] = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}

        hidden_coordinates = self._hidden_coordinates().to(
            batch.device
        )  # todo Simon, do we need this device movement here?
        x_hidden_latent = latlons_to_sincos(hidden_coordinates)
        x_hidden_latent = einops.repeat(x_hidden_latent, "n f -> (repeat n) f", repeat=bse)
        hidden_trainable_parameters = self.node_attributes(self._graph_name_hidden, batch_size=bse)
        if hidden_trainable_parameters is not None:
            hidden_trainable_parameters = hidden_trainable_parameters.to(
                x_hidden_latent.device
            )  # todo Simon, do we need this device movement?
            x_hidden_latent = torch.cat([x_hidden_latent, hidden_trainable_parameters], dim=-1)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group=model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)
        for dataset_name in dataset_names:
            data_coords, x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                batch[dataset_name],
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
                src_coords=data_coords,
                dst_coords=hidden_coordinates,
                model_comm_group=model_comm_group,
            )
            encoder_edge_attr = encoder_edge_attr.to(x_data_latent.device)  # todo SL: remove device movement
            encoder_edge_index = encoder_edge_index.to(x_data_latent.device)

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
            src_coords=hidden_coordinates,
            dst_coords=hidden_coordinates,
            batch_size=bse,
            model_comm_group=model_comm_group,
        )
        processor_edge_attr = processor_edge_attr.to(x_latent.device)  # todo SL: remove device movement
        processor_edge_index = processor_edge_index.to(x_latent.device)

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
            x_latent_proc = x_latent_proc + x_latent

        # Decoder
        out_batch = conditioned_target
        for dataset_name in dataset_names:
            decoder_target_view = conditioned_target[dataset_name]
            if not self.use_encoder_data_output[dataset_name]:
                # Match the deterministic decoder: condition the target nodes on
                # the output-time decoding forcings alongside the corrupted target.
                if target_forcing is None or dataset_name not in target_forcing:
                    msg = (
                        f"Dataset '{dataset_name}' decodes from target-side features; a 'target_forcing' "
                        "batch with the output-time decoding forcings is required."
                    )
                    raise ValueError(msg)
                decoder_target_view = decoder_target_view.clone(
                    data=_cat_feature_data(decoder_target_view.data, target_forcing[dataset_name].data),
                )
            target_coords, target_data_latent, shard_sizes_target, _target_batch_sizes = self._assemble_target(
                decoder_target_view,
                x_data_latent_dict.get(dataset_name, None),
                batch_size=bse,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            # Compute decoder edges using updated latent representation
            (
                decoder_edge_attr,
                decoder_edge_index,
                dec_edge_shard_sizes,
            ) = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=bse,
                src_coords=hidden_coordinates,
                dst_coords=target_coords,
                model_comm_group=model_comm_group,
            )
            decoder_edge_attr = decoder_edge_attr.to(x_latent.device)  # todo SL: remove device movement
            decoder_edge_index = decoder_edge_index.to(x_latent.device)

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_target,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            x_out = self.decoder[dataset_name](
                (x_latent_proc, target_data_latent),
                batch_size=bse,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],
                **bwd_mapper_kwargs[dataset_name],
            )

            target_view = conditioned_target[dataset_name]
            target_data = target_view.data[0] if isinstance(target_view.data, list) else target_view.data
            out_view = self._assemble_output(
                x_out,
                x_skip_dict[dataset_name],
                target_view,
                target_data.dtype,
                dataset_name,
            )
            out_batch = out_batch.update_source(dataset_name, out_view)

        return out_batch

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

        return (
            self._make_sampling_batch(
                xs,
                variable_space="input",
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
            ),
        ), grid_shard_sizes

    def _after_sampling(
        self,
        out: Batch,
        post_processors: dict[str, nn.Module],
        before_sampling_data: SamplingData,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, Data]:
        """Post-process sampled output and gather shards when needed.

        Parameters
        ----------
        out : Batch
            Sampled output batch.
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
        dict[str, Data]
            Post-processed output data.
        """
        out_data: dict[str, Data] = {}
        for dataset_name in out.keys():
            processed = post_processors[dataset_name](out[dataset_name], in_place=False)
            dataset_data = processed.data

            if gather_out and model_comm_group is not None:
                assert grid_shard_sizes is not None
                if isinstance(dataset_data, list):
                    msg = "Distributed gather is not supported for sparse transport sampling outputs."
                    raise NotImplementedError(msg)
                dataset_data = gather_tensor(
                    dataset_data,
                    -2,
                    grid_shard_sizes[dataset_name],
                    model_comm_group,
                )

            out_data[dataset_name] = dataset_data

        return out_data

    def _sampling_variables(self, dataset_name: str, variable_space: str) -> list[str]:
        if variable_space == "input":
            indices = self.data_indices[dataset_name].model.input
        elif variable_space == "output":
            indices = self.data_indices[dataset_name].model.output
        else:
            msg = f"Unknown sampling variable space {variable_space!r}; expected 'input' or 'output'."
            raise ValueError(msg)
        return list(indices.ordered_names)

    def _sampling_statistics(self, dataset_name: str, variable_space: str) -> dict:
        variables = self._sampling_variables(dataset_name, variable_space)
        name_to_index = self.data_indices[dataset_name].name_to_index
        positions = [name_to_index[name] for name in variables]
        statistics = {}
        for key, value in self.statistics.get(dataset_name, {}).items():
            if hasattr(value, "__getitem__") and hasattr(value, "shape") and len(value.shape) > 0:
                statistics[key] = value[positions]
            else:
                statistics[key] = value
        return statistics

    def _sampling_coordinates(
        self,
        dataset_name: str,
        dataset_data: Data,
        *,
        layout: TensorLayout,
        template: Batch | None,
        model_comm_group: Optional[ProcessGroup],
        grid_shard_sizes: DatasetShardSizes | None,
    ) -> torch.Tensor | list[torch.Tensor]:
        if template is not None and dataset_name in template.coordinates:
            coordinates = template.coordinates[dataset_name]
            dataset_grid_shard_sizes = grid_shard_sizes.get(dataset_name) if grid_shard_sizes is not None else None
            if dataset_grid_shard_sizes is None:
                return coordinates
            if isinstance(dataset_data, list) or isinstance(coordinates, list):
                msg = "Grid sharding is not supported for sparse transport sampling templates."
                raise NotImplementedError(msg)

            coordinates = coordinates.to(data_device(dataset_data))
            data_grid_size = dataset_data.shape[layout.axis("grid", ndim=dataset_data.ndim)]
            if coordinates.ndim == 2:
                coordinate_grid_dim = 0
            elif coordinates.ndim == 3:
                coordinate_grid_dim = 1
            else:
                msg = (
                    "Sampling template coordinates must have shape (grid, 2) or (batch, grid, 2), "
                    f"got {tuple(coordinates.shape)} for dataset '{dataset_name}'."
                )
                raise ValueError(msg)

            coordinate_grid_size = coordinates.shape[coordinate_grid_dim]
            if coordinate_grid_size == data_grid_size:
                return coordinates

            full_grid_size = sum(dataset_grid_shard_sizes)
            if coordinate_grid_size == full_grid_size:
                return shard_tensor(coordinates, coordinate_grid_dim, dataset_grid_shard_sizes, model_comm_group)

            msg = (
                f"Sampling template coordinates for dataset '{dataset_name}' have grid size "
                f"{coordinate_grid_size}, but sampled data has grid size {data_grid_size} and full sharded grid size "
                f"{full_grid_size}."
            )
            raise ValueError(msg)

        if isinstance(dataset_data, list):
            msg = (
                "Sparse transport sampling requires a Batch template carrying per-sample coordinates. "
                f"Dataset '{dataset_name}' has sparse data but no template coordinates."
            )
            raise NotImplementedError(msg)

        if not self.is_dataset_static.get(dataset_name, False):
            msg = (
                "Transport inference for non-static gridded datasets requires a Batch template carrying coordinates. "
                f"Dataset '{dataset_name}' has no template coordinates."
            )
            raise NotImplementedError(msg)

        try:
            coordinates = self._graph_data[dataset_name].x
        except Exception as exc:
            msg = (
                f"Cannot infer sampling coordinates for dataset '{dataset_name}' from the graph. "
                "Pass a Batch with coordinates instead."
            )
            raise NotImplementedError(msg) from exc

        coordinates = coordinates.to(data_device(dataset_data))
        if grid_shard_sizes is not None and grid_shard_sizes.get(dataset_name) is not None:
            coordinates = shard_tensor(coordinates, 0, grid_shard_sizes[dataset_name], model_comm_group)
        return coordinates

    def _make_sampling_batch(
        self,
        data: dict[str, Data],
        *,
        variable_space: str,
        template: Batch | None = None,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
    ) -> Batch:
        layouts = {}
        coordinates = {}
        variables = {}
        statistics = {}
        grid_sizes = {}
        for dataset_name, dataset_data in data.items():
            if template is not None and dataset_name in template.layouts:
                layouts[dataset_name] = template.layouts[dataset_name]
            elif isinstance(dataset_data, list):
                layouts[dataset_name] = TensorLayout(grid=0, variables=1, time_in_grid=True)
            else:
                layouts[dataset_name] = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)

            coordinates[dataset_name] = self._sampling_coordinates(
                dataset_name,
                dataset_data,
                layout=layouts[dataset_name],
                template=template,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
            )
            variables[dataset_name] = self._sampling_variables(dataset_name, variable_space)
            statistics[dataset_name] = self._sampling_statistics(dataset_name, variable_space)
            grid_sizes[dataset_name] = (
                sum(sample.shape[layouts[dataset_name].axis("grid", ndim=sample.ndim)] for sample in dataset_data)
                if isinstance(dataset_data, list)
                else dataset_data.shape[layouts[dataset_name].axis("grid", ndim=dataset_data.ndim)]
            )

        if template is not None:
            metadata = dict(template.metadata)
            static_coords = template.static_coord_datasets & set(data)
        else:
            metadata = {}
            static_coords = frozenset(name for name in data if self.is_dataset_static.get(name, False))
        metadata[STATIC_COORDS_META_KEY] = frozenset(static_coords)

        return Batch(
            data=data,
            coordinates=coordinates,
            metadata=metadata,
            grid_sizes=grid_sizes,
            timedeltas=(
                {}
                if template is None
                else {name: template.timedeltas[name] for name in data if name in template.timedeltas}
            ),
            shard_sizes={} if grid_shard_sizes is None else grid_shard_sizes,
            layouts=layouts,
            variables=variables,
            statistics=statistics,
        )

    def build_sampling_source(
        self,
        x: Batch,
        *,
        target_template: Batch,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        default_kind: str = "gaussian",
    ) -> dict[str, Data]:
        """Build the starting/source field used by transport sampling."""
        request = TransportSourceRequest(
            specs=sampling_source_specs(
                target_template.data,
                num_output_channels=self.num_output_channels,
                grid_shard_sizes=grid_shard_sizes,
            ),
            default_kind=default_kind,
            custom_source_factories={
                "reference_state": lambda: reference_state_sampling_source(
                    x.data,
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
        target_template: Batch,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        pre_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        post_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        target_forcing: Optional[Batch] = None,
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
        target_template : Batch
            Output Batch template carrying the coordinates, sparse observation
            boundaries, timedeltas and layouts to sample onto.
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
        target_forcing : Optional[Batch]
            Output-time decoding forcings (raw values) at the target locations.
            Normalized here with the input pre-processors and used to condition
            the decoder. Required for datasets that decode from target-side
            features (e.g. observations).
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

            if target_forcing is not None:
                # Normalize the output-time decoding forcings like the model inputs.
                for dataset_name in list(target_forcing.keys()):
                    if dataset_name in pre_processors:
                        target_forcing = target_forcing.update_source(
                            dataset_name,
                            pre_processors[dataset_name](target_forcing[dataset_name], in_place=False),
                        )

            out = self.sample(
                x,
                target_template=target_template,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
                schedule_params=schedule_params,
                sampler_params=sampler_params,
                target_forcing=target_forcing,
                **kwargs,
            )
            out = out.with_data(
                {
                    dataset_name: map_data(
                        out.data[dataset_name],
                        lambda sample, name=dataset_name: sample.to(batch[name].dtype),
                    )
                    for dataset_name in out.keys()
                },
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
        x: Batch,
        *,
        target_template: Batch,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        **kwargs,
    ) -> Batch:
        """Run the sampler selected by the transport objective."""
        return self.transport_model_objective.sample(
            self,
            x,
            target_template=target_template,
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
        model_graph_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        is_dataset_static: dict[str, bool],
        n_step_input: int,
        n_step_output: int,
    ) -> None:
        model_config = DotDict(model_config)

        self.condition_on_residual = model_config.model.condition_on_residual
        super().__init__(
            model_config=model_config,
            model_graph_config=model_graph_config,
            data_indices=data_indices,
            statistics=statistics,
            is_dataset_static=is_dataset_static,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
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
        del post_processors, dataset_name
        return x

    def _assemble_input(
        self,
        x: "SourceView",
        y_noised: "SourceView",
        bse: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        if isinstance(x.data, list) or isinstance(y_noised.data, list):
            msg = "Tendency transport is not implemented for sparse observation datasets."
            raise NotImplementedError(msg)

        data_coords, x_data_latent, _x_skip, shard_sizes_data = super()._assemble_input(
            x,
            y_noised,
            bse,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            dataset_name=dataset_name,
        )

        x_skip = None
        if self.condition_on_residual:
            x_skip = self.residual[dataset_name](
                x.data,
                shard_sizes_data,
                model_comm_group,
                n_step_output=self.n_step_output,
            )[..., self._internal_input_idx[dataset_name]]
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            x_skip = einops.rearrange(x_skip, "batch time ensemble grid vars -> (batch ensemble) grid (time vars)")
            x_data_latent = torch.cat(
                (x_data_latent, einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars")), dim=-1
            )

        return data_coords, x_data_latent, x_skip, shard_sizes_data

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

        x_batch = self._make_sampling_batch(
            xs,
            variable_space="input",
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        x_t0_batch = self._make_sampling_batch(
            x_t0s,
            variable_space="input",
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        return (x_batch, x_t0_batch), grid_shard_sizes

    def build_sampling_source(
        self,
        x: Batch,
        *,
        target_template: Batch,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        default_kind: str = "gaussian",
    ) -> dict[str, Data]:
        """Build the starting/source field for tendency-space transport sampling."""
        # Tendency prediction can use Gaussian noise, zeros, or the latest
        # input state projected to the variables the model predicts.
        request = TransportSourceRequest(
            specs=sampling_source_specs(
                target_template.data,
                num_output_channels=self.num_output_channels,
                grid_shard_sizes=grid_shard_sizes,
            ),
            default_kind=default_kind,
            custom_source_factories={
                "reference_state": lambda: reference_state_sampling_source(
                    x.data,
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
        out: Batch,
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

        x_t0s = self.apply_reference_state_truncation(x_t0s.data, grid_shard_sizes, model_comm_group)
        x_refs = {}
        for dataset_name, ref in x_t0s.items():
            assert ref.ndim == 5, f"Expected 5D reference state for '{dataset_name}', got {ref.ndim}D."
            x_refs[dataset_name] = ref[:, -1]
        assert post_processors_tendencies is not None, "Per-step tendency processors must be provided."

        out_data = dict(out.data)
        for dataset_name, out_dataset in out_data.items():
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
            out_data[dataset_name] = out_dataset

        return out_data

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
