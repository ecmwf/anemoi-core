# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Optional

import einops
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.encoder_processor_decoder import gathered_batch_sizes
from anemoi.models.models.encoder_processor_decoder import latlons_to_sincos
from anemoi.models.transport import EdmSettings
from anemoi.models.transport import NoiseConditioningSettings
from anemoi.models.transport import StochasticInterpolantSettings
from anemoi.models.transport import TransportSourceBuilder
from anemoi.models.transport import TransportSourceRequest
from anemoi.models.transport import get_transport_model_objective
from anemoi.models.transport import reference_state_sampling_source
from anemoi.models.transport.sources import Data
from anemoi.utils.config import DotDict

if TYPE_CHECKING:
    from anemoi.models.data.sources.base import Source

LOGGER = logging.getLogger(__name__)

SamplingData = tuple[Batch, ...]


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
        n_step_input: dict[str, int],
        n_step_output: dict[str, int],
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
        if "encoded_data" not in self._decoder_target_feature_names(dataset_name):
            # In addition to the deterministic decoder inputs (output-time
            # target features), the transport decoder
            # consumes the corrupted target values at the target locations.
            target_dim += super()._calculate_output_dim(dataset_name)
        return target_dim

    def _decoder_target_feature_names(self, dataset_name: str) -> set[str]:
        decoder_name = self.dataset2decoder[dataset_name]
        return {feature.name for feature in self.decoders_target_input[decoder_name].features}

    def _create_noise_conditioning_mlp(self) -> nn.Sequential:
        mlp = nn.Sequential()
        mlp.add_module("linear1_no_gradscaling", nn.Linear(self.noise_channels, self.noise_channels))
        mlp.add_module("activation", nn.SiLU())
        mlp.add_module("linear2_no_gradscaling", nn.Linear(self.noise_channels, self.noise_cond_dim))
        return mlp

    def _assemble_input(
        self,
        x: "Source",
        y_noised: "Source",
        bse: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, ShardSizes, tuple[int, ...] | None, torch.Tensor | None]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        x_features = x.flatten()
        y_noised_features = y_noised.flatten()
        same_coordinates = torch.equal(x_features.coordinates, y_noised_features.coordinates)
        if same_coordinates:
            grid_shard_sizes = x_features.shard_sizes
            data_coords = x_features.coordinates
            x_input_features = x_features.data
        else:
            if not (x.is_tabular and y_noised.is_tabular):
                raise AssertionError("Input and conditioned target coordinates must match for dense transport data.")
            # The encoded rows are the conditioned target's nodes, and they are sharded like the target
            grid_shard_sizes = y_noised_features.shard_sizes
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
            # The attributes cover every node; a sharded target holds only this rank's share.
            num_target_nodes = (
                sum(grid_shard_sizes) if grid_shard_sizes is not None else y_noised_features.data.shape[0]
            )
            if node_attributes_data.shape[0] != num_target_nodes:
                msg = (
                    "Trainable node attributes are not implemented for dynamic sparse transport nodes. "
                    f"Dataset '{dataset_name}' has {num_target_nodes} target nodes, "
                    f"but static node attributes provide {node_attributes_data.shape[0]} rows."
                )
                raise NotImplementedError(msg)
            if grid_shard_sizes is not None:
                node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)
            inputs.append(node_attributes_data)

        x_data_latent = torch.cat(inputs, dim=-1)

        # Gather the coordinates so the encoder graph is built on all nodes, as in the base model.
        batch_sizes = gathered_batch_sizes(y_noised_features.batch_sizes, grid_shard_sizes)
        timedeltas = y_noised_features.timedeltas
        if grid_shard_sizes is not None:
            data_coords = gather_tensor(data_coords, dim=0, sizes=grid_shard_sizes, mgroup=model_comm_group)
            if timedeltas is not None:
                timedeltas = gather_tensor(timedeltas, dim=0, sizes=grid_shard_sizes, mgroup=model_comm_group)

        return (
            data_coords,
            x_data_latent,
            None,
            grid_shard_sizes,
            batch_sizes,
            timedeltas,
        )

    def _assemble_output(self, x_out: torch.Tensor, x_skip, target: "Source", dtype: torch.dtype, dataset_name: str):
        # The transport network predicts the conditioned target itself, so the output takes its shape and variables.
        del x_skip
        pred = target.template().unflatten(x_out.to(dtype=torch.promote_types(dtype, torch.float32)))
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

    def _make_noise_emb_for_view(self, noise_emb: torch.Tensor, view: "Source") -> torch.Tensor:
        """Repeat noise embeddings over the actual flattened nodes in a source view."""
        if not view.is_tabular:
            grid_size = view.data.shape[view.layout.axis("grid", ndim=view.data.ndim)]
            return self._make_noise_emb(noise_emb, repeat=grid_size)

        noise_base = noise_emb[:, 0, :, 0, :]
        if noise_base.shape[1] != view.ensemble_size:
            raise ValueError("Sparse transport noise embeddings must match the source view's ensemble size.")
        chunks = []
        for sample_index, sample in enumerate(view.data):
            num_nodes = sample.shape[view.layout.axis("grid", ndim=sample.ndim)]
            # Flatten in the same (sample, member, node) order as TabularSource.
            sample_noise = noise_base[sample_index, :, None, :].expand(-1, num_nodes, -1)
            chunks.append(sample_noise.reshape(-1, noise_base.shape[-1]).to(sample.device))
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
        data_view: Optional["Source"] = None,
        data_shard_sizes: ShardSizes = None,
        edge_conditioning: bool = False,
    ) -> torch.Tensor:

        if data_view is None:
            c_data = self._make_noise_emb(noise_cond, repeat=self._graph_data[dataset_name].num_nodes)
        elif data_shard_sizes is not None and len(data_shard_sizes) > 1:
            # Model sharding permits one sample/member, so all nodes share one embedding.
            c_data = self._make_noise_emb(noise_cond, repeat=sum(data_shard_sizes))
        else:
            c_data = self._make_noise_emb_for_view(noise_cond, data_view)
        c_hidden = self._make_noise_emb(noise_cond, repeat=self._graph_data[self._graph_name_hidden].num_nodes)

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
            data_view = conditioned_target[dataset_name]
            data_shard_sizes = data_view.template().flatten().shard_sizes
            c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(
                noise_cond,
                dataset_name=dataset_name,
                data_view=data_view,
                data_shard_sizes=data_shard_sizes,
                edge_conditioning=False,
            )
            c_hidden_shard_sizes = get_shard_sizes(c_hidden, 0, model_comm_group=model_comm_group)
            c_hidden = shard_tensor(c_hidden, 0, c_hidden_shard_sizes, model_comm_group)

            # Conditioning enters each mapper with the same node layout as its features.
            if data_shard_sizes is not None:
                c_data = shard_tensor(c_data, 0, data_shard_sizes, model_comm_group)

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
        batch_size = batch.batch_size
        ensemble_size = batch.ensemble_size

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
        hidden_coordinates_batched = einops.repeat(hidden_coordinates, "n f -> (repeat n) f", repeat=bse)
        hidden_batch_sizes = (hidden_coordinates.shape[0],) * bse
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
            if dataset_name not in self.input_datasets:
                continue

            data_coords, x_data_latent, x_skip, shard_sizes_data, data_batch_sizes, data_timedeltas = (
                self._assemble_input(
                    batch[dataset_name],
                    conditioned_target[dataset_name],
                    bse,
                    grid_shard_sizes,
                    model_comm_group,
                    dataset_name,
                )
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
                dst_coords=hidden_coordinates_batched if data_batch_sizes is not None else hidden_coordinates,
                src_timedeltas=data_timedeltas,
                model_comm_group=model_comm_group,
                **(
                    {"src_batch_sizes": data_batch_sizes, "dst_batch_sizes": hidden_batch_sizes}
                    if data_batch_sizes is not None
                    else {}
                ),
            )
            encoder_edge_attr = encoder_edge_attr.to(x_data_latent.device)  # todo SL: remove device movement
            encoder_edge_index = encoder_edge_index.to(x_data_latent.device)

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset
            encoder_name = self.dataset2encoder[dataset_name]
            x_data_latent, dataset_latents[dataset_name] = self.encoder[encoder_name](
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

        # Combine all dataset latents
        x_latent = self.latent_aggregator(x_hidden_latent, dataset_latents)

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
        for dataset_name in self.target_datasets:
            target_feature_names = self._decoder_target_feature_names(dataset_name)
            decoder_target_view = conditioned_target[dataset_name]
            if "target_forcings" in target_feature_names:
                if target_forcing is None or dataset_name not in target_forcing:
                    msg = (
                        f"Dataset '{dataset_name}' configures the 'target_forcings' decoder feature; "
                        "a 'target_forcing' batch with the output-time forcings is required."
                    )
                    raise ValueError(msg)
                decoder_target_view = target_forcing[dataset_name]

            target_coords, target_data_latent, shard_sizes_target, target_batch_sizes, target_timedeltas = (
                self._assemble_target(
                    batch[dataset_name],
                    x_data_latent_dict.get(dataset_name),
                    decoder_target_view,
                    conditioned_target[dataset_name],
                    batch_size=bse,
                    model_comm_group=model_comm_group,
                    dataset_name=dataset_name,
                )
            )
            if "encoded_data" not in target_feature_names:
                conditioned_target_data = conditioned_target[dataset_name].flatten().data
                conditioned_target_data = conditioned_target_data.to(
                    device=target_data_latent.device,
                    dtype=target_data_latent.dtype,
                )
                target_data_latent = torch.cat([conditioned_target_data, target_data_latent], dim=-1)
            # Compute decoder edges using updated latent representation
            (
                decoder_edge_attr,
                decoder_edge_index,
                dec_edge_shard_sizes,
            ) = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=bse,
                src_coords=hidden_coordinates_batched if target_batch_sizes is not None else hidden_coordinates,
                dst_coords=target_coords,
                dst_timedeltas=target_timedeltas,
                model_comm_group=model_comm_group,
                **(
                    {"src_batch_sizes": hidden_batch_sizes, "dst_batch_sizes": target_batch_sizes}
                    if target_batch_sizes is not None
                    else {}
                ),
            )
            decoder_edge_attr = decoder_edge_attr.to(x_latent.device)  # todo SL: remove device movement
            decoder_edge_index = decoder_edge_index.to(x_latent.device)

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_target,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            decoder_name = self.dataset2decoder[dataset_name]
            x_out = self.decoder[decoder_name](
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
            out_view = self._assemble_output(
                x_out,
                x_skip_dict.get(dataset_name),
                target_view,
                target_view.dtype,
                dataset_name,
            )
            out_batch = out_batch.replace(dataset_name, out_view)

        return out_batch

    def _before_sampling(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        n_step_input: dict[str, int],
        model_comm_group: Optional[ProcessGroup] = None,
        spatial_pre_processors: Optional[nn.ModuleDict] = None,
        **kwargs,
    ) -> tuple[SamplingData, DatasetShardSizes | None]:
        """Prepare batch before sampling.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input batch after pre-processing.
        pre_processors : dict[str, nn.Module]
            Pre-processing module (already applied).
        n_step_input : dict[str, int]
            Number of input timesteps per node for each dataset.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        spatial_pre_processors : Optional[nn.ModuleDict]
            Spatial preprocessors keyed by dataset name (e.g. CrossGridProjector).
            Applied after grid sharding but before normalisation.
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
            x = x[:, 0 : n_step_input[dataset_name], None, ...]  # add dummy ensemble dimension as 3rd index

            if model_comm_group is not None:
                shard_sizes = get_shard_sizes(x, -2, model_comm_group=model_comm_group)
                assert grid_shard_sizes is not None
                grid_shard_sizes[dataset_name] = shard_sizes
                x = shard_tensor(x, -2, shard_sizes, model_comm_group)

            # Spatial preprocessing: applied after grid sharding, before normalisation.
            (x,), grid_shard_sizes = self._apply_spatial_preprocessor(
                (x,),
                dataset_name,
                spatial_pre_processors,
                model_comm_group,
                grid_shard_sizes,
            )

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
                if processed.is_tabular:
                    raise NotImplementedError(
                        "Distributed gather is not supported for sparse transport sampling outputs."
                    )
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
            raise ValueError(f"Unknown sampling variable space {variable_space!r}; expected 'input' or 'output'.")
        return list(indices.ordered_names)

    def _sampling_statistics(self, dataset_name: str, variable_space: str) -> dict:
        variables = self._sampling_variables(dataset_name, variable_space)
        name_to_index = self.data_indices[dataset_name].name_to_index
        positions = [name_to_index[name] for name in variables]
        return {key: values[positions] for key, values in self.statistics[dataset_name].items()}

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
        if template is not None and dataset_name in template and template[dataset_name].coordinates is not None:
            coordinates = template[dataset_name].coordinates
            dataset_grid_shard_sizes = grid_shard_sizes.get(dataset_name) if grid_shard_sizes is not None else None
            if dataset_grid_shard_sizes is None:
                return coordinates
            if template[dataset_name].is_tabular:
                raise NotImplementedError("Grid sharding is not supported for sparse transport sampling templates.")

            coordinates = coordinates.to(dataset_data.device)
            data_grid_size = dataset_data.shape[layout.axis("grid", ndim=dataset_data.ndim)]
            if coordinates.ndim == 2:
                coordinate_grid_dim = 0
            elif coordinates.ndim == 3:
                coordinate_grid_dim = 1
            else:
                raise ValueError(
                    "Sampling template coordinates must have shape (grid, 2) or (batch, grid, 2), "
                    f"got {tuple(coordinates.shape)} for dataset '{dataset_name}'."
                )

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

        # Without a template the payload is all there is: a list means per-sample (tabular) data.
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

        if dataset_name not in self._graph_data.node_types or "x" not in self._graph_data[dataset_name]:
            msg = (
                f"Cannot infer sampling coordinates for dataset '{dataset_name}' from the graph. "
                "Pass a Batch with coordinates instead."
            )
            raise ValueError(msg)

        # Only gridded payloads reach this point
        coordinates = self._graph_data[dataset_name].x.to(dataset_data.device)
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
        # Without a template the sampling inputs are the raw gridded tensors of predict_step,
        # which _before_sampling lays out as (batch, time, ensemble, grid, variables).
        gridded_layout = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
        source_layouts = (
            {name: template[name].layout for name in template.dataset_names}
            if template is not None
            else {name: gridded_layout for name in data}
        )

        sources = {}
        for dataset_name, dataset_data in data.items():
            layout = source_layouts[dataset_name]
            template_source = template[dataset_name] if template is not None and dataset_name in template else None

            common = {
                "name": dataset_name,
                "variables": self._sampling_variables(dataset_name, variable_space),
                "layout": layout,
                "statistics": self._sampling_statistics(dataset_name, variable_space),
                "data": dataset_data,
                "coordinates": self._sampling_coordinates(
                    dataset_name,
                    dataset_data,
                    layout=layout,
                    template=template,
                    model_comm_group=model_comm_group,
                    grid_shard_sizes=grid_shard_sizes,
                ),
                "shard_sizes": None if grid_shard_sizes is None else grid_shard_sizes.get(dataset_name),
            }
            # Only a tabular template can supply the timedeltas and boundaries a tabular source needs.
            if isinstance(template_source, TabularSource):
                sources[dataset_name] = TabularSource(
                    **common,
                    timedeltas=template_source.timedeltas,
                    boundaries=template_source.boundaries,
                )
            else:
                sources[dataset_name] = GriddedSource(**common)

        return Batch(sources)

    def _sampling_template(
        self,
        target_template: Batch,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
    ) -> Batch:
        """Describe the sampled field: the target template's geometry in model-output variables.

        The payloads are zero-stride views in the output width (nothing is allocated); only their
        shape, dtype and device are read. Variables are the last payload axis.
        """
        payloads = {
            name: template.map_data(
                lambda data, n_out=self.num_output_channels[name]: data.new_zeros(()).expand(*data.shape[:-1], n_out),
                variables=self._sampling_variables(name, "output"),
                statistics=self._sampling_statistics(name, "output"),
            ).data
            for name, template in target_template.items()
        }
        return self._make_sampling_batch(
            payloads,
            variable_space="output",
            template=target_template,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )

    #: Named in the error raised for an invalid transport source kind.
    _sampling_source_context = "state prediction"

    def build_sampling_source(
        self,
        x: Batch,
        *,
        target_template: Batch,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        default_kind: str = "gaussian",
    ) -> Batch:
        """Build the starting/source field used by transport sampling, in model-output space.

        Gaussian noise or zeros, or the latest input state projected to the predicted variables.
        """
        request = TransportSourceRequest(
            templates=self._sampling_template(
                target_template,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
            ),
            default_kind=default_kind,
            custom_source_factories={
                "reference_state": lambda: reference_state_sampling_source(
                    {name: source.data for name, source in x.items()},
                    data_indices=self.data_indices,
                    n_step_output=self.n_step_output,
                ),
            },
            model_comm_group=model_comm_group,
            error_context=self._sampling_source_context,
        )
        return self.transport_source.build(request)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        post_processors: dict[str, nn.Module],
        n_step_input: dict[str, int],
        target_template: Batch,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        statistics_tendencies: Optional[dict[str, Mapping]] = None,
        target_forcing: Optional[Batch] = None,
        spatial_pre_processors: Optional[nn.ModuleDict] = None,
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
        n_step_input : dict[str, int]
            Number of input timesteps to embed per node for each dataset.
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
        statistics_tendencies : Optional[dict[str, Mapping]]
            Per-dataset tendency statistics, used by tendency models to convert
            the sampled tendencies into states.
        target_forcing : Optional[Batch]
            Output-time decoding forcings (raw values) at the target locations.
            Normalized here with the input pre-processors and used to condition
            the decoder. Required for datasets that decode from target-side
            features (e.g. observations).
        spatial_pre_processors : Optional[nn.ModuleDict]
            Spatial preprocessors keyed by dataset name (e.g. CrossGridProjector).
            Applied after grid sharding but before normalisation.
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
                spatial_pre_processors=spatial_pre_processors,
                **kwargs,
            )

            x = before_sampling_data[0]

            if target_forcing is not None:
                # Normalize the output-time decoding forcings like the model inputs.
                for dataset_name in list(target_forcing.keys()):
                    if dataset_name in pre_processors:
                        target_forcing = target_forcing.replace(
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
            out = out.with_sources(
                {
                    dataset_name: source.map_data(lambda data, dtype=batch[dataset_name].dtype: data.to(dtype))
                    for dataset_name, source in out.items()
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
                statistics_tendencies=statistics_tendencies,
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
                "input_timesteps": self.n_step_input[dataset],
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes


class AnemoiTransportTendModelEncProcDec(AnemoiTransportModelEncProcDec):
    """Transport model that predicts tendencies and converts them back to state fields."""

    _sampling_source_context = "tendency prediction"

    def __init__(
        self,
        *,
        model_config: DictConfig,
        model_graph_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        is_dataset_static: dict[str, bool],
        n_step_input: dict[str, int],
        n_step_output: dict[str, int],
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
            input_dim += len(self.data_indices[dataset_name].model.input.prognostic) * self.n_step_output[dataset_name]
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
        x: "Source",
        y_noised: "Source",
        bse: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, ShardSizes, tuple[int, ...] | None, torch.Tensor | None
    ]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        if x.is_tabular or y_noised.is_tabular:
            raise NotImplementedError("Tendency transport is not implemented for sparse observation datasets.")

        data_coords, x_data_latent, _x_skip, shard_sizes_data, batch_sizes, timedeltas = super()._assemble_input(
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
                n_step_output=self.n_step_output[dataset_name],
            )[..., self._internal_input_idx[dataset_name]]
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            x_skip = einops.rearrange(x_skip, "batch time ensemble grid vars -> (batch ensemble) grid (time vars)")
            x_data_latent = torch.cat(
                (x_data_latent, einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars")), dim=-1
            )

        return data_coords, x_data_latent, x_skip, shard_sizes_data, batch_sizes, timedeltas

    def tendency_statistics(self, dataset_name: str, statistics_tendencies: Optional[Mapping]) -> list[Mapping]:
        """Return the tendency statistics of each output step of ``dataset_name``, in lead-time order.

        ``statistics_tendencies`` holds one statistics mapping per lead time, listed in order under
        ``"lead_times"``. A single-output model may instead pass one flat statistics mapping.
        Each mapping has per-variable arrays over the dataset's full variable set.
        """
        n_step_output = self.n_step_output[dataset_name]
        if statistics_tendencies is None:
            raise ValueError(f"Tendency statistics are required for dataset '{dataset_name}'.")
        if "lead_times" not in statistics_tendencies:
            if n_step_output != 1:
                raise ValueError(
                    f"Dataset '{dataset_name}' predicts {n_step_output} output steps and needs tendency "
                    "statistics per lead time ('lead_times')."
                )
            return [statistics_tendencies]

        lead_times = list(statistics_tendencies["lead_times"])
        if len(lead_times) != n_step_output:
            raise ValueError(
                f"Expected {n_step_output} tendency statistics entries for dataset '{dataset_name}', "
                f"got {len(lead_times)} ({lead_times})."
            )
        missing = [lead_time for lead_time in lead_times if statistics_tendencies.get(lead_time) is None]
        if missing:
            raise ValueError(f"Missing tendency statistics for dataset '{dataset_name}', lead times {missing}.")
        return [statistics_tendencies[lead_time] for lead_time in lead_times]

    def _as_tendency(self, dataset_name: str, source: "Source", tendency_statistics: Mapping) -> "Source":
        """Return ``source`` with the tendency statistics on its prognostic variables.

        The other variables (diagnostics) keep their state statistics: the model predicts them as states.
        Statistic keys without a tendency counterpart are dropped, so a processor that needs them fails
        instead of normalising a tendency with state statistics.
        """
        data_indices = self.data_indices[dataset_name]
        prognostic = set(data_indices.prognostic)
        positions = [position for position, name in enumerate(source.variables) if name in prognostic]
        full_positions = [data_indices.name_to_index[source.variables[position]] for position in positions]

        statistics = {}
        for key, values in source.statistics.items():
            if key not in tendency_statistics:
                continue
            mixed = np.array(values, copy=True)
            mixed[positions] = np.asarray(tendency_statistics[key])[full_positions]
            statistics[key] = mixed
        return source.clone(statistics=statistics)

    def _prognostic_positions(self, dataset_name: str, state: "Source", reference: "Source") -> tuple[list, list]:
        """Positions of the prognostic variables of ``state`` in ``state`` and in ``reference``."""
        prognostic = set(self.data_indices[dataset_name].prognostic)
        names = [name for name in state.variables if name in prognostic]
        missing = [name for name in names if name not in reference.name_to_index]
        if missing:
            raise ValueError(
                f"The tendency reference of dataset '{dataset_name}' lacks prognostic variables {missing}."
            )
        return [state.name_to_index[name] for name in names], [reference.name_to_index[name] for name in names]

    def reference_state(
        self,
        x: Batch,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: Optional[ProcessGroup],
    ) -> Batch:
        """Return the tendency reference: the latest input state, truncated like the residual.

        ``x`` holds the normalised model inputs. Each returned source is ``(batch, 1, ensemble, grid,
        variables)`` over the prognostic input variables, with their statistics.
        """
        truncated = self.apply_reference_state_truncation(
            {name: source.data for name, source in x.items()},
            grid_shard_sizes,
            model_comm_group,
        )
        return x.with_sources(
            {
                name: source.clone(
                    data=truncated[name][:, -1:],
                    **source._select_variable_metadata(self.data_indices[name].model.input.prognostic),
                )
                for name, source in x.items()
            },
        )

    def compute_tendency(
        self,
        dataset_name: str,
        state: "Source",
        reference: "Source",
        tendency_statistics: list[Mapping],
        pre_processors: nn.Module,
        post_processors: nn.Module,
        skip_imputation: bool = True,
    ) -> "Source":
        """Turn normalised output states into normalised tendencies.

        Each output step is un-normalised, the reference is subtracted from its prognostic variables,
        and the result is normalised again as a source carrying that step's tendency statistics
        (:meth:`_as_tendency`). Diagnostic variables stay normalised states.

        Parameters
        ----------
        dataset_name : str
            Dataset of ``state``.
        state : Source
            Normalised states, ``(batch, time, ensemble, grid, variables)`` with one time step per output step.
        reference : Source
            Normalised reference state ``(batch, 1, ensemble, grid, variables)``, see :meth:`reference_state`.
        tendency_statistics : list[Mapping]
            Tendency statistics of each output step, see :meth:`tendency_statistics`.
        pre_processors, post_processors : nn.Module
            The dataset's state processors.
        skip_imputation : bool, optional
            Skip imputation in the processors, by default True.

        Returns
        -------
        Source
            Normalised tendencies with the variables of ``state``. Its statistics are the state
            statistics, because each step is normalised with its own tendency statistics; convert back
            with :meth:`add_tendency_to_state`, not with the post-processors.
        """
        if state.time_size != len(tendency_statistics):
            raise ValueError(
                f"Dataset '{dataset_name}' has {state.time_size} output steps but "
                f"{len(tendency_statistics)} tendency statistics."
            )
        state_positions, reference_positions = self._prognostic_positions(dataset_name, state, reference)
        prognostic = torch.as_tensor(state_positions, dtype=torch.long, device=state.device)

        state_physical = post_processors(state, in_place=False, skip_imputation=skip_imputation)
        reference_physical = post_processors(reference, in_place=False, skip_imputation=skip_imputation)
        reference_prognostic = reference_physical.data[..., reference_positions]

        steps = []
        for step, statistics in enumerate(tendency_statistics):
            step_state = state_physical.select(time=[step])
            step_data = step_state.data.index_add(
                -1, prognostic, reference_prognostic.to(step_state.data.dtype), alpha=-1
            )
            tendency = step_state.clone(data=step_data)
            steps.append(
                pre_processors(
                    self._as_tendency(dataset_name, tendency, statistics),
                    in_place=True,
                    skip_imputation=skip_imputation,
                ).data,
            )
        return state.clone(data=torch.cat(steps, dim=1))

    def add_tendency_to_state(
        self,
        dataset_name: str,
        reference: "Source",
        tendency: "Source",
        tendency_statistics: list[Mapping],
        post_processors: nn.Module,
        pre_processors: Optional[nn.Module] = None,
        skip_imputation: bool = True,
    ) -> "Source":
        """Turn normalised tendencies back into states; the inverse of :meth:`compute_tendency`.

        Parameters
        ----------
        dataset_name : str
            Dataset of ``tendency``.
        reference : Source
            Normalised reference state ``(batch, 1, ensemble, grid, variables)``, see :meth:`reference_state`.
        tendency : Source
            Normalised tendencies, one time step per output step, carrying the state statistics of its variables.
        tendency_statistics : list[Mapping]
            Tendency statistics of each output step, see :meth:`tendency_statistics`.
        post_processors : nn.Module
            The dataset's state post-processors.
        pre_processors : nn.Module, optional
            The dataset's state pre-processors. When given, the states are normalised again;
            otherwise they are returned un-normalised.
        skip_imputation : bool, optional
            Skip imputation in the processors, by default True.

        Returns
        -------
        Source
            States with the variables and statistics of ``tendency``.
        """
        if tendency.time_size != len(tendency_statistics):
            raise ValueError(
                f"Dataset '{dataset_name}' has {tendency.time_size} output steps but "
                f"{len(tendency_statistics)} tendency statistics."
            )
        tendency_positions, reference_positions = self._prognostic_positions(dataset_name, tendency, reference)
        prognostic = torch.as_tensor(tendency_positions, dtype=torch.long, device=tendency.device)
        reference_physical = post_processors(reference, in_place=False, skip_imputation=skip_imputation)
        reference_prognostic = reference_physical.data[..., reference_positions]

        steps = []
        for step, statistics in enumerate(tendency_statistics):
            step_tendency = self._as_tendency(dataset_name, tendency.select(time=[step]), statistics)
            physical = post_processors(step_tendency, in_place=False, skip_imputation=skip_imputation).data
            steps.append(physical.index_add(-1, prognostic, reference_prognostic.to(physical.dtype)))

        state = tendency.clone(data=torch.cat(steps, dim=1))
        if pre_processors is not None:
            state = pre_processors(state, in_place=True, skip_imputation=skip_imputation)
        return state

    def _before_sampling(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        n_step_input: dict[str, int],
        model_comm_group: Optional[ProcessGroup] = None,
        spatial_pre_processors: Optional[nn.ModuleDict] = None,
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
            x_in = x[:, 0 : n_step_input[dataset_name], None, ...]  # add dummy ensemble dimension as 3rd index
            x_t0 = x[:, -1:, None, ...]  # keep time dim and add dummy ensemble dimension

            if model_comm_group is not None:
                shard_sizes = get_shard_sizes(x_in, -2, model_comm_group=model_comm_group)
                assert grid_shard_sizes is not None
                grid_shard_sizes[dataset_name] = shard_sizes
                x_in = shard_tensor(x_in, -2, shard_sizes, model_comm_group)
                shard_sizes = get_shard_sizes(x_t0, -2, model_comm_group=model_comm_group)
                x_t0 = shard_tensor(x_t0, -2, shard_sizes, model_comm_group)

            # Spatial preprocessing: applied after grid sharding, before normalisation.
            (x_in, x_t0), grid_shard_sizes = self._apply_spatial_preprocessor(
                (x_in, x_t0),
                dataset_name,
                spatial_pre_processors,
                model_comm_group,
                grid_shard_sizes,
            )

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

    def _after_sampling(
        self,
        out: Batch,
        post_processors: dict[str, nn.Module],
        before_sampling_data: SamplingData,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        gather_out: bool = True,
        statistics_tendencies: Optional[dict[str, Mapping]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Convert sampled tendencies into (un-normalised) state predictions."""
        if isinstance(before_sampling_data, tuple) and len(before_sampling_data) >= 2:
            x_t0s = before_sampling_data[1]
        else:
            raise ValueError("Expected before_sampling_data to contain x_t0s")
        if statistics_tendencies is None:
            raise ValueError("Tendency statistics must be provided to convert sampled tendencies into states.")

        references = self.reference_state(x_t0s, grid_shard_sizes, model_comm_group)
        out_data = {}
        for dataset_name, tendency in out.items():
            state = self.add_tendency_to_state(
                dataset_name,
                references[dataset_name],
                tendency,
                self.tendency_statistics(dataset_name, statistics_tendencies.get(dataset_name)),
                post_processors[dataset_name],
            )
            out_dataset = self._apply_imputer_inverse(post_processors, dataset_name, state.data)
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
                in_x, grid_shard_sizes_i, model_comm_group, n_step_output=self.n_step_output[dataset_name]
            )
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            # Keep only prognostic input variables, matching the tendency reference state.
            x_skips[dataset_name] = x_skip[..., self.data_indices[dataset_name].model.input.prognostic]

        return x_skips
