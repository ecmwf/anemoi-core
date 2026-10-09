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
from collections.abc import Sequence
from dataclasses import replace
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
from anemoi.models.data.flat import FlatSource
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
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
from anemoi.utils.config import DotDict

if TYPE_CHECKING:
    from anemoi.models.data.sources import BaseTemplate
    from anemoi.models.data.sources.base import Source

LOGGER = logging.getLogger(__name__)


def _join_blocks(
    first: torch.Tensor,
    first_sizes: Sequence[int],
    second: torch.Tensor,
    second_sizes: Sequence[int],
) -> torch.Tensor:
    """Join two sets of rows block by block: each ``(sample, member)`` block's rows of ``first``, then of ``second``."""
    blocks = []
    for first_block, second_block in zip(first.split(list(first_sizes)), second.split(list(second_sizes)), strict=True):
        blocks.extend((first_block, second_block))
    return torch.cat(blocks, dim=0)


def _template_to(template: "BaseTemplate", device: torch.device) -> "BaseTemplate":
    """Return ``template`` with its node tensors (coordinates, timedeltas) on ``device``."""

    def move(value):
        if isinstance(value, list):
            return [item.to(device) for item in value]
        return value.to(device)

    fields = {name: move(getattr(template, name)) for name in ("coordinates", "timedeltas") if hasattr(template, name)}
    return replace(template, **fields)


def _second_blocks(joined: torch.Tensor, first_sizes: Sequence[int], second_sizes: Sequence[int]) -> torch.Tensor:
    """Return the rows of ``second`` from rows joined by :func:`_join_blocks`."""
    sizes = [size for pair in zip(first_sizes, second_sizes, strict=True) for size in pair]
    return torch.cat(joined.split(sizes)[1::2], dim=0)


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

        transport_params = model_config.model.transport
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

        self._assert_no_boundings()

        self.noise_embedder = instantiate(transport_params.noise_embedder)
        self.noise_cond_mlp = self._create_noise_conditioning_mlp()

    def _assert_no_boundings(self) -> None:
        """Reject configured output bounds.

        The network outputs EDM's pre-combination field or the interpolant's drift, not the predicted
        state, so bounds on its output would constrain the wrong quantity.
        """
        bounded = [name for name, bounding in self.boundings.items() if len(bounding) > 0]
        if bounded:
            raise ValueError(f"Bounding is not supported for transport models, but it is configured for {bounded}.")

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        output_dim = super()._calculate_output_dim(dataset_name)
        input_dim = base_input_dim + output_dim  # input history plus corrupted target
        if not self.is_dataset_static.get(dataset_name, True):
            input_dim += 1  # flag: 1 on a noisy-target row, 0 on a history row
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

    def _assemble_transport_input(
        self,
        x: "Source",
        y_noised: "Source",
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, ShardSizes, tuple[int, ...] | None, torch.Tensor | None]:
        """Build the encoder rows of one dataset from its history ``x`` and its noisy target ``y_noised``.

        A gridded dataset has the same nodes at input and target time, so each row holds both:
        ``[history | noisy target | coordinates | ...]``. A tabular dataset has different nodes, so
        each ``(sample, member)`` block holds its history rows, ``[history | 0 | ... | flag 0]``,
        followed by its noisy-target rows, ``[0 | noisy target | ... | flag 1]``.
        """
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        x_flat = x.flatten()
        y_noised_flat = y_noised.flatten()
        if x.is_tabular:
            nodes, values, flag = self._history_and_noisy_target_rows(x_flat, y_noised_flat)
        else:
            if not torch.equal(x_flat.coordinates, y_noised_flat.coordinates):
                raise AssertionError("Input and conditioned target coordinates must match for gridded transport data.")
            nodes = x_flat
            values = torch.cat([x_flat.data.to(y_noised_flat.data.dtype), y_noised_flat.data], dim=-1)
            flag = None

        grid_shard_sizes = nodes.shard_sizes
        inputs = [values, latlons_to_sincos(nodes.coordinates)]
        dynamic_node_attributes = self._encode_dynamic_node_attributes(dataset_name, nodes)
        if dynamic_node_attributes is not None:
            inputs.append(dynamic_node_attributes.to(device=values.device, dtype=values.dtype))

        if dataset_name in self.node_attributes:
            node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size).to(values.device)
            # The attributes cover every node; a sharded dataset holds only this rank's share.
            num_nodes = sum(grid_shard_sizes) if grid_shard_sizes is not None else values.shape[0]
            if node_attributes_data.shape[0] != num_nodes:
                msg = (
                    "Trainable node attributes are not implemented for dynamic sparse transport nodes. "
                    f"Dataset '{dataset_name}' has {num_nodes} encoder nodes, "
                    f"but static node attributes provide {node_attributes_data.shape[0]} rows."
                )
                raise NotImplementedError(msg)
            if grid_shard_sizes is not None:
                node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)
            inputs.append(node_attributes_data)

        if flag is not None:
            inputs.append(flag)
        x_data_latent = torch.cat(inputs, dim=-1)

        # Gather the coordinates so the encoder graph is built on all nodes, as in the base model.
        batch_sizes = gathered_batch_sizes(nodes.batch_sizes, grid_shard_sizes)
        data_coords = nodes.coordinates
        timedeltas = nodes.timedeltas
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

    @staticmethod
    def _history_and_noisy_target_rows(
        x_flat: FlatSource, y_noised_flat: FlatSource
    ) -> tuple[FlatSource, torch.Tensor, torch.Tensor]:
        """Stack a tabular dataset's history nodes and noisy-target nodes, block by block.

        Returns the stacked nodes, their values ``[history | noisy target]`` with zeros in the
        part a row does not have, and the flag column (1 on noisy-target rows).
        """
        history = x_flat.data.to(y_noised_flat.data.dtype)
        noisy_target = y_noised_flat.data
        n_history, n_target = history.shape[0], noisy_target.shape[0]
        history_rows = torch.cat([history, history.new_zeros(n_history, noisy_target.shape[-1])], dim=-1)
        target_rows = torch.cat([noisy_target.new_zeros(n_target, history.shape[-1]), noisy_target], dim=-1)

        history_sizes, target_sizes = x_flat.batch_sizes, y_noised_flat.batch_sizes
        values = _join_blocks(history_rows, history_sizes, target_rows, target_sizes)
        flag = _join_blocks(history.new_zeros(n_history, 1), history_sizes, history.new_ones(n_target, 1), target_sizes)

        if (x_flat.shard_sizes is None) != (y_noised_flat.shard_sizes is None):
            raise ValueError("The history and the noisy target of a tabular dataset must be sharded alike.")
        shard_sizes = None
        if x_flat.shard_sizes is not None:
            shard_sizes = [a + b for a, b in zip(x_flat.shard_sizes, y_noised_flat.shard_sizes, strict=True)]

        nodes = FlatSource(
            data=None,
            coordinates=_join_blocks(x_flat.coordinates, history_sizes, y_noised_flat.coordinates, target_sizes),
            device=y_noised_flat.device,
            shard_sizes=shard_sizes,
            batch_sizes=tuple(a + b for a, b in zip(history_sizes, target_sizes, strict=True)),
            timedeltas=_join_blocks(x_flat.timedeltas, history_sizes, y_noised_flat.timedeltas, target_sizes),
        )
        return nodes, values, flag

    def _assemble_output(self, x_out: torch.Tensor, x_skip, target: "Source", dtype: torch.dtype, dataset_name: str):
        # The transport network predicts the conditioned target itself, so the output takes its shape and variables.
        del x_skip
        del dataset_name
        return target.template().unflatten(x_out.to(dtype=torch.promote_types(dtype, torch.float32)))

    def _make_noise_emb(self, noise_emb: torch.Tensor, repeat: int) -> torch.Tensor:
        assert noise_emb.ndim == 5, "noise_emb must be (batch, time, ensemble, 1, channels)."
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
        x: Batch,
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
            target_view = conditioned_target[dataset_name]
            c_target, c_hidden = self._view_conditioning(noise_cond, dataset_name, target_view, model_comm_group)
            c_hidden_shard_sizes = get_shard_sizes(c_hidden, 0, model_comm_group=model_comm_group)
            c_hidden = shard_tensor(c_hidden, 0, c_hidden_shard_sizes, model_comm_group)

            # Conditioning enters each mapper with the same node layout as its features: the encoder
            # of a tabular dataset sees its history nodes and its noisy-target nodes, block by block.
            c_encoder = c_target
            if dataset_name in x and x[dataset_name].is_tabular:
                history_view = x[dataset_name]
                c_history, _ = self._view_conditioning(noise_cond, dataset_name, history_view, model_comm_group)
                c_encoder = _join_blocks(
                    c_history,
                    history_view.template().flatten().batch_sizes,
                    c_target,
                    target_view.template().flatten().batch_sizes,
                )

            fwd_mapper_kwargs[dataset_name] = {"cond": (c_encoder, c_hidden)}
            bwd_mapper_kwargs[dataset_name] = {"cond": (c_hidden, c_target)}

        processor_kwargs = {"cond": c_hidden}
        return fwd_mapper_kwargs, processor_kwargs, bwd_mapper_kwargs

    def _view_conditioning(
        self,
        noise_cond: torch.Tensor,
        dataset_name: str,
        view: "Source",
        model_comm_group: Optional[ProcessGroup],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noise conditioning over the nodes of ``view`` (split like them) and over all hidden nodes."""
        shard_sizes = view.template().flatten().shard_sizes
        c_data, c_hidden, _, _, _ = self._generate_noise_conditioning(
            noise_cond,
            dataset_name=dataset_name,
            data_view=view,
            data_shard_sizes=shard_sizes,
            edge_conditioning=False,
        )
        if shard_sizes is not None:
            c_data = shard_tensor(c_data, 0, shard_sizes, model_comm_group)
        return c_data, c_hidden

    def forward(
        self,
        x: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        target_forcing: Optional[Batch] = None,
        **kwargs,
    ) -> Batch:
        return self.transport_model_objective.forward(
            self,
            x,
            conditioned_target,
            condition,
            model_comm_group=model_comm_group,
            target_forcing=target_forcing,
            **kwargs,
        )

    def _forward_transport_network(
        self,
        batch: Batch,
        conditioned_target: Batch,
        condition: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
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
            batch, conditioned_target, condition, model_comm_group=model_comm_group
        )

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict: dict[str, torch.Tensor | None] = {}
        x_data_latent_dict = {}

        hidden_coordinates = self._hidden_coordinates().to(batch.device)
        hidden_coordinates_batched = einops.repeat(hidden_coordinates, "n f -> (repeat n) f", repeat=bse)
        hidden_batch_sizes = (hidden_coordinates.shape[0],) * bse
        x_hidden_latent = latlons_to_sincos(hidden_coordinates)
        x_hidden_latent = einops.repeat(x_hidden_latent, "n f -> (repeat n) f", repeat=bse)
        hidden_trainable_parameters = self.node_attributes(self._graph_name_hidden, batch_size=bse)
        if hidden_trainable_parameters is not None:
            x_hidden_latent = torch.cat([x_hidden_latent, hidden_trainable_parameters], dim=-1)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group=model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        # Encoders run as in the deterministic model: in config order, each over its source
        # datasets, with the noisy target as part of each dataset's rows.
        for encoder_name, source_datasets in self.encoder2datasets.items():
            sources = []
            for dataset_name in source_datasets:
                if dataset_name not in batch:
                    continue

                assembled = self._assemble_transport_input(
                    batch[dataset_name],
                    conditioned_target[dataset_name],
                    batch_size=bse,
                    model_comm_group=model_comm_group,
                    dataset_name=dataset_name,
                )
                source = self._encoder_source_from_rows(
                    dataset_name,
                    assembled,
                    batch_size=bse,
                    hidden_coordinates=hidden_coordinates,
                    hidden_coordinates_batched=hidden_coordinates_batched,
                    hidden_batch_sizes=hidden_batch_sizes,
                    shard_sizes_hidden=shard_sizes_hidden,
                    model_comm_group=model_comm_group,
                )
                if source is None:  # no data points for this dataset in this batch
                    continue

                x_skip_dict[dataset_name] = source.x_skip
                sources.append(source)

            if not sources:
                continue

            dataset_latents.update(
                self._encode_sources(
                    encoder_name,
                    sources,
                    x_hidden_latent=x_hidden_latent,
                    x_data_latent_dict=x_data_latent_dict,
                    batch_size=bse,
                    model_comm_group=model_comm_group,
                    mapper_kwargs=fwd_mapper_kwargs,
                )
            )

        # Decoder features read the encoded rows of a tabular dataset at its target nodes only.
        for dataset_name, encoded in x_data_latent_dict.items():
            if batch[dataset_name].is_tabular:
                x_data_latent_dict[dataset_name] = _second_blocks(
                    encoded,
                    batch[dataset_name].template().flatten().batch_sizes,
                    conditioned_target[dataset_name].template().flatten().batch_sizes,
                )

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
        processor_edge_attr = processor_edge_attr.to(device=x_latent.device, dtype=x_latent.dtype)
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
                # The transport decoder also takes the noisy target at the target nodes.
                conditioned_target_data = conditioned_target[dataset_name].flatten().data
                conditioned_target_data = conditioned_target_data.to(
                    device=target_data_latent.device,
                    dtype=target_data_latent.dtype,
                )
                target_data_latent = torch.cat([conditioned_target_data, target_data_latent], dim=-1)

            x_out = self._decode_rows(
                dataset_name,
                x_latent_proc,
                (target_coords, target_data_latent, shard_sizes_target, target_batch_sizes, target_timedeltas),
                batch_size=bse,
                hidden_coordinates=hidden_coordinates,
                hidden_coordinates_batched=hidden_coordinates_batched,
                hidden_batch_sizes=hidden_batch_sizes,
                shard_sizes_hidden=shard_sizes_hidden,
                edge_dtype=x_latent.dtype,
                keep_x_dst_sharded=in_out_sharded[dataset_name],
                model_comm_group=model_comm_group,
                mapper_kwargs=bwd_mapper_kwargs[dataset_name],
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

    def _sampled_to_physical(
        self,
        sampled: Batch,
        x: Batch,
        post_processors: nn.ModuleDict,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> Batch:
        """Turn the sampled field into the prediction in physical units.

        Parameters
        ----------
        sampled : Batch
            Normalised sample in model-output variables.
        x : Batch
            Normalised inputs the sample was drawn from.
        post_processors : nn.ModuleDict
            Post-processing module.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        **kwargs
            Additional parameters for subclasses.

        Returns
        -------
        Batch
            The prediction, still split across ``model_comm_group`` like the sample.
        """
        del x, model_comm_group, kwargs
        return sampled.with_sources(
            {name: post_processors[name](source, in_place=False) for name, source in sampled.items()},
        )

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

    def _sampling_template(self, target_template: dict[str, "BaseTemplate"], x: Batch) -> Batch:
        """Describe the sampled field: the template's nodes, layout and sizes, with the model's output variables.

        Only decoded datasets are sampled. The payloads are zero-stride views in the output width;
        only their shape, dtype and device are read. They take the dtype and device of the matching
        input, and the template's nodes are moved to that device.
        """
        sources = {}
        for name, template in target_template.items():
            if name not in self.target_datasets:
                continue
            template = _template_to(
                template.with_variables(
                    self._sampling_variables(name, "output"),
                    self._sampling_statistics(name, "output"),
                ),
                x[name].device,
            )
            flat = template.flatten()
            payload = torch.zeros((), dtype=x[name].dtype, device=flat.device).expand(
                flat.coordinates.shape[0], self._calculate_output_dim(name)
            )
            sources[name] = template.unflatten(payload)
        return Batch(sources)

    #: Named in the error raised for an invalid transport source kind.
    _sampling_source_context = "state prediction"

    def build_sampling_source(
        self,
        x: Batch,
        *,
        target_template: dict[str, "BaseTemplate"],
        model_comm_group: Optional[ProcessGroup] = None,
        default_kind: str = "gaussian",
    ) -> Batch:
        """Build the starting/source field used by transport sampling, in model-output space.

        Gaussian noise or zeros, or the latest input state projected to the predicted variables.
        """
        request = TransportSourceRequest(
            templates=self._sampling_template(target_template, x),
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
        x: Batch,
        target_template: dict[str, "BaseTemplate"],
        pre_processors: nn.ModuleDict,
        post_processors: nn.ModuleDict,
        n_step_input: dict[str, int],
        target_forcing: Optional[Batch] = None,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        spatial_pre_processors: Optional[nn.ModuleDict] = None,
        schedule_params: Optional[dict] = None,
        sampler_params: Optional[dict] = None,
        statistics_tendencies: Optional[dict[str, Mapping]] = None,
        **kwargs,
    ) -> Batch:
        """Run inference by sampling from the selected transport objective.

        Parameters
        ----------
        x : Batch
            Input batched data (before pre-processing), holding the input time steps.
        target_template : dict[str, BaseTemplate]
            What to sample, one template per decoded dataset: the output nodes, layout and time steps.
        pre_processors : nn.ModuleDict
            Pre-processing module.
        post_processors : nn.ModuleDict
            Post-processing module.
        n_step_input : dict[str, int]
            Number of input time steps for each dataset; ``x`` already holds exactly these.
        target_forcing : Optional[Batch]
            Decoder conditioning (before pre-processing): the forcing variables at the output valid times.
        model_comm_group : Optional[ProcessGroup]
            Process group for distributed training.
        gather_out : bool
            Whether to gather the output across the model group.
        spatial_pre_processors : Optional[nn.ModuleDict]
            Spatial preprocessors keyed by dataset name (e.g. CrossGridProjector).
            Applied after grid sharding and before normalisation, as in training.
        schedule_params : Optional[dict]
            Sampling schedule parameters (schedule_type, num_steps, ...), overriding inference_defaults.
        sampler_params : Optional[dict]
            Sampler parameters (sampler, S_churn, S_min, S_max, S_noise, ...), overriding inference_defaults.
        statistics_tendencies : Optional[dict[str, Mapping]]
            Per-dataset tendency statistics, used by tendency models to convert
            the sampled tendencies into states.
        **kwargs
            Additional sampling parameters.

        Returns
        -------
        Batch
            Sampled prediction (after post-processing), on the nodes of ``target_template``.
        """
        del n_step_input
        with torch.no_grad():
            x, target_forcing = self._prepare_prediction_inputs(
                x,
                target_forcing,
                pre_processors,
                model_comm_group=model_comm_group,
                spatial_pre_processors=spatial_pre_processors,
            )

            sampled = self.sample(
                x,
                target_template=target_template,
                model_comm_group=model_comm_group,
                schedule_params=schedule_params,
                sampler_params=sampler_params,
                target_forcing=target_forcing,
                **kwargs,
            )
            y_hat = self._sampled_to_physical(
                sampled,
                x,
                post_processors,
                model_comm_group=model_comm_group,
                statistics_tendencies=statistics_tendencies,
            )

            if gather_out:
                y_hat = y_hat.with_sources(
                    {name: source.allgather(model_comm_group) for name, source in y_hat.items()},
                )

        return y_hat

    def sample(
        self,
        x: Batch,
        *,
        target_template: dict[str, "BaseTemplate"],
        model_comm_group: Optional[ProcessGroup] = None,
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

        self.condition_on_residual = model_config.condition_on_residual
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

    def _assemble_transport_input(
        self,
        x: "Source",
        y_noised: "Source",
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, ShardSizes, tuple[int, ...] | None, torch.Tensor | None
    ]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        if x.is_tabular or y_noised.is_tabular:
            raise NotImplementedError("Tendency transport is not implemented for sparse observation datasets.")

        (
            data_coords,
            x_data_latent,
            _x_skip,
            shard_sizes_data,
            batch_sizes,
            timedeltas,
        ) = super()._assemble_transport_input(
            x,
            y_noised,
            batch_size,
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
        ``"lead_times"``. Each mapping has per-variable arrays over the dataset's full variable set.
        """
        n_step_output = self.n_step_output[dataset_name]
        if statistics_tendencies is None:
            raise ValueError(f"Tendency statistics are required for dataset '{dataset_name}'.")
        if "lead_times" not in statistics_tendencies:
            raise ValueError(
                f"Tendency statistics of dataset '{dataset_name}' must be listed per lead time ('lead_times')."
            )

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

    def _sampled_to_physical(
        self,
        sampled: Batch,
        x: Batch,
        post_processors: nn.ModuleDict,
        model_comm_group: Optional[ProcessGroup] = None,
        statistics_tendencies: Optional[dict[str, Mapping]] = None,
        **kwargs,
    ) -> Batch:
        """Turn sampled tendencies into states in physical units, added to the latest input state."""
        del kwargs
        if statistics_tendencies is None:
            raise ValueError("Tendency statistics must be provided to convert sampled tendencies into states.")

        references = self.reference_state(
            x,
            {name: source.grid_shard_sizes for name, source in x.items()},
            model_comm_group,
        )
        return sampled.with_sources(
            {
                dataset_name: self.add_tendency_to_state(
                    dataset_name,
                    references[dataset_name],
                    tendency,
                    self.tendency_statistics(dataset_name, statistics_tendencies.get(dataset_name)),
                    post_processors[dataset_name],
                )
                for dataset_name, tendency in sampled.items()
            },
        )

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
