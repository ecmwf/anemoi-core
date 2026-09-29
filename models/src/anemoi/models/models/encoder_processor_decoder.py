# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.processor import NoOpProcessor
from anemoi.models.models import BaseGraphModel
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(BaseGraphModel):
    """Message passing graph neural network."""

    def _build_networks(self, model_config: DotDict) -> None:
        """Builds the model components."""
        # Encoder data -> hidden
        # Per-dataset encoder overrides may be supplied via `model.encoders.<dataset_name>`;
        # otherwise the shared `model.encoder` block is used for every dataset.
        self.encoder_graph_provider = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.input_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the input as it doesn't have a corresponding encoder."
                )
                continue

            encoder_config = model_config.encoders[self.dataset2encoder[dataset_name]]

            # Create graph providers
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(dataset_name, "to", self._graph_name_hidden)],
                edge_attributes=encoder_config.mapper.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[dataset_name],
                dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                trainable_size=encoder_config.mapper.get("trainable_size", 0),
            )

        self.encoder = torch.nn.ModuleDict()
        for encoder_name, encoder_config in model_config.encoders.items():
            encoder_in_channels_src = [self.input_dim[d] for d in self.encoder2datasets[encoder_name]]
            assert all(ch == encoder_in_channels_src[0] for ch in encoder_in_channels_src), (
                f"All datasets for encoder {encoder_name} must have the same input dimension, "
                f"but got {encoder_in_channels_src}."
            )

            self.encoder[encoder_name] = instantiate(
                encoder_config.mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=encoder_in_channels_src[0],
                in_channels_dst=self.input_dim_latent,
                edge_dim=self.encoder_graph_provider[encoder_config.source_datasets[0]].edge_dim,
            )

        # Latent aggregator: combines encoder outputs before the processor
        # (e.g. SumAggregator, or GatedFusionAggregator for gated latent fusion).
        self._build_latent_aggregator(model_config.latent_aggregator)

        # Processor hidden -> hidden
        self.processor_graph_provider = create_graph_provider(
            graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            edge_attributes=model_config.processor.get("sub_graph_edge_attributes"),
            src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            trainable_size=model_config.processor.get("trainable_size", 0),
        )

        self.processor = instantiate(
            model_config.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            edge_dim=self.processor_graph_provider.edge_dim,
        )

        # Principal dataset is defined in config as model.principal_dataset.
        # Dropout probabilities are handled in anemoi-training and only a list of
        # dropped dataset names is passed into forward.
        self.principal_dataset_name = model_config.model.get("principal_dataset", self.dataset_names[0])
        assert self.principal_dataset_name in self.dataset_names, (
            "Configured principal dataset given in config file '%s' not in dataset_names=%s."
            % (self.principal_dataset_name, self.dataset_names)
        )
        assert (
            isinstance(self.processor, NoOpProcessor)
            or self.processor.num_channels == self.latent_aggregator.hidden_dim
        ), (
            f"Processor number of channels ({self.processor.num_channels}) must match latent aggregator output channels"
            f" ({self.latent_aggregator.hidden_dim})."
        )

        # Decoder hidden -> data
        # Per-dataset decoder overrides may be supplied via `model.decoders.<dataset_name>`;
        # otherwise the shared `model.decoder` block is used for every dataset.
        self.decoder_graph_provider = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            if dataset_name not in self.target_datasets:
                LOGGER.info(
                    f"Dataset {dataset_name} is not part of the output as it doesn't have a corresponding decoder."
                )
                continue

            decoder_config = model_config.decoders[self.dataset2decoder[dataset_name]]
            self.decoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(self._graph_name_hidden, "to", dataset_name)],
                edge_attributes=decoder_config.mapper.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                dst_size=self.node_attributes.num_nodes[dataset_name],
                trainable_size=decoder_config.mapper.get("trainable_size", 0),
            )

        self.decoder = torch.nn.ModuleDict()
        for decoder_name, decoder_config in model_config.decoders.items():
            decoder_in_channels_dst = [self.target_dim[d] for d in self.decoder2datasets[decoder_name]]
            assert all(ch == decoder_in_channels_dst[0] for ch in decoder_in_channels_dst), (
                f"All datasets for decoder {decoder_name} must have the same target dimension, "
                f"but got {decoder_in_channels_dst}."
            )
            decoder_output_channels_dst = [self.output_dim[d] for d in self.decoder2datasets[decoder_name]]
            assert all(ch == decoder_output_channels_dst[0] for ch in decoder_output_channels_dst), (
                f"All datasets for decoder {decoder_name} must have the same output dimension, "
                f"but got {decoder_output_channels_dst}."
            )

            self.decoder[decoder_name] = instantiate(
                decoder_config.mapper,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.latent_aggregator.hidden_dim,
                in_channels_dst=decoder_in_channels_dst[0],
                out_channels_dst=decoder_output_channels_dst[0],
                edge_dim=self.decoder_graph_provider[decoder_config.target_datasets[0]].edge_dim,
            )

        self._log_encoder_decoder_param_counts()

    def _log_encoder_decoder_param_counts(self) -> None:
        """Log trainable parameter counts for each per-dataset encoder and decoder."""

        def _count(mod: nn.Module) -> int:
            return sum(p.numel() for p in mod.parameters() if p.requires_grad)

        def _fmt(n: int) -> str:
            return f"{n:,} ({n / 1e6:.2f}M)"

        LOGGER.info("Trainable parameters per encoder/decoder:")
        enc_total = 0
        for name, mod in self.encoder.items():
            n = _count(mod)
            enc_total += n
            LOGGER.info(f"  encoder[{name}]: {_fmt(n)}")
        LOGGER.info(f"  encoder total : {_fmt(enc_total)}")

        dec_total = 0
        for name, mod in self.decoder.items():
            n = _count(mod)
            dec_total += n
            LOGGER.info(f"  decoder[{name}]: {_fmt(n)}")
        LOGGER.info(f"  decoder total : {_fmt(dec_total)}")

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ShardSizes]:
        """Assemble the encoder source features and the residual skip for a single dataset.

        Flattens the raw input over ``(batch, ensemble, grid)`` and ``(time, vars)``, concatenates
        the per-node attributes on the feature dimension, and computes the residual skip tensor.

        Returns
        -------
        tuple[Tensor, Tensor, ShardSizes]
            ``(x_data_latent, x_skip, grid_shard_sizes)`` where ``x_data_latent`` is the encoder
            source input, and ``x_skip`` is the residual to add to the decoder output.
        """
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        x_skip = self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )

        return x_data_latent, x_skip, grid_shard_sizes

    def _assemble_targets(
        self,
        x_input_data: Tensor,
        x_encoded_data: Tensor | None,
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[Tensor, ShardSizes]:
        """Assemble the decoder destination features for a single dataset.

        Concatenates the feature blocks listed in ``decoders_target_input`` for this dataset's
        decoder into the per-node vector fed to the decoder as ``x_dst``.

        Returns
        -------
        tuple[Tensor, ShardSizes]
            ``(x_target_latent, grid_shard_sizes)`` where ``x_target_latent`` has width
            ``target_dim[dataset_name]``.
        """
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        x_target_latent = self.decoders_target_input[self.dataset2decoder[dataset_name]].tensor(
            x_input_data,
            x_encoded_data,
            batch_size=batch_size,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            dataset_name=dataset_name,
        )

        return x_target_latent, grid_shard_sizes

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor | None,
        batch_size: int,
        ensemble_size: int,
        dtype: torch.dtype,
        dataset_name: str,
    ):
        """Reshape decoder output, add the prognostic residual and apply output boundings.

        Rearranges the flat decoder output back to ``(batch, time, ensemble, grid, vars)``, adds
        ``x_skip`` on the prognostic channels, and applies the per-dataset ``boundings`` in
        config order.
        """
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
                time=self.n_step_output,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        if x_skip is not None:
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            assert (
                x_skip.shape[1] == x_out.shape[1]
            ), f"Residual time dimension ({x_skip.shape[1]}) must match output time dimension ({x_out.shape[1]})."
            x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def _resolve_dropped_datasets(
        self,
        dropped_dataset_names: list[str] | set[str] | None,
        decoder_dropped_dataset_names: list[str] | set[str] | None,
    ) -> tuple[set[str], set[str]]:
        """Normalise the dataset-dropout selection passed in by the training step.

        The principal dataset is never dropped, and a fully-dropped dataset is
        removed from the decoder-only set (its decoder output is masked anyway).
        """
        primary_dataset = self.principal_dataset_name

        dropped = set() if dropped_dataset_names is None else set(dropped_dataset_names)
        dropped.discard(primary_dataset)
        decoder_dropped = set() if decoder_dropped_dataset_names is None else set(decoder_dropped_dataset_names)
        decoder_dropped.discard(primary_dataset)
        decoder_dropped -= dropped
        return dropped, decoder_dropped

    def _assert_valid_sharding(
        self,
        batch_size: int,
        ensemble_size: int,
        in_out_sharded: bool,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> None:
        assert not (
            in_out_sharded and model_comm_group is None
        ), "If input is sharded, model_comm_group must be provided."

        if model_comm_group is not None:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded across GPUs"

            assert (
                model_comm_group.size() == 1 or ensemble_size == 1
            ), "Ensemble size per device must be 1 when model is sharded across GPUs"

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        dropped_dataset_names: list[str] | set[str] | None = None,
        decoder_dropped_dataset_names: list[str] | set[str] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        dataset_names = list(x.keys())

        # Dataset dropout selection is passed in from the training step.
        dropped_dataset_names, decoder_dropped_dataset_names = self._resolve_dropped_datasets(
            dropped_dataset_names,
            decoder_dropped_dataset_names,
        )

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Process each active dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        for dataset_name in x.keys():
            if dataset_name not in self.input_datasets:
                continue

            x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                x[dataset_name],
                batch_size=batch_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_sizes_data_dict[dataset_name] = shard_sizes_data

            (
                encoder_edge_attr,
                encoder_edge_index,
                enc_edge_shard_sizes,
            ) = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data,  # None if not sharded (in_out_sharded=False)
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset
            encoder_name = self.dataset2encoder[dataset_name]
            x_data_latent, x_latent = self.encoder[encoder_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_size,
                shard_info=enc_shard_info,
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )
            x_data_latent_dict[dataset_name] = x_data_latent
            dataset_latents[dataset_name] = x_latent

        # Combine encoder latents. Dropped datasets are still passed in so the
        # aggregator can mask them while keeping the computation graph static.
        if dropped_dataset_names and not self.training:
            LOGGER.info(f"Evaluation with dropped datasets: {dropped_dataset_names}")
        x_latent = self.latent_aggregator(x_hidden_latent, dataset_latents, dropped_sources=dropped_dataset_names)

        # Processor
        (
            processor_edge_attr,
            processor_edge_index,
            proc_edge_shard_sizes,
        ) = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
        )

        # Latent skip connection
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        # Decoder — always run all decoders (static graph for DDP).
        # For dropped datasets, decoder still runs but output is replaced with input
        # so the loss sees zero tendency → no gradient signal for that dataset.
        x_out_dict = {}
        for dataset_name in self.target_datasets:
            x_target_latent, shard_sizes_target = self._assemble_targets(
                x[dataset_name],
                x_data_latent_dict.get(dataset_name, None),
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )

            # Compute decoder edges using updated latent representation
            (
                decoder_edge_attr,
                decoder_edge_index,
                dec_edge_shard_sizes,
            ) = self.decoder_graph_provider[
                dataset_name
            ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_target,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            decoder_name = self.dataset2decoder[dataset_name]
            x_out = self.decoder[decoder_name](
                (x_latent_proc, x_target_latent),
                batch_size=batch_size,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                x_skip_dict.get(dataset_name, None),
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                dtype=x[dataset_name].dtype,
                dataset_name=dataset_name,
            )

            if dataset_name in dropped_dataset_names or dataset_name in decoder_dropped_dataset_names:
                # Replace output with NaN — NaN-aware loss will ignore this dataset.
                # Multiply by 0 and add NaN to keep decoder params in the graph.
                # print(f"[decoder-dropout] masking {dataset_name} "
                #     f"(fully_dropped={dataset_name in dropped_dataset_names}, "
                #     f"decoder_only={dataset_name in decoder_dropped_dataset_names})", flush=True)
                x_out_dict[dataset_name] = x_out_dict[dataset_name] * 0 + float("nan")

        return x_out_dict

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.n_step_input,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes
