# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

from anemoi.graphs.create import HeteroData
import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data.batch import Batch
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.models import BaseGraphModel
from anemoi.utils.config import DotDict
from torch.utils import data

LOGGER = logging.getLogger(__name__)


def latlons_to_sincos(latlon: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.sin(latlon), torch.cos(latlon)], dim=-1)



class AnemoiModelEncProcDec(BaseGraphModel):
    """Message passing graph neural network."""

    def _build_networks(self, model_config: DotDict, static_graph: HeteroData, dynamic_graph_config: DotDict) -> None:
        """Builds the model components."""
        # Encoder data -> hidden
        self.encoder_graph_provider = torch.nn.ModuleDict()
        self.encoder = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            bipartite_graph_name = (dataset_name, "to", self._graph_name_hidden)

            # Create graph providers
            self.encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=static_graph[bipartite_graph_name],
                edge_attribute_names=model_config.model.encoder.get("sub_graph_edge_attributes"),
                **dynamic_graph_config[bipartite_graph_name],
                src_size=self.node_attributes.num_nodes.get(dataset_name, None),
                dst_size=self.node_attributes.num_nodes.get(self._graph_name_hidden, None),
                trainable_size=model_config.model.encoder.get("trainable_size", 0),
            )

            self.encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.input_dim[dataset_name],
                in_channels_dst=self.input_dim_latent,
                hidden_dim=self.num_channels,
                edge_dim=self.encoder_graph_provider[dataset_name].edge_dim,
            )

        # Processor hidden -> hidden
        self.processor_graph_provider = create_graph_provider(
            graph=static_graph[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            edge_attribute_names=model_config.model.processor.get("sub_graph_edge_attributes"),
            src_size=self.node_attributes.num_nodes.get(self._graph_name_hidden, None),
            dst_size=self.node_attributes.num_nodes.get(self._graph_name_hidden, None),
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
        for dataset_name in self.dataset_names:
            bipartite_graph_name = (self._graph_name_hidden, "to", dataset_name)

            self.decoder_graph_provider[dataset_name] = create_graph_provider(
                graph=static_graph[bipartite_graph_name],
                edge_attribute_names=model_config.model.decoder.get("sub_graph_edge_attributes"),
                **dynamic_graph_config[bipartite_graph_name],
                src_size=self.node_attributes.num_nodes.get(self._graph_name_hidden, None),
                dst_size=self.node_attributes.num_nodes.get(dataset_name, None),
                trainable_size=model_config.model.decoder.get("trainable_size", 0),
            )

            self.decoder[dataset_name] = instantiate(
                model_config.model.decoder,
                _recursive_=False,  # Avoids instantiation of layer_kernels here
                in_channels_src=self.num_channels,
                in_channels_dst=self.target_dim[dataset_name],
                hidden_dim=self.num_channels,
                out_channels_dst=self.output_dim[dataset_name],
                edge_dim=self.decoder_graph_provider[dataset_name].edge_dim,
            )

    def _assemble_input(
        self,
        x: "DatasetView",
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        grid_shard_sizes = x.grid_shard_indices
        if grid_shard_sizes == slice(None):
            grid_shard_sizes = None

        if x.is_static:
            x_skip = self.residual[dataset_name](
                x.data,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                n_step_output=self.n_step_output,
            )
        else:
            x_skip = None

        inputs = []

        if x.is_static:
            input_coordinates = x.coordinates.to(x.data.device)
            inputs = [
                einops.rearrange(x.data, f"{x.layout.pattern} -> (batch ensemble grid) (time variables)"),
                einops.repeat(latlons_to_sincos(input_coordinates), "grid latlon -> (batch grid) latlon", batch=batch_size)
            ]

            trainable_parameters = self.node_attributes(dataset_name, batch_size=batch_size)
            if trainable_parameters is not None:
                trainable_parameters = trainable_parameters.to(x.data.device)
                if grid_shard_sizes is not None:
                    trainable_parameters = shard_tensor(trainable_parameters, 0, grid_shard_sizes, model_comm_group)
                
                inputs.append(trainable_parameters)
        else:
            if len(x.data) == 1:
                input_coordinates = x.coordinates[0].to(x.data[0].device)
                inputs = [x.data[0], latlons_to_sincos(input_coordinates)]
            else:
                input_coordinates = torch.cat(x.coordinates, dim=0).to(x.data[0].device)
                inputs = [torch.cat(x.data, dim=0), latlons_to_sincos(input_coordinates)]

        x_data_latent = torch.cat(inputs, dim=-1)  # feature dimension

        return input_coordinates, x_data_latent, x_skip, grid_shard_sizes
    
    def _assemble_target(
        self,
        x: "DatasetView",
        encoder_data_output: torch.Tensor | None,
        batch_size: int = 1,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ):
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."

        grid_shard_sizes = x.grid_shard_indices
        if grid_shard_sizes == slice(None):
            grid_shard_sizes = None

        if x.is_static:
            input_coordinates = x.coordinates.to(x.data.device)
        else:
            if len(x.data) == 1:
                input_coordinates = x.coordinates[0].to(x.data[0].device)
            else:
                input_coordinates = torch.cat(x.coordinates, dim=0).to(x.data[0].device)

        if self.use_encoder_data_output[dataset_name]:
            assert encoder_data_output is not None
            target_decoder_data = encoder_data_output
        else:
            target_decoder_data = latlons_to_sincos(input_coordinates)

            if not x.is_static:
                trainable_parameters = self.node_attributes(dataset_name, batch_size=batch_size)
                if trainable_parameters is not None:
                    trainable_parameters = trainable_parameters.to(x.data.device)
                    if grid_shard_sizes is not None:
                        trainable_parameters = shard_tensor(trainable_parameters, 0, grid_shard_sizes, model_comm_group)
                    
                    target_decoder_data = torch.cat([target_decoder_data, trainable_parameters], dim=-1)

        assert input_coordinates.shape[0] == target_decoder_data.shape[0], "Coordinate and data sizes must match."
        return input_coordinates, target_decoder_data, grid_shard_sizes

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor,
        target: "DatasetView",
        batch_size: int,
        ensemble_size: int,
        dtype: torch.dtype,
        dataset_name: str,
    ):
        if target.is_static:
            x_out = (
                einops.rearrange(
                    x_out,
                    f"(batch ensemble grid) (time variables) -> {target.layout.pattern}",
                    batch=batch_size,
                    ensemble=ensemble_size,
                    time=self.n_step_output,
                )
                .to(dtype=dtype)
                .clone()
            )
        else:
            x_out = x_out.to(dtype=dtype).clone()
            assert self.n_step_output == 1, "Only n_step_output=1 is supported for non-static targets."
            assert len(x_out.shape) == 2, "Expected x_out to have shape (grid, variables) for non-static target."

        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        if x_skip is not None:
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, variables)."
            assert (
                x_skip.shape[1] == x_out.shape[1]
            ), f"Residual time dimension ({x_skip.shape[1]}) must match output time dimension ({x_out.shape[1]})."
            x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)

        return x_out if target.is_static else [x_out]

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
        batch: Batch,
        target: Optional[Batch] = None,
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        batch : Batch
            Typed batch envelope. ``batch.data`` carries the per-dataset input
            tensors; ``batch.coordinates`` carries the per-dataset coordinate
            tensors used by dynamic graph providers / node attributes.
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
        dataset_names = list(batch.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(batch, 0)
        ensemble_size = self._get_consistent_dim(batch, 2)

        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}

        # Prepare hidden latent
        hidden_coordinates = self._hidden_coordinates()
        x_hidden_latent = latlons_to_sincos(hidden_coordinates)
        x_hidden_latent = einops.repeat(x_hidden_latent, "n f -> (repeat n) f", repeat=batch_size)

        hidden_trainable_parameters = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        if hidden_trainable_parameters is not None:
            hidden_trainable_parameters = hidden_trainable_parameters.to(x_hidden_latent.device)
            x_hidden_latent = torch.cat([x_hidden_latent, hidden_trainable_parameters], dim=-1)

        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        for dataset_name in dataset_names:
            data_coords, x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                batch[dataset_name],
                batch_size=batch_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            if data_coords.shape[0] == 0:
                continue

            x_skip_dict[dataset_name] = x_skip
            shard_sizes_data_dict[dataset_name] = shard_sizes_data

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                src_coords=data_coords,
                dst_coords=hidden_coordinates,
                model_comm_group=model_comm_group,
            )
            encoder_edge_attr = encoder_edge_attr.to(x_data_latent.device)
            encoder_edge_index = encoder_edge_index.to(x_data_latent.device)

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data,  # None if not sharded (in_out_sharded=False)
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
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

        # Combine all dataset latents
        x_latent = sum(dataset_latents.values())

        # Processor
        processor_edge_attr, processor_edge_index, proc_edge_shard_sizes = self.processor_graph_provider.get_edges(
            src_coords=hidden_coordinates,
            dst_coords=hidden_coordinates,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        processor_edge_attr = processor_edge_attr.to(x_latent.device)
        processor_edge_index = processor_edge_index.to(x_latent.device)

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
            x_latent = x_latent_proc + x_latent

        # Decoder
        x_out_dict = {}
        for dataset_name in dataset_names:
            data_coords, target_data_latent, shard_sizes_data = self._assemble_target(
                target[dataset_name],
                x_data_latent_dict.get(dataset_name, None),
                batch_size=batch_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )

            if data_coords.shape[0] == 0:
                continue

            # Compute decoder edges using updated latent representation
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                src_coords=hidden_coordinates,
                dst_coords=data_coords,
                model_comm_group=model_comm_group,
            )
            decoder_edge_attr = decoder_edge_attr.to(x_latent.device)
            decoder_edge_index = decoder_edge_index.to(x_latent.device)

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            x_out = self.decoder[dataset_name](
                (x_latent, target_data_latent),
                batch_size=batch_size,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                x_skip_dict[dataset_name],
                target[dataset_name],
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                dtype=x_out.dtype, 
                dataset_name=dataset_name,
            )

        
        return target.with_data(x_out_dict)

    def fill_metadata(self, md_dict) -> None:
        for dataset in self.input_dim.keys():
            shapes = {
                "variables": self.input_dim[dataset],
                "input_timesteps": self.n_step_input,
                "ensemble": 1,
                "grid": None,  # grid size is dynamic
            }
            md_dict["metadata_inference"][dataset]["shapes"] = shapes
