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

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph_providers import create_graph_provider
from anemoi.models.models import BaseGraphModel
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(BaseGraphModel):
    """Message passing graph neural network."""

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

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_shapes: dict | None,
        model_comm_group=None,
        dataset_name: str = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=batch_size)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

        x_skip = self.residual[dataset_name](x, grid_shard_shapes=grid_shard_shapes, model_comm_group=model_comm_group)

        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = get_or_apply_shard_shapes(
            x_data_latent, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
        )

        return x_data_latent, x_skip, shard_shapes_data

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor,
        batch_size: int,
        ensemble_size: int,
        dtype: torch.dtype,
        dataset_name: str,
    ):
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

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
        grid_shard_shapes: dict[str, list] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : dict[str, Tensor]
            Input data
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group, by default None
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None

        Returns
        -------
        dict[str, Tensor]
            Output of the model, with the same shape as the input (sharded if input is sharded)
        """
        dataset_names = list(x.keys())

        # Extract and validate batch sizes across datasets
        batch_sizes = [x[dataset_name].shape[0] for dataset_name in dataset_names]
        ensemble_sizes = [x[dataset_name].shape[2] for dataset_name in dataset_names]

        # Assert all datasets have the same batch and ensemble sizes
        assert all(
            bs == batch_sizes[0] for bs in batch_sizes
        ), f"Batch sizes must be the same across datasets: {batch_sizes}"
        assert all(
            es == ensemble_sizes[0] for es in ensemble_sizes
        ), f"Ensemble sizes must be the same across datasets: {ensemble_sizes}"

        batch_size = batch_sizes[0]
        ensemble_size = ensemble_sizes[0]
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}
        shard_shapes_hidden_dict = {}

        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
                x[dataset_name],
                batch_size=batch_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            x_data_latent_dict[dataset_name] = x_data_latent
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            x_hidden_latent = self.node_attributes[dataset_name](self._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden_dict[dataset_name] = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )

            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data, shard_shapes_hidden),
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                x_dst_is_sharded=False,  # x_latent does not come sharded
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                edge_shard_shapes=enc_edge_shard_shapes,
            )
            dataset_latents[dataset_name] = x_latent

        # Combine all dataset latents
        x_latent = sum(dataset_latents.values())

        # Processor
        shard_shapes_hidden = shard_shapes_hidden_dict[dataset_names[0]]
        assert all(
            shard_shape == shard_shapes_hidden for shard_shape in shard_shapes_hidden_dict.values()
        ), "All datasets must have the same shard shapes for the hidden graph."

        processor_edge_attr, processor_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = self.processor(
            x=x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
        )

        # Skip
        x_latent_proc = x_latent_proc + x_latent

        # Compute decoder edges using updated latent representation
        decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )

        # Decoder
        x_out_dict = {}
        for dataset_name in dataset_names:
            x_out = self.decoder[dataset_name](
                (x_latent_proc, x_data_latent_dict[dataset_name]),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data_dict[dataset_name]),
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,  # x_latent always comes sharded
                x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
                keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
                edge_shard_shapes=dec_edge_shard_shapes,
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out, x_skip_dict[dataset_name], batch_size, ensemble_size, x[dataset_name].dtype, dataset_name
            )

        return x_out_dict

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
