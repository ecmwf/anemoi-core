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
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiEnsModelEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble functionality."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        self.condition_on_residual = DotDict(model_config).model.condition_on_residual
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

    def _build_networks(self, model_config):
        super()._build_networks(model_config)
        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        base_input_dim += 1  # for forecast step (fcstep)
        if self.condition_on_residual:
            base_input_dim += self.num_input_channels_prognostic[dataset_name]
        return base_input_dim

    def _assemble_input(
        self,
        x: torch.Tensor,
        fcstep: int,
        batch_ens_size: int,
        grid_shard_shapes: dict = None,
        model_comm_group=None,
        dataset_name: str = None,
    ):
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=batch_ens_size)
        grid_shard_shapes = grid_shard_shapes[dataset_name]

        x_skip = self.residual[dataset_name](x, grid_shard_shapes=grid_shard_shapes, model_comm_group=model_comm_group)

        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
                torch.ones(batch_ens_size * x.shape[3], device=x.device).unsqueeze(-1) * fcstep,
            ),
            dim=-1,  # feature dimension
        )

        if self.condition_on_residual:
            x_data_latent = torch.cat(
                (
                    x_data_latent,
                    einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars"),
                ),
                dim=-1,
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
        batch_ens_size: int,
        dtype: torch.dtype,
        dataset_name: str = None,
    ):
        ensemble_size = batch_ens_size // batch_size
        x_out = (
            einops.rearrange(x_out, "(bs e n) f -> bs e n f", bs=batch_size, e=ensemble_size).to(dtype=dtype).clone()
        )

        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        self,
        x: torch.Tensor,
        *,
        fcstep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward operator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (bs, m, e, n, f)
        fcstep : int
            Forecast step
        model_comm_group : ProcessGroup, optional
            Model communication group
        grid_shard_shapes : list, optional
            Shard shapes of the grid, by default None
        **kwargs
            Additional keyword arguments

        Returns
        -------
        torch.Tensor
            Output tensor
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
        batch_ens_size = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = grid_shard_shapes is not None
        self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded, model_comm_group)

        fcstep = min(1, fcstep)
        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_shapes_data_dict = {}
        shard_shapes_hidden_dict = {}

        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_shapes_data = self._assemble_input(
                x[dataset_name],
                fcstep=fcstep,
                batch_ens_size=batch_ens_size,
                grid_shard_shapes=grid_shard_shapes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            x_data_latent_dict[dataset_name] = x_data_latent
            shard_shapes_data_dict[dataset_name] = shard_shapes_data

            x_hidden_latent = self.node_attributes[dataset_name](self._graph_name_hidden, batch_size=batch_ens_size)
            shard_shapes_hidden_dict[dataset_name] = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_shapes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_ens_size,
                shard_shapes=(shard_shapes_data_dict[dataset_name], shard_shapes_hidden_dict[dataset_name]),
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

        shard_shapes_hidden = shard_shapes_hidden_dict[dataset_names[0]]
        assert all(
            shard_shape == shard_shapes_hidden for shard_shape in shard_shapes_hidden_dict.values()
        ), "All datasets must have the same shard shapes for the hidden graph."

        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=self.node_attributes[dataset_names[0]].num_nodes[self._graph_name_hidden],
            shard_shapes_ref=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        processor_edge_attr, processor_edge_index, proc_edge_shard_shapes = self.processor_graph_provider.get_edges(
            batch_size=batch_ens_size,
            model_comm_group=model_comm_group,
        )

        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        # Processor
        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=batch_ens_size,
            shard_shapes=shard_shapes_hidden,
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            edge_shard_shapes=proc_edge_shard_shapes,
            **processor_kwargs,
        )

        x_latent_proc = x_latent_proc + x_latent

        x_out_dict = {}
        for dataset_name in dataset_names:
            # Compute decoder edges using updated latent representation
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_shapes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            x_out = self.decoder[dataset_name](
                (x_latent_proc, x_data_latent_dict[dataset_name]),
                batch_size=batch_ens_size,
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
                x_out,
                x_skip_dict[dataset_name],
                batch_size,
                batch_ens_size,
                dtype=x[dataset_name].dtype,
                dataset_name=dataset_name,
            )

        return x_out_dict
