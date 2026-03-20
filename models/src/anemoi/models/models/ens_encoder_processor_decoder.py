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
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
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
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=batch_ens_size)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        x_skip = self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

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
            x_skip_cond = x_skip[:, 0] if x_skip.ndim == 5 else x_skip
            x_data_latent = torch.cat(
                (
                    x_data_latent,
                    einops.rearrange(x_skip_cond, "bse grid vars -> (bse grid) vars"),
                ),
                dim=-1,
            )

        return x_data_latent, x_skip, grid_shard_sizes

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor,
        batch_size: int,
        batch_ens_size: int,
        dtype: torch.dtype,
        dataset_name: str | None = None,
    ):
        ensemble_size = batch_ens_size // batch_size
        x_out = (
            einops.rearrange(
                x_out,
                "(bs e n) (time vars) -> bs time e n vars",
                bs=batch_size,
                e=ensemble_size,
                time=self.n_step_output,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
        assert (
            x_skip.shape[1] == x_out.shape[1]
        ), f"Residual time dimension ({x_skip.shape[1]}) must match output time dimension ({x_out.shape[1]})."
        x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        self,
        x: dict[str, torch.Tensor],
        *,
        fcstep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward operator.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input tensor, shape (bs, m, e, n, f)
        fcstep : int
            Forecast step
        model_comm_group : ProcessGroup, optional
            Model communication group
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Tensor]
            Output tensor per dataset
        """
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        batch_ens_size = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        fcstep = min(1, fcstep)
        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}
        shard_sizes_hidden_dict = {}

        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                x[dataset_name],
                fcstep=fcstep,
                batch_ens_size=batch_ens_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_sizes_data_dict[dataset_name] = shard_sizes_data

            x_hidden_latent = self.node_attributes[dataset_name](self._graph_name_hidden, batch_size=batch_ens_size)
            shard_sizes_hidden_dict[dataset_name] = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
            x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden_dict[dataset_name], model_comm_group)

            encoder_edge_attr, encoder_edge_index, enc_edge_shard_sizes = self.encoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                dst_nodes=shard_sizes_hidden_dict[dataset_name],
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset
            x_data_latent, x_latent = self.encoder[dataset_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_ens_size,
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

        shard_sizes_hidden = shard_sizes_hidden_dict[dataset_names[0]]
        assert all(
            shard_size == shard_sizes_hidden for shard_size in shard_sizes_hidden_dict.values()
        ), "All datasets must have the same shard sizes for the hidden graph."

        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=self.node_attributes[dataset_names[0]].num_nodes[self._graph_name_hidden],
            grid_shard_sizes=shard_sizes_hidden,
            model_comm_group=model_comm_group,
        )

        processor_edge_attr, processor_edge_index, proc_edge_shard_sizes = self.processor_graph_provider.get_edges(
            batch_size=batch_ens_size,
            model_comm_group=model_comm_group,
        )
        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        # Processor
        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=batch_ens_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )

        x_latent_proc = x_latent_proc + x_latent

        x_out_dict = {}
        for dataset_name in dataset_names:
            # Compute decoder edges using updated latent representation
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            x_out = self.decoder[dataset_name](
                (x_latent_proc, x_data_latent_dict[dataset_name]),
                batch_size=batch_ens_size,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
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
