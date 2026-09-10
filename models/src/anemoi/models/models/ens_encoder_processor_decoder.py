# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from dataclasses import replace
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.data.batch import Batch
from anemoi.models.data.tensor_layout import TensorLayout
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.models.encoder_processor_decoder import EncoderSource
from anemoi.models.models.encoder_processor_decoder import latlons_to_sincos
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiEnsModelEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble functionality."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        model_graph_config: DotDict,
        data_indices: dict[str, IndexCollection],
        statistics: dict[str, dict],
        is_dataset_static: dict[str, bool],
        data_layouts: dict[str, TensorLayout],
        n_step_input: int,
        n_step_output: int,
    ) -> None:
        # Read before super().__init__, which calls _calculate_input_dim.
        self.condition_on_residual = DotDict(model_config).model.condition_on_residual
        super().__init__(
            model_config=model_config,
            model_graph_config=model_graph_config,
            data_indices=data_indices,
            statistics=statistics,
            is_dataset_static=is_dataset_static,
            data_layouts=data_layouts,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
        )

    def _build_networks(
        self,
        model_config: DotDict,
        static_graph: HeteroData,
        dynamic_graph_config: DotDict,
    ) -> None:
        super()._build_networks(model_config, static_graph, dynamic_graph_config)

        self.noise_injector = instantiate(
            model_config.noise_injector,
            _recursive_=False,
            graph_data=static_graph,
            sparse_projector_num_chunks=model_config.get("sparse_projector", {}).get("num_chunks", 1),
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        base_input_dim += 1  # for forecast step (fcstep)
        if self.condition_on_residual:
            base_input_dim += self.num_input_channels_prognostic[dataset_name]
        return base_input_dim

    def _condition_source(self, source: EncoderSource, fcstep: int) -> EncoderSource:
        """Append the ensemble conditioning channels to an assembled encoder source.

        We concat first the forecast step, then the residual's prognostic channels if
        `condition_on_residual` is set.
        """
        x_data_latent = source.x_data_latent

        extra = [
            torch.full(
                (x_data_latent.shape[0], 1),
                float(fcstep),
                device=x_data_latent.device,
                dtype=x_data_latent.dtype,
            )
        ]

        if self.condition_on_residual:
            extra.append(self._residual_conditioning(source.x_skip, source.dataset_name, x_data_latent))

        return replace(source, x_data_latent=torch.cat([x_data_latent, *extra], dim=-1))

    def _residual_conditioning(self, x_skip: Tensor | None, dataset_name: str, x_data_latent: Tensor) -> Tensor:
        """The residual's prognostic channels, one row per node, for input conditioning."""
        assert x_skip is not None, (
            f"condition_on_residual is set but dataset {dataset_name!r} has no residual; "
            "add one under model.residual.datasets, or turn the conditioning off."
        )
        assert x_skip.ndim == 5, (
            f"Residual conditioning needs a (batch, time, ensemble, grid, variables) residual "
            f"for dataset {dataset_name!r}, got shape {tuple(x_skip.shape)}."
        )
        prognostic = x_skip[:, 0][..., self._internal_input_idx[dataset_name]]
        rows = einops.rearrange(prognostic, "batch ensemble grid vars -> (batch ensemble grid) vars")
        assert rows.shape[0] == x_data_latent.shape[0], (
            f"Residual conditioning for dataset {dataset_name!r} has {rows.shape[0]} rows but the "
            f"input latent has {x_data_latent.shape[0]}; the residual is not on the same nodes."
        )
        return rows

    def forward(
        self,
        batch: Batch,
        target: Batch,
        *,
        fcstep: int = 0,
        model_comm_group: Optional[ProcessGroup] = None,
        **_kwargs,
    ) -> Batch:
        """Forward pass of the ensemble model.

        Parameters
        ----------
        batch : Batch
            Batch envelope, one source view per dataset.
        target : Batch
            Decoder conditioning: the forcing variables at the output valid times.
        fcstep : int, optional
            Forecast step to condition on, clamped to `min(1, fcstep)`.
        model_comm_group : Optional[ProcessGroup], optional
            Model communication group.

        Returns
        -------
        Batch
            Model output, built by updating the `target` for each decoded dataset.
        """
        dataset_names = list(batch.keys())

        batch_size = self._get_consistent_dim(batch, 0)
        ensemble_size = self._get_consistent_dim(batch, 2)

        batch_ens_size = batch_size * ensemble_size  # batch and ensemble dimensions are merged

        in_out_sharded = self._resolve_in_out_sharded(batch)
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        fcstep = min(1, fcstep)

        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}

        # TODO: revisit this (should not need an explicit move to device here)
        hidden_coordinates = self._hidden_coordinates().to(batch.device)
        hidden_coordinates_batched = einops.repeat(hidden_coordinates, "n f -> (repeat n) f", repeat=batch_ens_size)
        hidden_batch_sizes = (hidden_coordinates.shape[0],) * batch_ens_size
        x_hidden_latent = latlons_to_sincos(hidden_coordinates)
        x_hidden_latent = einops.repeat(x_hidden_latent, "n f -> (repeat n) f", repeat=batch_ens_size)

        hidden_trainable_parameters = self.node_attributes(self._graph_name_hidden, batch_size=batch_ens_size)
        if hidden_trainable_parameters is not None:
            x_hidden_latent = torch.cat([x_hidden_latent, hidden_trainable_parameters], dim=-1)

        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)

        # Encoders, in config order, over their source datasets in listed order
        for encoder_name, source_datasets in self.encoder2datasets.items():
            sources = []
            for dataset_name in source_datasets:
                if dataset_name not in batch:
                    continue

                source = self._prepare_encoder_source(
                    batch[dataset_name],
                    dataset_name=dataset_name,
                    batch_size=batch_ens_size,
                    hidden_coordinates=hidden_coordinates,
                    hidden_coordinates_batched=hidden_coordinates_batched,
                    hidden_batch_sizes=hidden_batch_sizes,
                    shard_sizes_hidden=shard_sizes_hidden,
                    model_comm_group=model_comm_group,
                )
                if source is None:  # no data points for this dataset in this batch
                    continue

                source = self._condition_source(source, fcstep)
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
                    batch_size=batch_ens_size,
                    model_comm_group=model_comm_group,
                )
            )

        x_latent = self.latent_aggregator(x_hidden_latent, dataset_latents)

        x_latent_noised, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=self._graph_data[self._graph_name_hidden].num_nodes,
            grid_shard_sizes=shard_sizes_hidden,
            model_comm_group=model_comm_group,
        )
        noise_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        processor_edge_attr, processor_edge_index, proc_edge_shard_sizes = self.processor_graph_provider.get_edges(
            src_coords=hidden_coordinates,
            dst_coords=hidden_coordinates,
            batch_size=batch_ens_size,
            model_comm_group=model_comm_group,
        )
        processor_edge_attr = processor_edge_attr.to(dtype=x_latent.dtype)

        x_latent_proc = self.processor(
            x=x_latent_noised,
            batch_size=batch_ens_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            **noise_kwargs,
        )

        # Latent skip connection
        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        # Decoder
        x_out_dict = {}
        for dataset_name in self.target_datasets:
            target_coords, target_data_latent, shard_sizes_data, data_batch_sizes, data_timedeltas = (
                self._assemble_target(
                    batch[dataset_name],
                    x_data_latent_dict.get(dataset_name, None),
                    target[dataset_name],
                    batch_size=batch_ens_size,
                    model_comm_group=model_comm_group,
                    dataset_name=dataset_name,
                )
            )

            if target_coords.numel() == 0:
                LOGGER.debug(
                    "No data points for dataset %s in the batch (data_coords.shape = %s), "
                    + "will decode to a size-zero tensor ...",
                    dataset_name,
                    list(target_coords.shape),
                )

            graph_batch_kwargs = (
                {"src_batch_sizes": hidden_batch_sizes, "dst_batch_sizes": data_batch_sizes}
                if data_batch_sizes is not None
                else {}
            )
            decoder_edge_attr, decoder_edge_index, dec_edge_shard_sizes = self.decoder_graph_provider[
                dataset_name
            ].get_edges(
                batch_size=batch_ens_size,
                src_coords=hidden_coordinates_batched if data_batch_sizes is not None else hidden_coordinates,
                dst_coords=target_coords,
                dst_timedeltas=data_timedeltas,
                model_comm_group=model_comm_group,
                **graph_batch_kwargs,
            )
            decoder_edge_attr = decoder_edge_attr.to(dtype=x_latent.dtype)

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_data,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            decoder_name = self.dataset2decoder[dataset_name]
            x_out = self.decoder[decoder_name](
                (x_latent_proc, target_data_latent),
                batch_size=batch_ens_size,
                shard_info=dec_shard_info,
                edge_attr=decoder_edge_attr,
                edge_index=decoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
            )

            x_out_dict[dataset_name] = self._assemble_output(
                x_out,
                x_skip_dict.get(dataset_name, None),
                target[dataset_name],
                dtype=x_out.dtype,
                dataset_name=dataset_name,
            )

        # Preserve the reconstructed output metadata rather than the decoder
        # conditioning metadata carried by target.
        output = target
        for dataset_name in x_out_dict.keys():
            do_coords_match = target[dataset_name].coordinates == x_out_dict[dataset_name].coordinates
            assert (
                do_coords_match if isinstance(do_coords_match, bool) else torch.all(do_coords_match)
            ), "Target and output coordinates must match."
            output = output.update_source(dataset_name, x_out_dict[dataset_name])

        return output
