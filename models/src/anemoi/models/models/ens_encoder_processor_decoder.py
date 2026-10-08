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
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_tensor
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

    # Model-input positions passed on to the encoder when the input noise consumes some
    # channels. A class-level default keeps models pickled before it existed loadable.
    _encoder_input_keep_idx: Optional[torch.Tensor] = None

    def __init__(
        self,
        *,
        model_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        n_step_input: int,
        n_step_output: int,
    ) -> None:
        self.condition_on_residual = DotDict(model_config).condition_on_residual
        self._input_noise_config = model_config.get("input_noise", None)
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
        )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        # The noise widens the encoder input, so it has to exist before input dims are computed.
        self._build_input_noise()
        super()._calculate_shapes_and_indices(data_indices)

    def _build_input_noise(self) -> None:
        """Instantiate the FourCastNet 3 style input perturbation, if configured."""
        self.input_noise = None
        self.input_noise_dataset = None
        self._encoder_input_keep_idx = None
        if self._input_noise_config is None:
            return

        dataset_name = self._input_noise_config.get("dataset", None)
        if dataset_name is None:
            if len(self.input_datasets) != 1:
                raise ValueError(
                    "model.input_noise.dataset must be set when the model has more than one "
                    f"input dataset (got {self.input_datasets})."
                )
            dataset_name = self.input_datasets[0]
        elif dataset_name not in self.input_datasets:
            raise ValueError(
                f"model.input_noise.dataset '{dataset_name}' is not an input dataset {self.input_datasets}."
            )

        name_to_index = self.data_indices[dataset_name].model.input.name_to_index
        self.input_noise_dataset = dataset_name
        # Statistics are addressed by data-input position, as for the output boundings.
        self.input_noise = instantiate(
            self._input_noise_config,
            _recursive_=False,
            dataset=dataset_name,
            num_time_steps=self.n_step_input,
            num_grid_points=self.node_attributes.num_nodes[dataset_name],
            name_to_index=name_to_index,
            statistics=self.statistics[dataset_name] if self.statistics is not None else None,
            name_to_index_stats=self.data_indices[dataset_name].data.input.name_to_index,
        )

        # Channels the noise reads (e.g. spread fields) shape the perturbation only; they
        # are not encoder features. The data indices keep the full layout, since rollout,
        # normalisation and inference all address the model input by those positions.
        consumed = set(self.input_noise.consumed_input_idx)
        if consumed:
            self._encoder_input_keep_idx = torch.as_tensor(
                [index for index in range(len(name_to_index)) if index not in consumed], dtype=torch.long
            )
            LOGGER.info(
                "Input noise reads %d '%s' input channels, which are dropped from the encoder input.",
                len(consumed),
                dataset_name,
            )

    def _build_networks(self, model_config: DotDict) -> None:
        super()._build_networks(model_config)

        self.noise_injector = instantiate(
            model_config.noise_injector,
            _recursive_=False,
            graph_data=self._graph_data,
            sparse_projector_num_chunks=model_config.get("sparse_projector", {}).get("num_chunks", 1),
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        base_input_dim += 1  # for forecast step (fcstep)
        if self.condition_on_residual:
            base_input_dim += self.num_input_channels_prognostic[dataset_name]
        if self.input_noise is not None and dataset_name == self.input_noise_dataset:
            base_input_dim += self.n_step_input * self.input_noise.n_channels
            base_input_dim -= self.n_step_input * len(self.input_noise.consumed_input_idx)
        return base_input_dim

    def _input_noise_inputs(
        self,
        x: torch.Tensor,
        grid_shard_sizes: DatasetShardSizes | None,
        model_comm_group: ProcessGroup | None,
    ) -> torch.Tensor:
        """The channels the input noise reads, on the full grid, shape ``(batch, time, grid, vars)``.

        Taken from the first ensemble member: these are forcings, identical across members.
        """
        with torch.no_grad():
            inputs = x[:, :, 0, :, self.input_noise.consumed_input_idx]
            shard_sizes = grid_shard_sizes.get(self.input_noise_dataset) if grid_shard_sizes is not None else None
            if shard_sizes is not None:
                # spectral operations need the whole sphere
                inputs = gather_tensor(inputs, -2, shard_sizes, model_comm_group)
        return inputs

    def _assemble_input(
        self,
        x: torch.Tensor,
        fcstep: int,
        batch_ens_size: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
        input_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_ens_size)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        x_skip = self.residual[dataset_name](
            x,
            grid_shard_sizes=grid_shard_sizes,
            model_comm_group=model_comm_group,
            n_step_output=self.n_step_output,
        )

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

        if dataset_name == self.input_noise_dataset and self._encoder_input_keep_idx is not None:
            # after the residual, which addresses x by its full model-input positions
            x = x[..., self._encoder_input_keep_idx]

        if input_noise is not None:
            # Appended per time step, so the "(time vars)" flattening below interleaves
            # noise with data exactly as makani's flatten_history does.
            input_noise = einops.rearrange(
                input_noise, "batch ensemble time vars grid -> batch time ensemble grid vars"
            )
            if grid_shard_sizes is not None:
                input_noise = shard_tensor(input_noise, 3, grid_shard_sizes, model_comm_group)
            x = torch.cat((x, input_noise.to(dtype=x.dtype)), dim=-1)

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
        x_skip: torch.Tensor | None,
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

    def forward(
        self,
        x: dict[str, torch.Tensor],
        *,
        fcstep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        ensemble_member_offset: int = 0,
        ensemble_members_total: Optional[int] = None,
        ensemble_group_id: int = 0,
        input_noise_seed: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward operator.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input tensor, shape (bs, m, e, n, f).
        fcstep : int
            Forecast step.
        model_comm_group : ProcessGroup, optional
            Model communication group.
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.
        ensemble_member_offset : int, optional
            Global index of this rank's first ensemble member. Used to seed the
            input noise so a realisation is independent of how members are spread
            over devices.
        ensemble_members_total : int, optional
            Total number of ensemble members across the whole ensemble group.
        ensemble_group_id : int, optional
            Index of the ensemble communication group, decorrelating concurrent batches.
        input_noise_seed : int, optional
            Base seed of the input noise, set by training from the run seed. Without
            it the noise follows torch's seed, so differently seeded inference runs
            draw different members.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict[str, Tensor]
            Output tensor per dataset.
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

        # Driven by the unclamped step: fcstep 0 starts a new rollout and redraws the
        # noise history, later steps advance the existing trajectory.
        input_noise = None
        if self.input_noise is not None:
            noise_inputs = None
            if fcstep == 0 and self.input_noise.conditioned:
                noise_inputs = self._input_noise_inputs(x[self.input_noise_dataset], grid_shard_sizes, model_comm_group)
            self.input_noise.advance(
                fcstep=fcstep,
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                member_offset=ensemble_member_offset,
                num_members_total=ensemble_members_total,
                group_id=ensemble_group_id,
                device=x[self.input_noise_dataset].device,
                inputs=noise_inputs,
                seed=input_noise_seed,
            )
            input_noise = self.input_noise.sample()

        fcstep = min(1, fcstep)
        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_ens_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)
        for dataset_name in x.keys():
            if dataset_name not in self.input_datasets:
                continue

            x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                x[dataset_name],
                fcstep=fcstep,
                batch_ens_size=batch_ens_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
                input_noise=input_noise if dataset_name == self.input_noise_dataset else None,
            )
            if dataset_name == self.input_noise_dataset:
                input_noise = None  # now part of x_data_latent: free it before the encoder runs
            x_skip_dict[dataset_name] = x_skip
            shard_sizes_data_dict[dataset_name] = shard_sizes_data

            (
                encoder_edge_attr,
                encoder_edge_index,
                enc_edge_shard_sizes,
            ) = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            enc_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                dst_nodes=shard_sizes_hidden,
                edges=enc_edge_shard_sizes,
            )

            # Encoder for this dataset
            encoder_name = self.dataset2encoder[dataset_name]
            x_data_latent, x_latent = self.encoder[encoder_name](
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
        x_latent = self.latent_aggregator(x_hidden_latent, dataset_latents)

        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            grid_shard_sizes=shard_sizes_hidden,
            model_comm_group=model_comm_group,
        )

        (
            processor_edge_attr,
            processor_edge_index,
            proc_edge_shard_sizes,
        ) = self.processor_graph_provider.get_edges(
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

        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

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
            ) = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_target,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            decoder_name = self.dataset2decoder[dataset_name]
            x_out = self.decoder[decoder_name](
                (x_latent_proc, x_target_latent),
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
                batch_size,
                batch_ens_size,
                dtype=x[dataset_name].dtype,
                dataset_name=dataset_name,
            )

        return x_out_dict
