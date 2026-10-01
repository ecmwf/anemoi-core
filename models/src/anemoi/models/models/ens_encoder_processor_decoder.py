# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from dataclasses import dataclass
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.encoder_processor_decoder import ForwardContext
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EnsForwardContext(ForwardContext):
    """Forward context of the ensemble model.

    Attributes
    ----------
    fcstep : int
        Forecast step the input is conditioned on, clipped to ``{0, 1}``.
    """

    fcstep: int


class AnemoiEnsModelEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble functionality.

    Extends the base model with a forecast-step input feature, an optional conditioning on the
    residual, and noise injection into the latent before the processor.
    """

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
        self.condition_on_residual = DotDict(model_config).model.condition_on_residual
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
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
        return base_input_dim

    def _init_forward_context(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: Optional[ProcessGroup],
        grid_shard_sizes: DatasetShardSizes | None,
        fcstep: int,
        **kwargs,
    ) -> EnsForwardContext:
        ctx = super()._init_forward_context(
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            **kwargs,
        )
        return EnsForwardContext(**vars(ctx), fcstep=min(1, fcstep))

    def _assemble_input(
        self,
        x: Tensor,
        ctx: EnsForwardContext,
        dataset_name: str,
    ) -> tuple[Tensor, Tensor, ShardSizes]:
        """Base input features, followed by the forecast step and optionally the residual."""
        x_data_latent, x_skip, grid_shard_sizes = super()._assemble_input(x, ctx, dataset_name)

        fcstep_feature = torch.full((x_data_latent.shape[0], 1), float(ctx.fcstep), device=x.device)
        x_data_latent = torch.cat((x_data_latent, fcstep_feature), dim=-1)

        if self.condition_on_residual:
            # Prognostic variables of the first residual step, matching ``_calculate_input_dim``.
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            x_skip_cond = x_skip[:, 0][..., self._internal_input_idx[dataset_name]]
            x_skip_cond = einops.rearrange(x_skip_cond, "batch ensemble grid vars -> (batch ensemble grid) vars")
            x_data_latent = torch.cat((x_data_latent, x_skip_cond), dim=-1)

        return x_data_latent, x_skip, grid_shard_sizes

    def _prepare_processor_input(self, x_latent: Tensor, ctx: EnsForwardContext) -> tuple[Tensor, dict]:
        """Inject noise into the latent entering the processor and condition the processor on it."""
        hidden_name = self._hidden_names[-1]

        x_latent_noised, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=ctx.batch_size,
            ensemble_size=ctx.ensemble_size,
            grid_size=self.node_attributes.num_nodes[hidden_name],
            grid_shard_sizes=ctx.shard_sizes_hidden[hidden_name],
            model_comm_group=ctx.model_comm_group,
        )

        processor_kwargs = dict(ctx.processor_kwargs)
        if latent_noise is not None:
            processor_kwargs["cond"] = latent_noise

        return x_latent_noised, processor_kwargs

    def forward(
        self,
        x: dict[str, Tensor],
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
        return super().forward(
            x,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
            fcstep=fcstep,
            **kwargs,
        )
