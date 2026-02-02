# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Callable
from typing import Optional

import einops
import torch
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import get_or_apply_shard_shapes
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.models.tendency_helpers import TendencyPredictHelperMixin
from anemoi.utils.config import DotDict


class AnemoiTendModelEncProcDec(TendencyPredictHelperMixin, AnemoiModelEncProcDec):
    """Deterministic tendency model."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
    ) -> None:
        model_config_local = DotDict(model_config)
        self.condition_on_residual = model_config_local.model.get("condition_on_residual", False)
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
        )

    def _calculate_input_dim(self, dataset_name: str) -> int:
        input_dim = super()._calculate_input_dim(dataset_name)
        if self.condition_on_residual:
            input_dim += len(self.data_indices[dataset_name].model.input.prognostic) * self.multi_out
        return input_dim

    @staticmethod
    def _apply_imputer_inverse(
        post_processors: dict[str, nn.Module],
        dataset_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return TendencyPredictHelperMixin._tendency_apply_imputer_inverse(post_processors, dataset_name, x)

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_shapes: dict | None = None,
        model_comm_group=None,
        dataset_name=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes[dataset_name](self._graph_name_data, batch_size=batch_size)
        grid_shard_shapes = grid_shard_shapes[dataset_name] if grid_shard_shapes is not None else None

        x_skip = self.residual[dataset_name](x, grid_shard_shapes, model_comm_group, multi_out=self.multi_out)[
            ..., self._internal_input_idx[dataset_name]
        ]
        assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
        x_skip = einops.rearrange(x_skip, "batch time ensemble grid vars -> (batch ensemble) grid (time vars)")

        if grid_shard_shapes is not None:
            shard_shapes_nodes = get_or_apply_shard_shapes(
                node_attributes_data, 0, shard_shapes_dim=grid_shard_shapes, model_comm_group=model_comm_group
            )
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,
        )
        if self.condition_on_residual:
            x_data_latent = torch.cat(
                (x_data_latent, einops.rearrange(x_skip, "bse grid vars -> (bse grid) vars")), dim=-1
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
    ) -> torch.Tensor:
        del x_skip
        x_out = einops.rearrange(
            x_out,
            "(batch ensemble grid) (time vars) -> batch time ensemble grid vars",
            batch=batch_size,
            ensemble=ensemble_size,
            time=self.multi_out,
        ).to(dtype=dtype)
        return x_out

    def compute_tendency(
        self,
        x_t1: dict[str, torch.Tensor],
        x_t0: dict[str, torch.Tensor],
        pre_processors_state: dict[str, Callable],
        pre_processors_tendencies: dict[str, Callable],
        input_post_processor: Optional[Callable] = None,
        skip_imputation: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self._tendency_compute_tendency(
            x_t1,
            x_t0,
            pre_processors_state,
            pre_processors_tendencies,
            input_post_processor=input_post_processor,
            skip_imputation=skip_imputation,
        )

    def add_tendency_to_state(
        self,
        state_inp: dict[str, torch.Tensor],
        tendency: dict[str, torch.Tensor],
        post_processors_state: dict[str, Callable],
        post_processors_tendencies: dict[str, Callable],
        output_pre_processor: dict[str, Optional[Callable]] = None,
        skip_imputation: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self._tendency_add_tendency_to_state(
            state_inp,
            tendency,
            post_processors_state,
            post_processors_tendencies,
            output_pre_processor=output_pre_processor,
            skip_imputation=skip_imputation,
        )

    def prepare_predict_inputs(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        **kwargs,
    ) -> tuple[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]], dict[str, Optional[list]]]:
        del kwargs
        return self._tendency_prepare_predict_inputs(batch, pre_processors, multi_step, model_comm_group)

    def finalize_predict_outputs(
        self,
        out: dict[str, torch.Tensor],
        post_processors: dict[str, nn.Module],
        prepared_inputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: dict[str, Optional[list]] = None,
        gather_out: bool = True,
        post_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        return self._tendency_finalize_predict_outputs(
            out,
            post_processors,
            prepared_inputs,
            model_comm_group=model_comm_group,
            grid_shard_shapes=grid_shard_shapes,
            gather_out=gather_out,
            post_processors_tendencies=post_processors_tendencies,
        )

    def apply_reference_state_truncation(
        self, x: dict[str, torch.Tensor], grid_shard_shapes: dict[str, list], model_comm_group: ProcessGroup
    ) -> dict[str, torch.Tensor]:
        return self._tendency_apply_reference_state_truncation(x, grid_shard_shapes, model_comm_group)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        pre_processors: dict[str, nn.Module],
        post_processors: dict[str, nn.Module],
        multi_step: int,
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        post_processors_tendencies: Optional[dict[str, nn.Module]] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        with torch.no_grad():
            prepared_inputs, grid_shard_shapes = self.prepare_predict_inputs(
                batch,
                pre_processors,
                multi_step,
                model_comm_group,
            )
            xs = prepared_inputs[0]
            out = self.forward(
                xs,
                model_comm_group=model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
            )
            out = self.finalize_predict_outputs(
                out,
                post_processors,
                prepared_inputs,
                model_comm_group,
                grid_shard_shapes,
                gather_out,
                post_processors_tendencies=post_processors_tendencies,
            )

        return out
