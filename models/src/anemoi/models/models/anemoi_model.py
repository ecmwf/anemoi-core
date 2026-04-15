# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import uuid
from typing import Optional

import torch
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.utils.config import DotDict


class AnemoiModel(torch.nn.Module):
    """Standard implementation of :class:`ModelInterface`."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        graph_data: dict[str, HeteroData],
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        pre_processors: torch.nn.ModuleDict,
        post_processors: torch.nn.ModuleDict,
        pre_processors_tendencies: torch.nn.ModuleDict,
        post_processors_tendencies: torch.nn.ModuleDict,
        statistics_tendencies: dict | None = None,
        supporting_arrays: dict | None = None,
        **_,
    ) -> None:
        super().__init__()
        self.id = str(uuid.uuid4())
        self.n_step_input = model_config.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.data_indices = data_indices
        self.pre_processors = pre_processors
        self.post_processors = post_processors
        self.pre_processors_tendencies = pre_processors_tendencies
        self.post_processors_tendencies = post_processors_tendencies

        nn_cfg = {
            "_target_": model_config.backbone._target_,
            "_convert_": getattr(model_config.backbone, "_convert_", "none"),
        }
        self.backbone = instantiate(
            nn_cfg,
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            _recursive_=False,
        )
        self.data_nodes_name = self.backbone.data_nodes_name
        self.hidden_nodes_name = self.backbone.hidden_nodes_name

    def pre_process(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.pre_processors[name](x[name], in_place=False) for name in x}

    def post_process(self, y: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.post_processors[name](y[name], in_place=False) for name in y}

    def forward(
        self,
        x: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.backbone(x, **kwargs)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            x = {name: batch[name][:, : self.n_step_input, None, ...] for name in batch}
            x = self.pre_process(x)
            y = self.forward(x, model_comm_group=model_comm_group, **kwargs)
            y = self.post_process(y)
            return y


class AnemoiDiffusionModel(AnemoiModel):
    """Anemoi wrapper for diffusion-capable backbones."""

    def get_diffusion_parameters(self) -> tuple[float, float, float]:
        return self.backbone.sigma_max, self.backbone.sigma_min, self.backbone.sigma_data

    def forward_with_preconditioning(
        self,
        x: dict[str, torch.Tensor],
        y_noised: dict[str, torch.Tensor],
        sigma: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.backbone.fwd_with_preconditioning(x, y_noised, sigma, **kwargs)

    def apply_imputer_inverse(self, dataset_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.backbone._apply_imputer_inverse(self.post_processors, dataset_name, x)

    def apply_reference_state_truncation(
        self,
        x: dict[str, torch.Tensor],
        grid_shard_shapes,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> dict[str, torch.Tensor]:
        return self.backbone.apply_reference_state_truncation(x, grid_shard_shapes, model_comm_group)


class AnemoiDiffusionTendencyModel(AnemoiDiffusionModel):
    """Anemoi wrapper for diffusion backbones that predict tendencies."""

    def get_tendency_processors(self, dataset_name: str) -> tuple[object, object]:
        return self.pre_processors_tendencies[dataset_name], self.post_processors_tendencies[dataset_name]

    def compute_tendency_step(
        self,
        dataset_name: str,
        y_step: torch.Tensor,
        x_ref_step: torch.Tensor,
        tendency_pre_processor: object,
    ) -> torch.Tensor:
        return self.backbone.compute_tendency(
            {dataset_name: y_step},
            {dataset_name: x_ref_step},
            {dataset_name: self.pre_processors[dataset_name]},
            {dataset_name: tendency_pre_processor},
            input_post_processor={dataset_name: self.post_processors[dataset_name]},
            skip_imputation=True,
        )[dataset_name]

    def add_tendency_to_state_step(
        self,
        dataset_name: str,
        x_ref_step: torch.Tensor,
        tendency_step: torch.Tensor,
        tendency_post_processor: object,
    ) -> torch.Tensor:
        return self.backbone.add_tendency_to_state(
            {dataset_name: x_ref_step},
            {dataset_name: tendency_step},
            {dataset_name: self.post_processors[dataset_name]},
            {dataset_name: tendency_post_processor},
            output_pre_processor={dataset_name: self.pre_processors[dataset_name]},
            skip_imputation=True,
        )[dataset_name]
