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

from anemoi.models.interface import AnemoiModelInterface


class AnemoiModel(AnemoiModelInterface):
    """Standard implementation of :class:`AnemoiModelInterface`."""

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
        self.n_step_input = model_config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.data_indices = data_indices
        self.pre_processors = pre_processors
        self.post_processors = post_processors
        self.pre_processors_tendencies = pre_processors_tendencies
        self.post_processors_tendencies = post_processors_tendencies

        # Instantiate the backbone from model_config.model.backbone
        nn_cfg = {
            "_target_": model_config.model.backbone._target_,
            "_convert_": getattr(model_config.model.backbone, "_convert_", "none"),
        }
        self.backbone = instantiate(
            nn_cfg,
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            _recursive_=False,
        )
    def pre_process(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.pre_processors[name](x[name], in_place=False) for name in x}

    def post_process(self, y: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.post_processors[name](y[name], in_place=False) for name in y}

    def forward(
        self,
        x: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        return self.post_process(self.backbone(self.pre_process(x), **kwargs))

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        model_comm_group: Optional[ProcessGroup] = None,
        gather_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            x = {name: batch[name][:, : self.n_step_input, None, ...] for name in batch}
            return self.forward(x, model_comm_group=model_comm_group, **kwargs)
