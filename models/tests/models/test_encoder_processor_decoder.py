# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from torch import nn

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec


class _AggregationReached(RuntimeError):
    pass


class _GraphProvider(nn.Module):
    def get_edges(self, **kwargs):
        return torch.zeros(1, 1), torch.zeros(2, 1, dtype=torch.long), None


class _HiddenAttributes(nn.Module):
    def forward(self, node_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(1, 4)


class _SharedEncoder(nn.Module):
    hidden_dim = 4

    def forward(self, x, **kwargs):
        return x[0], x[0]


class _CaptureAggregator(nn.Module):
    def forward(self, hidden_latent: torch.Tensor, latents: dict[str, torch.Tensor]) -> torch.Tensor:
        self.latents = dict(latents)
        raise _AggregationReached


class _SharedEncoderModel(AnemoiModelEncProcDec):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.input_datasets = ["dataset_a", "dataset_b"]
        self.dataset2encoder = {"dataset_a": "shared", "dataset_b": "shared"}
        self.encoder2datasets = {"shared": self.input_datasets}
        self.encoder_fusing_strategy = {"shared": "sequential"}
        self._graph_name_hidden = "hidden"
        self.input_dim_latent = 4
        self.node_attributes = _HiddenAttributes()
        self.encoder_graph_provider = nn.ModuleDict(
            {dataset_name: _GraphProvider() for dataset_name in self.input_datasets},
        )
        self.encoder = nn.ModuleDict({"shared": _SharedEncoder()})
        self.encoder_src_projection = nn.ModuleDict()
        self.latent_aggregator = _CaptureAggregator()

    def _hidden_coordinates(self) -> torch.Tensor:
        return torch.zeros(1, 2)

    def _build_networks(self, model_config) -> None:
        raise NotImplementedError

    def _assemble_input(
        self,
        x: torch.Tensor,
        batch_size: int,
        grid_shard_sizes=None,
        model_comm_group=None,
        dataset_name: str | None = None,
    ):
        del x, batch_size, grid_shard_sizes, model_comm_group
        value = float(self.input_datasets.index(dataset_name) + 1)
        return torch.zeros(1, 2), torch.full((1, 4), value), None, None, None, None

    def _assemble_output(self, *args, **kwargs):
        raise NotImplementedError


def test_shared_encoder_preserves_each_dataset_latent() -> None:
    model = _SharedEncoderModel()
    inputs = Batch(
        data={name: torch.zeros(1, 1, 1, 1, 1) for name in model.input_datasets},
        coordinates={name: torch.zeros(1, 2) for name in model.input_datasets},
        metadata={"static_coords": frozenset(model.input_datasets)},
        layouts={name: TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4) for name in model.input_datasets},
        variables={name: ["a"] for name in model.input_datasets},
        statistics={name: {} for name in model.input_datasets},
    )

    with pytest.raises(_AggregationReached):
        model(inputs, target=inputs)

    assert list(model.latent_aggregator.latents) == ["dataset_a", "dataset_b"]
    torch.testing.assert_close(model.latent_aggregator.latents["dataset_a"], torch.full((1, 4), 1.0))
    torch.testing.assert_close(model.latent_aggregator.latents["dataset_b"], torch.full((1, 4), 2.0))
