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

from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec


class _AggregationReached(RuntimeError):
    pass


class _GraphProvider(nn.Module):
    def get_edges(self, **kwargs):
        return None, None, None


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
        self.dataset2encoder = {"dataset_a": "dataset_a", "dataset_b": "dataset_a"}
        self._graph_name_hidden = "hidden"
        self.input_dim_latent = 4
        self.node_attributes = _HiddenAttributes()
        self.encoder_graph_provider = nn.ModuleDict(
            {dataset_name: _GraphProvider() for dataset_name in self.input_datasets},
        )
        self.encoder = nn.ModuleDict({"dataset_a": _SharedEncoder()})
        self.latent_aggregator = _CaptureAggregator()

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
        value = float(self.input_datasets.index(dataset_name) + 1)
        return torch.full((1, 4), value), None, [1]

    def _assemble_output(self, *args, **kwargs):
        raise NotImplementedError


def test_shared_encoder_preserves_each_dataset_latent() -> None:
    model = _SharedEncoderModel()
    inputs = {
        "dataset_a": torch.zeros(1, 1, 1, 1, 1),
        "dataset_b": torch.zeros(1, 1, 1, 1, 1),
    }

    with pytest.raises(_AggregationReached):
        model(inputs)

    assert list(model.latent_aggregator.latents) == ["dataset_a", "dataset_b"]
    torch.testing.assert_close(model.latent_aggregator.latents["dataset_a"], torch.full((1, 4), 1.0))
    torch.testing.assert_close(model.latent_aggregator.latents["dataset_b"], torch.full((1, 4), 2.0))
