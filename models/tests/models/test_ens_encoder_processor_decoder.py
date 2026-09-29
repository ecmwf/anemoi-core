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

from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec


class _AggregationReached(RuntimeError):
    pass


class _GraphProvider(nn.Module):
    def get_edges(self, **kwargs):
        return None, None, None


class _HiddenAttributes(nn.Module):
    def forward(self, node_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, 4)


class _Encoder(nn.Module):
    hidden_dim = 4

    def forward(self, x, **kwargs):
        return x[0], x[0]


class _CaptureAggregator(nn.Module):
    def forward(
        self,
        hidden_latent: torch.Tensor,
        latents: dict[str, torch.Tensor],
        dropped_sources=None,
    ) -> torch.Tensor:
        self.latents = dict(latents)
        self.dropped_sources = set(dropped_sources or ())
        raise _AggregationReached


class _MultiEncoderEnsModel(AnemoiEnsModelEncProcDec):
    """Ensemble model stub with two encoders that stops at the latent aggregation."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.input_datasets = ["data", "ocean"]
        self.principal_dataset_name = "data"
        self.dataset2encoder = {"data": "data", "ocean": "ocean"}
        self._graph_name_hidden = "hidden"
        self.input_dim_latent = 4
        self.node_attributes = _HiddenAttributes()
        self.encoder_graph_provider = nn.ModuleDict(
            {dataset_name: _GraphProvider() for dataset_name in self.input_datasets},
        )
        self.encoder = nn.ModuleDict({name: _Encoder() for name in self.input_datasets})
        self.latent_aggregator = _CaptureAggregator()

    def _build_networks(self, model_config) -> None:
        raise NotImplementedError

    def _assemble_input(
        self,
        x: torch.Tensor,
        fcstep: int,
        batch_ens_size: int,
        grid_shard_sizes=None,
        model_comm_group=None,
        dataset_name: str | None = None,
    ):
        value = float(self.input_datasets.index(dataset_name) + 1)
        return torch.full((batch_ens_size, 4), value), None, None

    def _assemble_output(self, *args, **kwargs):
        raise NotImplementedError


def _inputs(ensemble_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "data": torch.zeros(1, 1, ensemble_size, 1, 1),
        "ocean": torch.zeros(1, 1, ensemble_size, 1, 1),
    }


def test_ens_model_aggregates_each_dataset_latent_over_batch_and_ensemble() -> None:
    model = _MultiEncoderEnsModel()

    with pytest.raises(_AggregationReached):
        model(_inputs(ensemble_size=2), fcstep=0)

    assert list(model.latent_aggregator.latents) == ["data", "ocean"]
    # batch * ensemble rows per dataset
    torch.testing.assert_close(model.latent_aggregator.latents["data"], torch.full((2, 4), 1.0))
    torch.testing.assert_close(model.latent_aggregator.latents["ocean"], torch.full((2, 4), 2.0))
    assert model.latent_aggregator.dropped_sources == set()


def test_ens_model_forwards_dropped_datasets_to_the_aggregator() -> None:
    model = _MultiEncoderEnsModel()

    with pytest.raises(_AggregationReached):
        # The principal dataset can never be dropped; only `ocean` should reach the aggregator.
        model(_inputs(), fcstep=0, dropped_dataset_names=["data", "ocean"])

    assert model.latent_aggregator.dropped_sources == {"ocean"}


def test_ens_model_decoder_only_dropout_is_not_masked_in_the_aggregator() -> None:
    model = _MultiEncoderEnsModel()

    with pytest.raises(_AggregationReached):
        model(_inputs(), fcstep=0, decoder_dropped_dataset_names=["ocean"])

    # Decoder-only dropout keeps the encoder latent active.
    assert model.latent_aggregator.dropped_sources == set()


def test_resolve_dropped_datasets_protects_principal_and_deduplicates() -> None:
    model = _MultiEncoderEnsModel()

    dropped, decoder_dropped = model._resolve_dropped_datasets(["data", "ocean"], ["ocean", "data"])

    assert dropped == {"ocean"}
    # `ocean` is already fully dropped, so it is removed from the decoder-only set.
    assert decoder_dropped == set()
