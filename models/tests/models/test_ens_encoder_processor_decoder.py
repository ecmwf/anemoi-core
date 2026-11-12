# (C) Copyright 2026 Anemoi contributors.
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

from anemoi.models.models.encoder_processor_decoder import EncoderSource
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec


class _NodeAttributes(nn.Module):
    num_trainable_parameters: dict[str, int] = {}


def _ens_model(prognostic_indices: list[int], num_variables: int, input_steps: int) -> AnemoiEnsModelEncProcDec:
    model = AnemoiEnsModelEncProcDec.__new__(AnemoiEnsModelEncProcDec)
    nn.Module.__init__(model)
    model.condition_on_residual = True
    model.n_step_input = {"data": input_steps}
    model.num_input_channels = {"data": num_variables}
    model.num_input_channels_prognostic = {"data": len(prognostic_indices)}
    model._internal_input_idx = {"data": prognostic_indices}
    model.node_attributes = _NodeAttributes()
    model.dynamic_node_attribute_dims = {}
    return model


def _source(x_data_latent: torch.Tensor, x_skip: torch.Tensor | None) -> EncoderSource:
    return EncoderSource(
        dataset_name="data",
        x_data_latent=x_data_latent,
        x_skip=x_skip,
        edge_attr=torch.empty(0),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        shard_info=None,
    )


def test_condition_source_appends_fcstep_and_prognostic_residual_variables() -> None:
    batch_size, input_steps, output_steps, ensemble_size, num_nodes, num_variables = 2, 3, 2, 2, 5, 4
    prognostic_indices = [0, 2]
    model = _ens_model(prognostic_indices, num_variables, input_steps)

    # The assembled input width the encoder is built for: base input + fcstep + prognostic residual.
    base_dim = super(AnemoiEnsModelEncProcDec, model)._calculate_input_dim("data")
    expected_num_features = base_dim + 1 + len(prognostic_indices)
    assert model._calculate_input_dim("data") == expected_num_features

    num_rows = batch_size * ensemble_size * num_nodes
    x_data_latent = torch.randn(num_rows, base_dim)
    # Residual with several output steps: (batch, time, ensemble, grid, variables).
    x_skip = torch.arange(
        batch_size * output_steps * ensemble_size * num_nodes * num_variables,
        dtype=torch.float32,
    ).reshape(batch_size, output_steps, ensemble_size, num_nodes, num_variables)

    conditioned = model._condition_source(_source(x_data_latent, x_skip), fcstep=1)

    assert conditioned.x_data_latent.shape == (num_rows, expected_num_features)
    torch.testing.assert_close(conditioned.x_data_latent[:, :base_dim], x_data_latent)
    torch.testing.assert_close(conditioned.x_data_latent[:, base_dim], torch.ones(num_rows))
    # Only the prognostic variables of the first output step, rows ordered (batch, ensemble, grid).
    expected_conditioning = x_skip[:, 0][..., prognostic_indices].reshape(-1, len(prognostic_indices))
    torch.testing.assert_close(conditioned.x_data_latent[:, -len(prognostic_indices) :], expected_conditioning)


def test_condition_source_rejects_residual_on_other_nodes() -> None:
    model = _ens_model([0, 2], num_variables=4, input_steps=1)
    x_skip = torch.zeros(1, 1, 2, 5, 4)  # 10 rows, latent has 7
    with pytest.raises(AssertionError, match="not on the same nodes"):
        model._condition_source(_source(torch.zeros(7, 3), x_skip), fcstep=0)
