# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch import nn

from anemoi.models.layers.residual import SkipConnection
from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec


class _NodeAttributes(nn.Module):
    def __init__(self, num_nodes: int, num_features: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features

    def forward(self, node_name: str, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size * self.num_nodes, self.num_features)


def test_assemble_input_conditions_on_prognostic_residual_variables() -> None:
    batch_size, input_steps, ensemble_size, num_nodes, num_variables = 2, 3, 2, 5, 4
    prognostic_indices = [0, 2]
    node_features = 3

    model = AnemoiEnsModelEncProcDec.__new__(AnemoiEnsModelEncProcDec)
    nn.Module.__init__(model)
    model.condition_on_residual = True
    model.n_step_output = 2
    model._internal_input_idx = {"data": prognostic_indices}
    model.node_attributes = _NodeAttributes(num_nodes, node_features)
    model.residual = nn.ModuleDict({"data": SkipConnection()})

    x = torch.arange(
        batch_size * input_steps * ensemble_size * num_nodes * num_variables,
        dtype=torch.float32,
    ).reshape(batch_size, input_steps, ensemble_size, num_nodes, num_variables)

    x_data_latent, x_skip, grid_shard_sizes = model._assemble_input(
        x,
        fcstep=1,
        batch_ens_size=batch_size * ensemble_size,
        dataset_name="data",
    )

    expected_num_features = input_steps * num_variables + node_features + 1 + len(prognostic_indices)
    assert x_data_latent.shape == (batch_size * ensemble_size * num_nodes, expected_num_features)
    assert x_skip.shape == (batch_size, model.n_step_output, ensemble_size, num_nodes, num_variables)
    assert grid_shard_sizes is None

    expected_conditioning = x[:, -1, ..., prognostic_indices].reshape(-1, len(prognostic_indices))
    torch.testing.assert_close(x_data_latent[:, -len(prognostic_indices) :], expected_conditioning)
