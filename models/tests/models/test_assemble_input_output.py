# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Encoder input and decoder output layout of the encoder-processor-decoder models.

The first encoder layer of a trained model expects its input columns in a fixed order, so these
tests pin that order for every model, together with the residual each model hands to its decoder.
"""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.residual import SkipConnection
from anemoi.models.models import AnemoiEnsModelEncProcDec
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec

BATCH = 2
ENSEMBLE = 2
GRID = 3
N_STEP_INPUT = 2
N_STEP_OUTPUT = 2
NODE_ATTR_DIM = 2

# Model inputs are prog0, prog1, force; model outputs are prog0, prog1, diag.
N_INPUT_VARS = 3
N_OUTPUT_VARS = 3
INPUT_PROGNOSTIC = [0, 1]
OUTPUT_PROGNOSTIC = [0, 1]


class _NodeAttributes(nn.Module):
    """Stand-in for the graph node attributes, with values that are easy to tell apart from data."""

    attr_ndims = {"data": NODE_ATTR_DIM, "hidden": NODE_ATTR_DIM}

    def forward(self, name: str, batch_size: int) -> torch.Tensor:
        assert name == "data"
        per_node = -1.0 - torch.arange(GRID * NODE_ATTR_DIM, dtype=torch.float32).reshape(GRID, NODE_ATTR_DIM)
        return per_node.repeat(batch_size, 1)


def _make_model(model_class: type, condition_on_residual: bool = False) -> nn.Module:
    """Build a model with only the parts that input and output assembly use."""
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)

    data_config = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": []})
    name_to_index = {"prog0": 0, "prog1": 1, "force": 2, "diag": 3}
    model.data_indices = {"data": IndexCollection(data_config, name_to_index)}

    model.n_step_input = N_STEP_INPUT
    model.n_step_output = N_STEP_OUTPUT
    model.condition_on_residual = condition_on_residual
    model.node_attributes = _NodeAttributes()
    model._graph_name_hidden = "hidden"
    model.dataset2decoder = {"data": "decoder"}
    model.decoders_target_input = {"decoder": SimpleNamespace(dim=0)}
    model.residual = nn.ModuleDict({"data": SkipConnection()})
    model.boundings = {"data": []}
    model._calculate_shapes_and_indices(model.data_indices)
    return model


def _coded(time: int, n_vars: int, offset: float = 0.0) -> torch.Tensor:
    """Tensor of shape (batch, time, ensemble, grid, vars) whose values give away their own position."""
    b, t, e, g, v = torch.meshgrid(
        torch.arange(BATCH),
        torch.arange(time),
        torch.arange(ENSEMBLE),
        torch.arange(GRID),
        torch.arange(n_vars),
        indexing="ij",
    )
    return (10000 * b + 1000 * e + 100 * t + 10 * g + v).to(torch.float32) + offset


def _rows(x: torch.Tensor) -> torch.Tensor:
    """One row per (batch, ensemble, grid point), holding all time steps of all variables, time first."""
    batch, time, ensemble, grid, n_vars = x.shape
    return x.permute(0, 2, 3, 1, 4).reshape(batch * ensemble * grid, time * n_vars)


def _node_attribute_rows(batch_size: int) -> torch.Tensor:
    return _NodeAttributes()("data", batch_size)


def _last_input_step(x: torch.Tensor) -> torch.Tensor:
    """The skip connection: the last input step, repeated for every output step."""
    return x[:, -1:].expand(-1, N_STEP_OUTPUT, -1, -1, -1)


def test_enc_proc_dec_assemble_input() -> None:
    model = _make_model(AnemoiModelEncProcDec)
    x = _coded(N_STEP_INPUT, N_INPUT_VARS)

    x_data_latent, x_skip, grid_shard_sizes = model._assemble_input(x, BATCH * ENSEMBLE, None, dataset_name="data")

    expected = torch.cat((_rows(x), _node_attribute_rows(BATCH * ENSEMBLE)), dim=-1)
    torch.testing.assert_close(x_data_latent, expected, rtol=0, atol=0)
    assert x_data_latent.shape[-1] == model.input_dim["data"]
    torch.testing.assert_close(x_skip, _last_input_step(x), rtol=0, atol=0)
    assert grid_shard_sizes is None


@pytest.mark.parametrize("condition_on_residual", [False, True])
def test_ens_enc_proc_dec_assemble_input(condition_on_residual: bool) -> None:
    model = _make_model(AnemoiEnsModelEncProcDec, condition_on_residual=condition_on_residual)
    x = _coded(N_STEP_INPUT, N_INPUT_VARS)
    fcstep = 1

    x_data_latent, x_skip, _ = model._assemble_input(x, fcstep, BATCH * ENSEMBLE, dataset_name="data")

    blocks = [
        _rows(x),
        _node_attribute_rows(BATCH * ENSEMBLE),
        torch.full((BATCH * ENSEMBLE * GRID, 1), float(fcstep)),
    ]
    if condition_on_residual:
        # The first output step of the residual, prognostic variables only.
        blocks.append(_rows(_last_input_step(x)[:, :1, ..., INPUT_PROGNOSTIC]))
    torch.testing.assert_close(x_data_latent, torch.cat(blocks, dim=-1), rtol=0, atol=0)
    assert x_data_latent.shape[-1] == model.input_dim["data"]
    torch.testing.assert_close(x_skip, _last_input_step(x), rtol=0, atol=0)


def test_transport_assemble_input() -> None:
    model = _make_model(AnemoiTransportModelEncProcDec)
    x = _coded(N_STEP_INPUT, N_INPUT_VARS)
    y_noised = _coded(N_STEP_OUTPUT, N_OUTPUT_VARS, offset=0.5)

    x_data_latent, x_skip, _ = model._assemble_input(x, y_noised, BATCH * ENSEMBLE, dataset_name="data")

    expected = torch.cat((_rows(x), _rows(y_noised), _node_attribute_rows(BATCH * ENSEMBLE)), dim=-1)
    torch.testing.assert_close(x_data_latent, expected, rtol=0, atol=0)
    assert x_data_latent.shape[-1] == model.input_dim["data"]
    assert x_skip is None


@pytest.mark.parametrize("condition_on_residual", [False, True])
def test_transport_tend_assemble_input(condition_on_residual: bool) -> None:
    model = _make_model(AnemoiTransportTendModelEncProcDec, condition_on_residual=condition_on_residual)
    x = _coded(N_STEP_INPUT, N_INPUT_VARS)
    y_noised = _coded(N_STEP_OUTPUT, N_OUTPUT_VARS, offset=0.5)

    x_data_latent, x_skip, _ = model._assemble_input(x, y_noised, BATCH * ENSEMBLE, dataset_name="data")

    # The reference state keeps only the prognostic variables, one row per (batch, ensemble) member.
    prognostic_residual = _last_input_step(x)[..., INPUT_PROGNOSTIC]
    expected_skip = _rows(prognostic_residual).reshape(BATCH * ENSEMBLE, GRID, N_STEP_OUTPUT * len(INPUT_PROGNOSTIC))
    torch.testing.assert_close(x_skip, expected_skip, rtol=0, atol=0)

    blocks = [_rows(x), _rows(y_noised), _node_attribute_rows(BATCH * ENSEMBLE)]
    if condition_on_residual:
        blocks.append(_rows(prognostic_residual))
    torch.testing.assert_close(x_data_latent, torch.cat(blocks, dim=-1), rtol=0, atol=0)
    assert x_data_latent.shape[-1] == model.input_dim["data"]


def _flat_decoder_output() -> torch.Tensor:
    """Decoder output as the decoder returns it: one row per (batch, ensemble, grid point)."""
    return _rows(_coded(N_STEP_OUTPUT, N_OUTPUT_VARS, offset=0.25))


def _expected_output_with_residual(x_skip: torch.Tensor) -> torch.Tensor:
    expected = _coded(N_STEP_OUTPUT, N_OUTPUT_VARS, offset=0.25)
    expected[..., OUTPUT_PROGNOSTIC] += x_skip[..., INPUT_PROGNOSTIC]
    return expected


def test_enc_proc_dec_assemble_output_adds_residual_to_prognostic_variables() -> None:
    model = _make_model(AnemoiModelEncProcDec)
    x_skip = _last_input_step(_coded(N_STEP_INPUT, N_INPUT_VARS))

    x_out = model._assemble_output(_flat_decoder_output(), x_skip, BATCH, ENSEMBLE, torch.float32, "data")

    torch.testing.assert_close(x_out, _expected_output_with_residual(x_skip), rtol=0, atol=0)


def test_ens_enc_proc_dec_assemble_output_adds_residual_to_prognostic_variables() -> None:
    model = _make_model(AnemoiEnsModelEncProcDec)
    x_skip = _last_input_step(_coded(N_STEP_INPUT, N_INPUT_VARS))

    x_out = model._assemble_output(
        _flat_decoder_output(),
        x_skip,
        BATCH,
        BATCH * ENSEMBLE,
        dtype=torch.float32,
        dataset_name="data",
    )

    torch.testing.assert_close(x_out, _expected_output_with_residual(x_skip), rtol=0, atol=0)


@pytest.mark.parametrize("model_class", [AnemoiTransportModelEncProcDec, AnemoiTransportTendModelEncProcDec])
def test_transport_assemble_output_only_reshapes(model_class: type) -> None:
    model = _make_model(model_class)
    x_skip = torch.ones(BATCH * ENSEMBLE, GRID, N_STEP_OUTPUT * len(INPUT_PROGNOSTIC))

    x_out = model._assemble_output(_flat_decoder_output(), x_skip, BATCH, ENSEMBLE, torch.float32)

    torch.testing.assert_close(x_out, _coded(N_STEP_OUTPUT, N_OUTPUT_VARS, offset=0.25), rtol=0, atol=0)
