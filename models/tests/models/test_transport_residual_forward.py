# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Real-instantiation tests for :class:`AnemoiTransportResidualModelEncProcDec`.

Unlike ``test_transport_residual_model.py`` (which builds bare ``__new__`` instances to
unit-test index arithmetic), these tests construct the model through its actual
``__init__`` with a real graph, real graph-transformer encoders/decoders/processor and a
real ``InterpolationConnection`` loaded from an ``.npz`` file. This validates that

- with ``conditioning_only_datasets: [input]`` and ``history_less_datasets: [output]`` the encoders
  are built with the asymmetric-role dims (the source gets no corrupted-target term; the target gets
  the corrupted-target term but NO input-history term),
- a full ``forward()`` pass runs through the BASE transport path (the multi-encoder architecture:
  ``x = {source (native small grid)}`` ONLY — the history-less target is absent from ``x`` and is
  encoded from its noised target), encoding the source as conditioning-only, and produces the right
  output shape,
- ``_after_sampling`` reconstructs physical states from the RAW source reference through real
  affine processors, matching a hand-computed reference (raw-batch contract).

The residual subclass no longer builds any conditioning itself; the ``InterpolationConnection`` is
consumed only in ``_after_sampling`` for the residual reconstruction.

Construction boilerplate (graph, layer kernels, model_config skeleton, index collections,
statistics) is shared with ``test_transport_conditioning_only.py`` via the factory fixtures in
``conftest.py``; only the ``real_model`` fixture below is specific to this module's source/target
residual shape.
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.layers.residual import InterpolationConnection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportResidualModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.normalizer import InputNormalizer

# ── grid sizes ────────────────────────────────────────────────────────────────
N_SOURCE = 6  # "input" dataset grid
N_TARGET = 10  # "output" dataset grid (different size on purpose)
N_HIDDEN = 4  # tiny hidden mesh

NUM_CHANNELS = 16
NOISE_CHANNELS = 8
NOISE_COND_DIM = 4
N_STEP_INPUT = 2
N_STEP_OUTPUT = 1

# ── variable roles (mirrors test_transport_residual_model.py) ────────────────
# target "output": global order u(0) tgt(1) v(2) dp(3) diag(4) force(5)
#   forcing=[force], diagnostic=[diag], target=[tgt] -> prognostic=[u, v, dp]
#   residual channels = prognostic minus direct(dp) = [u, v]; model.output=[u, v, dp, diag]
# source "input": global order u(0) v(1) sforce(2), forcing=[sforce]
_TARGET_NAME_TO_INDEX = {"u": 0, "tgt": 1, "v": 2, "dp": 3, "diag": 4, "force": 5}
_SOURCE_NAME_TO_INDEX = {"u": 0, "v": 1, "sforce": 2}

# per-channel stdevs (mean-std normalization with zero mean => norm = x / stdev)
_TARGET_STATE_STDEV = [2.0, 1.0, 4.0, 5.0, 8.0, 1.0]  # u tgt v dp diag force
_TARGET_RESIDUAL_STDEV = [10.0, 1.0, 20.0, 1.0, 1.0, 1.0]  # only u, v matter
_SOURCE_STATE_STDEV = [3.0, 6.0, 1.5]  # u v sforce


def _target_indices(index_collection_factory):
    return index_collection_factory(_TARGET_NAME_TO_INDEX, forcing=["force"], diagnostic=["diag"], target=["tgt"])


def _source_indices(index_collection_factory):
    return index_collection_factory(_SOURCE_NAME_TO_INDEX, forcing=["sforce"])


def _processor_pair(indices, stats: dict) -> tuple[Processors, Processors]:
    cfg = DictConfig({"default": "mean-std"})
    pre = Processors([["normalizer", InputNormalizer(cfg, indices, {k: v.copy() for k, v in stats.items()})]])
    post = Processors(
        [["normalizer", InputNormalizer(cfg, indices, {k: v.copy() for k, v in stats.items()})]], inverse=True
    )
    return pre, post


# ── interpolation matrix: (target grid x source grid), asymmetric distinct values ──
def _interpolation_matrix() -> np.ndarray:
    rng = np.random.default_rng(1234)
    matrix = np.zeros((N_TARGET, N_SOURCE), dtype=np.float32)
    for row in range(N_TARGET):
        # two distinct source nodes per target node, distinct weights per row
        first = row % N_SOURCE
        second = (row * 2 + 1) % N_SOURCE
        matrix[row, first] += 0.25 + 0.1 * row
        matrix[row, second] += 0.5 + 0.05 * row
    assert (matrix.sum(axis=1) > 0).all()
    del rng
    return matrix


@pytest.fixture
def real_model(
    index_collection_factory,
    statistics_factory,
    tiny_graph_factory,
    transport_model_config_factory,
    interpolation_npz_factory,
) -> tuple[AnemoiTransportResidualModelEncProcDec, np.ndarray]:
    matrix = _interpolation_matrix()
    interpolation_file_path = interpolation_npz_factory(matrix)

    data_indices = {
        "input": _source_indices(index_collection_factory),
        "output": _target_indices(index_collection_factory),
    }
    statistics = {
        "input": statistics_factory(_SOURCE_STATE_STDEV),
        "output": statistics_factory(_TARGET_STATE_STDEV),
    }
    graph = tiny_graph_factory(
        {"input": N_SOURCE, "output": N_TARGET, "hidden": N_HIDDEN},
        [("input", "hidden"), ("output", "hidden"), ("hidden", "input"), ("hidden", "output"), ("hidden", "hidden")],
    )
    model_config = transport_model_config_factory(
        num_channels=NUM_CHANNELS,
        noise_channels=NOISE_CHANNELS,
        noise_cond_dim=NOISE_COND_DIM,
        model_extra={
            "residual_prediction": {"output": "input"},
            "direct_prediction": {"output": ["dp"]},
            # The source is encoded on its native grid by the base transport forward and never
            # decoded (multi-encoder conditioning).
            "conditioning_only_datasets": ["input"],
            # The target is predicted but never a model input: it supplies no input history, so its
            # encoder input is the noised target plus node attributes only.
            "history_less_datasets": ["output"],
        },
        residual={
            "_target_": "anemoi.models.layers.residual.InterpolationConnection",
            "interpolation_file_path": interpolation_file_path,
        },
    )
    model = AnemoiTransportResidualModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        n_step_input=N_STEP_INPUT,
        n_step_output=N_STEP_OUTPUT,
        graph_data=graph,
    )
    return model, matrix


def test_real_instantiation_builds_residual_interpolation(real_model) -> None:
    model, _ = real_model
    assert isinstance(model.residual["output"], InterpolationConnection)
    assert model._residual_pairs == {"output": "input"}
    # The -1 sentinel is registered until a residual training mode sets the alignment.
    assert model._output_reference_positions.tolist() == [-1] * N_STEP_OUTPUT


def test_real_instantiation_encoder_input_dims_follow_base_rules(real_model) -> None:
    model, _ = real_model

    # Asymmetric-role input-dim rules apply (no residual concat override):
    #   conditioning-only (source): history only  = n_step_input * len(model.input) + node-attr dims
    #   history-less (target):      target only    = node-attr dims + n_step_output * len(model.output)
    #                               (NO input-history term — the target is never a model input)
    assert model.conditioning_only_datasets == {"input"}
    assert model.history_less_datasets == {"output"}

    source_in = len(model.data_indices["input"].model.input)  # u, v, sforce -> 3
    source_node_attr = model.node_attributes.attr_ndims["input"]
    expected_source = N_STEP_INPUT * source_in + source_node_attr  # no corrupted-target term
    assert model.input_dim["input"] == expected_source
    assert model.encoder["input"].in_channels_src == expected_source

    target_node_attr = model.node_attributes.attr_ndims["output"]
    target_out = len(model.data_indices["output"].model.output)  # u, v, dp, diag -> 4
    expected_target = target_node_attr + N_STEP_OUTPUT * target_out  # no input-history term
    assert model.input_dim["output"] == expected_target
    assert model.encoder["output"].in_channels_src == expected_target


def _forward_inputs(model: AnemoiTransportResidualModelEncProcDec, batch_size: int = 2) -> tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(11)
    n_in_source = len(model.data_indices["input"].model.input)  # u, v, sforce -> 3
    n_out_target = len(model.data_indices["output"].model.output)  # u, v, dp, diag -> 4
    # The honest training shape: ``x`` carries ONLY the conditioning-only source (its native small
    # grid). The predicted target is history-less — it is absent from ``x`` and enters the network
    # solely through its noised (corrupted) target in ``conditioned_target``.
    x = {
        "input": torch.randn(batch_size, N_STEP_INPUT, 1, N_SOURCE, n_in_source, generator=generator),
    }
    # conditioned_target and condition contain ONLY the predicted dataset (mirrors the objective).
    conditioned_target = {
        "output": torch.randn(batch_size, N_STEP_OUTPUT, 1, N_TARGET, n_out_target, generator=generator),
    }
    condition = {"output": torch.full((batch_size, 1, 1, 1, 1), 0.5)}
    return x, conditioned_target, condition


def test_real_forward_pass_shape_and_finiteness(real_model) -> None:
    model, _ = real_model
    x, conditioned_target, condition = _forward_inputs(model)

    # The base transport forward encodes the source as conditioning-only, encodes the history-less
    # target from its noised target, and decodes only the target.
    out = model.forward(x, conditioned_target, condition)

    assert set(x.keys()) == {"input"}  # the honest shape: no target in the input dict
    assert set(out.keys()) == {"output"}  # the conditioning-only source is absent from the result
    assert out["output"].shape == (2, N_STEP_OUTPUT, 1, N_TARGET, 4)
    assert torch.isfinite(out["output"]).all()


def test_real_after_sampling_reconstructs_residual_channels_by_hand(
    real_model, index_collection_factory, statistics_factory
) -> None:
    model, matrix = real_model
    target_indices = model.data_indices["output"]

    pre_src, post_src = _processor_pair(
        _source_indices(index_collection_factory), statistics_factory(_SOURCE_STATE_STDEV)
    )
    _, post_tgt = _processor_pair(_target_indices(index_collection_factory), statistics_factory(_TARGET_STATE_STDEV))
    _, post_res = _processor_pair(
        _target_indices(index_collection_factory), statistics_factory(_TARGET_RESIDUAL_STDEV)
    )
    del pre_src

    post_processors = torch.nn.ModuleDict({"input": post_src, "output": post_tgt})
    post_processors_residuals = torch.nn.ModuleDict({"output": post_res})

    # The training mode persists the same-offset alignment; emulate it before reconstruction.
    model.set_output_reference_positions([1])

    generator = torch.Generator().manual_seed(23)
    # RAW physical source history (kept by the raw-batch _before_sampling): (1, 2, 1, N_SOURCE, 3).
    # before_sampling_data = (normalized conditioning inputs [unused here], RAW inputs).
    x_source_raw = torch.randn(1, N_STEP_INPUT, 1, N_SOURCE, 3, generator=generator)
    before_sampling_data = ({"input": None}, {"input": x_source_raw})

    # Network output in MODEL_OUTPUT layout [u, v, dp, diag]
    model_output = torch.randn(1, N_STEP_OUTPUT, 1, N_TARGET, 4, generator=generator)

    out = model._after_sampling(
        {"output": model_output.clone()},
        post_processors,
        before_sampling_data,
        model_comm_group=None,
        grid_shard_sizes=None,
        gather_out=False,
        post_processors_residuals=post_processors_residuals,
    )["output"]

    assert out.shape == (1, N_STEP_OUTPUT, 1, N_TARGET, 4)

    # Hand-computed reference, independent of the model code path (raw reference, no source_stdev):
    #   interp_phys[c] = matrix @ x_source_raw[:, 1, ..., c] for c in [u, v]
    #   out[u] = model_output[u] * residual_stdev[u] + interp_phys[u]   (residual channels)
    #   out[dp] = model_output[dp] * state_stdev[dp]                    (direct prediction)
    #   out[diag] = model_output[diag] * state_stdev[diag]              (diagnostic)
    projection = torch.from_numpy(matrix)
    source_step = x_source_raw[0, 1, 0]  # (N_SOURCE, 3), the step selected by positions=[1]
    interp_u_phys = projection @ source_step[:, 0]
    interp_v_phys = projection @ source_step[:, 1]

    model_pos = {
        name: int(target_indices.model.output.name_to_index[name]) for name in ("u", "v", "dp", "diag")
    }
    state_stdev_by_name = {
        name: _TARGET_STATE_STDEV[_TARGET_NAME_TO_INDEX[name]] for name in ("dp", "diag")
    }
    residual_stdev_by_name = {
        name: _TARGET_RESIDUAL_STDEV[_TARGET_NAME_TO_INDEX[name]] for name in ("u", "v")
    }

    expected_u = model_output[0, 0, 0, :, model_pos["u"]] * residual_stdev_by_name["u"] + interp_u_phys
    expected_v = model_output[0, 0, 0, :, model_pos["v"]] * residual_stdev_by_name["v"] + interp_v_phys
    expected_dp = model_output[0, 0, 0, :, model_pos["dp"]] * state_stdev_by_name["dp"]
    expected_diag = model_output[0, 0, 0, :, model_pos["diag"]] * state_stdev_by_name["diag"]

    torch.testing.assert_close(out[0, 0, 0, :, model_pos["u"]], expected_u, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0, 0, 0, :, model_pos["v"]], expected_v, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0, 0, 0, :, model_pos["dp"]], expected_dp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0, 0, 0, :, model_pos["diag"]], expected_diag, atol=1e-5, rtol=1e-5)


def test_real_after_sampling_refuses_unset_reference_positions(
    real_model, index_collection_factory, statistics_factory
) -> None:
    model, _ = real_model
    _, post_src = _processor_pair(_source_indices(index_collection_factory), statistics_factory(_SOURCE_STATE_STDEV))
    _, post_tgt = _processor_pair(_target_indices(index_collection_factory), statistics_factory(_TARGET_STATE_STDEV))
    _, post_res = _processor_pair(
        _target_indices(index_collection_factory), statistics_factory(_TARGET_RESIDUAL_STDEV)
    )

    x_source_raw = torch.randn(1, N_STEP_INPUT, 1, N_SOURCE, 3)
    model_output = torch.randn(1, N_STEP_OUTPUT, 1, N_TARGET, 4)

    with pytest.raises(RuntimeError, match="unset"):
        model._after_sampling(
            {"output": model_output},
            torch.nn.ModuleDict({"input": post_src, "output": post_tgt}),
            ({"input": None}, {"input": x_source_raw}),
            model_comm_group=None,
            grid_shard_sizes=None,
            gather_out=False,
            post_processors_residuals=torch.nn.ModuleDict({"output": post_res}),
        )
