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

- ``_calculate_input_dim`` has every attribute it needs at the point the parent
  ``__init__`` builds the encoders (construction-order contract),
- a full ``forward()`` pass runs through the concat conditioning
  (interp(source) ++ target forcings) and produces the right output shape,
- ``_after_sampling`` reconstructs physical states through real affine processors,
  matching a hand-computed reference.
"""

import numpy as np
import pytest
import scipy.sparse
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
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

# ── index collections (mirrors test_transport_residual_model.py) ─────────────
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


def _target_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": ["tgt"]})
    return IndexCollection(cfg, dict(_TARGET_NAME_TO_INDEX))


def _source_indices() -> IndexCollection:
    cfg = DictConfig({"forcing": ["sforce"], "diagnostic": [], "target": []})
    return IndexCollection(cfg, dict(_SOURCE_NAME_TO_INDEX))


def _norm_stats(stdev: list[float]) -> dict:
    n = len(stdev)
    return {
        "mean": np.zeros(n, dtype=np.float32),
        "stdev": np.asarray(stdev, dtype=np.float32),
        "minimum": np.zeros(n, dtype=np.float32),
        "maximum": np.ones(n, dtype=np.float32),
    }


def _processor_pair(indices: IndexCollection, stats: dict) -> tuple[Processors, Processors]:
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
def interpolation_npz(tmp_path) -> tuple[str, np.ndarray]:
    matrix = _interpolation_matrix()
    path = tmp_path / "source_to_target.npz"
    scipy.sparse.save_npz(path, scipy.sparse.csr_matrix(matrix))
    return str(path), matrix


def _coords(n: int, offset: float) -> torch.Tensor:
    lats = torch.linspace(-1.0, 1.0, n) + offset
    lons = torch.linspace(0.0, 2.0, n) - offset
    return torch.stack((lats, lons), dim=-1).float()


def _dense_bipartite_edges(n_src: int, n_dst: int) -> torch.Tensor:
    src = torch.arange(n_src).repeat_interleave(n_dst)
    dst = torch.arange(n_dst).repeat(n_src)
    return torch.stack((src, dst), dim=0)


def _graph() -> HeteroData:
    graph = HeteroData()
    for name, n, offset in (("input", N_SOURCE, 0.0), ("output", N_TARGET, 0.1), ("hidden", N_HIDDEN, 0.2)):
        graph[name].x = _coords(n, offset)
        graph[name].num_nodes = n

    sizes = {"input": N_SOURCE, "output": N_TARGET, "hidden": N_HIDDEN}
    pairs = [
        ("input", "hidden"),
        ("output", "hidden"),
        ("hidden", "input"),
        ("hidden", "output"),
        ("hidden", "hidden"),
    ]
    generator = torch.Generator().manual_seed(7)
    for src, dst in pairs:
        edge_index = _dense_bipartite_edges(sizes[src], sizes[dst])
        store = graph[(src, "to", dst)]
        store.edge_index = edge_index
        store.edge_length = torch.rand((edge_index.shape[1], 1), generator=generator)
    return graph


def _layer_kernels() -> dict:
    return {
        "LayerNorm": {
            "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
            "normalized_shape": NUM_CHANNELS,
            "condition_shape": NOISE_COND_DIM,
            "zero_init": True,
            "autocast": False,
        },
        "Linear": {"_target_": "torch.nn.Linear"},
        "Activation": {"_target_": "torch.nn.GELU"},
        "QueryNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "bias": False,
        },
        "KeyNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "bias": False,
        },
    }


def _model_config(interpolation_file_path: str) -> DictConfig:
    layer_kernels = _layer_kernels()
    mapper_common = {
        "trainable_size": 0,
        "sub_graph_edge_attributes": ["edge_length"],
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "mlp_implementation": "mlp",
        "num_heads": 4,
        "qk_norm": True,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": layer_kernels,
        "shard_strategy": "edges",
        "graph_attention_backend": "pyg",
    }
    return DictConfig(
        {
            "model": {
                "num_channels": NUM_CHANNELS,
                "cpu_offload": False,
                "keep_batch_sharded": False,
                "model": {
                    "hidden_nodes_name": "hidden",
                    "latent_skip": True,
                    "residual_prediction": {"output": "input"},
                    "direct_prediction": {"output": ["dp"]},
                    "transport": {
                        "objective": "edm_diffusion",
                        "sigma_data": 1.0,
                        "noise_channels": NOISE_CHANNELS,
                        "noise_cond_dim": NOISE_COND_DIM,
                        "noise_embedder": {
                            "_target_": "anemoi.models.layers.diffusion.SinusoidalEmbeddings",
                            "num_channels": NOISE_CHANNELS,
                            "max_period": 1000,
                        },
                    },
                },
                "layer_kernels": layer_kernels,
                "encoder": {
                    "_target_": "anemoi.models.layers.mapper.GraphTransformerForwardMapper",
                    **mapper_common,
                },
                "decoder": {
                    "_target_": "anemoi.models.layers.mapper.GraphTransformerBackwardMapper",
                    "initialise_data_extractor_zero": False,
                    **mapper_common,
                },
                "processor": {
                    "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
                    "num_layers": 1,
                    **mapper_common,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.InterpolationConnection",
                    "interpolation_file_path": interpolation_file_path,
                },
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                    "data2hidden": 0,
                    "hidden2data": 0,
                    "hidden2hidden": 0,
                },
                "attributes": {"edges": ["edge_length"], "nodes": []},
                "bounding": [],
            },
        },
    )


@pytest.fixture
def real_model(interpolation_npz) -> tuple[AnemoiTransportResidualModelEncProcDec, np.ndarray]:
    interpolation_file_path, matrix = interpolation_npz
    data_indices = {"input": _source_indices(), "output": _target_indices()}
    statistics = {
        "input": _norm_stats(_SOURCE_STATE_STDEV),
        "output": _norm_stats(_TARGET_STATE_STDEV),
    }
    model = AnemoiTransportResidualModelEncProcDec(
        model_config=_model_config(interpolation_file_path),
        data_indices=data_indices,
        statistics=statistics,
        n_step_input=N_STEP_INPUT,
        n_step_output=N_STEP_OUTPUT,
        graph_data=_graph(),
    )
    return model, matrix


def test_real_instantiation_builds_residual_interpolation(real_model) -> None:
    model, _ = real_model
    assert isinstance(model.residual["output"], InterpolationConnection)
    assert model._residual_pairs == {"output": "input"}
    # The -1 sentinel is registered until a residual training mode sets the alignment.
    assert model._output_reference_positions.tolist() == [-1] * N_STEP_OUTPUT


def test_real_instantiation_encoder_input_dim_matches_hand_formula(real_model) -> None:
    model, _ = real_model
    source_input_full = len(model.data_indices["input"].data.input.full)  # u, v, sforce -> 3
    target_forcing = len(model.data_indices["output"].data.input.forcing)  # force -> 1
    node_attr_dims = model.node_attributes.attr_ndims["output"]  # 2 * coord dims, no trainables -> 4
    output_channels = len(model.data_indices["output"].model.output.full)  # u, v, dp, diag -> 4

    assert source_input_full == 3
    assert target_forcing == 1
    assert node_attr_dims == 4
    assert output_channels == 4

    expected = N_STEP_INPUT * (source_input_full + target_forcing) + node_attr_dims + N_STEP_OUTPUT * output_channels
    assert model.input_dim["output"] == expected
    # The real encoder was constructed with that dim (this is what a stub test cannot check).
    assert model.encoder["output"].in_channels_src == expected


def _forward_inputs(batch_size: int = 2) -> tuple[dict, dict, dict]:
    generator = torch.Generator().manual_seed(11)
    x = {
        # source history: (batch, time, ensemble, source grid, source data.input.full)
        "input": torch.randn(batch_size, N_STEP_INPUT, 1, N_SOURCE, 3, generator=generator),
        # target history: (batch, time, ensemble, target grid, target data.input.full = [u, v, dp, force])
        "output": torch.randn(batch_size, N_STEP_INPUT, 1, N_TARGET, 4, generator=generator),
    }
    conditioned_target = {
        "output": torch.randn(batch_size, N_STEP_OUTPUT, 1, N_TARGET, 4, generator=generator),
    }
    condition = {"output": torch.full((batch_size, 1, 1, 1, 1), 0.5)}
    return x, conditioned_target, condition


def test_real_forward_pass_shape_and_finiteness(real_model) -> None:
    model, _ = real_model
    x, conditioned_target, condition = _forward_inputs()

    out = model.forward(x, conditioned_target, condition)

    assert set(out.keys()) == {"output"}
    assert out["output"].shape == (2, N_STEP_OUTPUT, 1, N_TARGET, 4)
    assert torch.isfinite(out["output"]).all()


def test_real_forward_requires_source_and_target_inputs(real_model) -> None:
    model, _ = real_model
    x, conditioned_target, condition = _forward_inputs()

    with pytest.raises(KeyError, match="requires source dataset"):
        model.forward({"output": x["output"]}, conditioned_target, condition)
    with pytest.raises(KeyError, match="requires its own dataset"):
        model.forward({"input": x["input"]}, conditioned_target, condition)


def test_real_after_sampling_reconstructs_residual_channels_by_hand(real_model) -> None:
    model, matrix = real_model
    target_indices = model.data_indices["output"]

    pre_src, post_src = _processor_pair(_source_indices(), _norm_stats(_SOURCE_STATE_STDEV))
    _, post_tgt = _processor_pair(_target_indices(), _norm_stats(_TARGET_STATE_STDEV))
    _, post_res = _processor_pair(_target_indices(), _norm_stats(_TARGET_RESIDUAL_STDEV))
    del pre_src

    post_processors = torch.nn.ModuleDict({"input": post_src, "output": post_tgt})
    post_processors_residuals = torch.nn.ModuleDict({"output": post_res})

    # The training mode persists the same-offset alignment; emulate it before reconstruction.
    model.set_output_reference_positions([1])

    generator = torch.Generator().manual_seed(23)
    # NORMALIZED source history (as produced by _before_sampling): (1, 2, 1, N_SOURCE, 3)
    x_source_norm = torch.randn(1, N_STEP_INPUT, 1, N_SOURCE, 3, generator=generator)
    before_sampling_data = ({"input": x_source_norm, "output": None},)

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

    # Hand-computed reference, independent of the model code path:
    #   interp_phys[c] = matrix @ (x_source_norm[:, 1, ..., c] * source_stdev[c]) for c in [u, v]
    #   out[u] = model_output[u] * residual_stdev[u] + interp_phys[u]   (residual channels)
    #   out[dp] = model_output[dp] * state_stdev[dp]                    (direct prediction)
    #   out[diag] = model_output[diag] * state_stdev[diag]              (diagnostic)
    projection = torch.from_numpy(matrix)
    source_step = x_source_norm[0, 1, 0]  # (N_SOURCE, 3), the step selected by positions=[1]
    interp_u_phys = projection @ (source_step[:, 0] * _SOURCE_STATE_STDEV[0])
    interp_v_phys = projection @ (source_step[:, 1] * _SOURCE_STATE_STDEV[1])

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


def test_real_after_sampling_refuses_unset_reference_positions(real_model) -> None:
    model, _ = real_model
    _, post_src = _processor_pair(_source_indices(), _norm_stats(_SOURCE_STATE_STDEV))
    _, post_tgt = _processor_pair(_target_indices(), _norm_stats(_TARGET_STATE_STDEV))
    _, post_res = _processor_pair(_target_indices(), _norm_stats(_TARGET_RESIDUAL_STDEV))

    x_source_norm = torch.randn(1, N_STEP_INPUT, 1, N_SOURCE, 3)
    model_output = torch.randn(1, N_STEP_OUTPUT, 1, N_TARGET, 4)

    with pytest.raises(RuntimeError, match="unset"):
        model._after_sampling(
            {"output": model_output},
            torch.nn.ModuleDict({"input": post_src, "output": post_tgt}),
            ({"input": x_source_norm, "output": None},),
            model_comm_group=None,
            grid_shard_sizes=None,
            gather_out=False,
            post_processors_residuals=torch.nn.ModuleDict({"output": post_res}),
        )
