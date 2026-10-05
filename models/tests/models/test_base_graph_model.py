# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.models.base import BaseGraphModel


class DummyGraphModel(BaseGraphModel):
    def _build_networks(self, model_config) -> None:
        self.seen_hidden_name = model_config.model.model.hidden_nodes_name

    def _assemble_input(self, x, batch_size, grid_shard_sizes=None, model_comm_group=None):
        return x

    def _assemble_output(self, x_out, x_skip, batch_size, ensemble_size, dtype):
        return x_out

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _IndexGroup(SimpleNamespace):
    def __len__(self):
        return len(self.prognostic)


def _make_data_indices() -> dict:
    dataset_indices = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0]),
            output=_IndexGroup(prognostic=[0], full=[0], diagnostic=[], name_to_index={"var": 0}),
            _forcing=[],
        ),
        data=SimpleNamespace(
            input=SimpleNamespace(
                name_to_index={"var": 0},
            ),
        ),
        name_to_index={"var": 0},
    )
    return {"data": dataset_indices}


def _make_graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.zeros(2, 2)
    graph["data"].num_nodes = 2
    graph["hidden"].x = torch.zeros(1, 2)
    graph["hidden"].num_nodes = 1
    return graph


def _make_hierarchical_graph() -> HeteroData:
    graph = HeteroData()
    graph["data"].x = torch.zeros(2, 2)
    graph["data"].num_nodes = 2
    for hidden_name in ["hidden_1", "hidden_2", "hidden_3"]:
        graph[hidden_name].x = torch.zeros(1, 2)
        graph[hidden_name].num_nodes = 1
    return graph


def test_base_graph_model_builds_with_omegaconf_config() -> None:
    model_config = OmegaConf.create(
        {
            "model": {
                "num_channels": 8,
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                },
                "model": {
                    "hidden_nodes_name": "hidden",
                    "latent_skip": False,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                },
                "bounding": [],
            },
        },
    )

    model = DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_graph(),
    )

    assert model.seen_hidden_name == "hidden"
    assert "data" in model.residual


def test_base_graph_model_accepts_omegaconf_hidden_node_lists() -> None:
    model_config = OmegaConf.create(
        {
            "model": {
                "num_channels": 8,
                "trainable_parameters": {
                    "data": 0,
                    "hidden": 0,
                },
                "model": {
                    "hidden_nodes_name": ["hidden_1", "hidden_2", "hidden_3"],
                    "latent_skip": False,
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                },
                "bounding": [],
            },
        },
    )

    model = DummyGraphModel(
        model_config=model_config,
        data_indices=_make_data_indices(),
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=_make_hierarchical_graph(),
    )

    assert list(model.seen_hidden_name) == ["hidden_1", "hidden_2", "hidden_3"]
    assert model.node_attributes.num_nodes["hidden_3"] == 1


def _make_data_indices_with_decoder_forcing(n_df: int = 2) -> dict:
    dataset_indices = SimpleNamespace(
        model=SimpleNamespace(
            input=_IndexGroup(prognostic=[0]),
            output=_IndexGroup(prognostic=[0], full=[0], diagnostic=[], name_to_index={"var": 0}),
            _forcing=[],
        ),
        data=SimpleNamespace(
            input=SimpleNamespace(
                name_to_index={"var": 0},
                decoder_forcing=torch.arange(n_df, dtype=torch.int),
            ),
        ),
        name_to_index={"var": 0},
    )
    return {"data": dataset_indices}


def _build_dummy_model(data_indices: dict, n_step_output: int = 1) -> DummyGraphModel:
    model_config = OmegaConf.create(
        {
            "model": {
                "num_channels": 8,
                "trainable_parameters": {"data": 0, "hidden": 0},
                "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
                "residual": {"_target_": "anemoi.models.layers.residual.SkipConnection"},
                "bounding": [],
            },
        },
    )
    return DummyGraphModel(
        model_config=model_config,
        data_indices=data_indices,
        statistics={"data": None},
        n_step_input=1,
        n_step_output=n_step_output,
        graph_data=_make_graph(),
    )


def test_target_dim_without_decoder_forcing_equals_input_dim() -> None:
    model = _build_dummy_model(_make_data_indices())
    assert model.num_decoder_forcing_channels["data"] == 0
    assert model.target_dim["data"] == model.input_dim["data"]


def test_target_dim_includes_decoder_forcing_channels() -> None:
    n_df, n_step_output = 2, 3
    model = _build_dummy_model(_make_data_indices_with_decoder_forcing(n_df), n_step_output=n_step_output)
    assert model.num_decoder_forcing_channels["data"] == n_df
    assert model.target_dim["data"] == model.input_dim["data"] + n_step_output * n_df


# ── _append_decoder_forcings ──────────────────────────────────────────────────


def _df_stub(n_step_output: int = 2, n_df: int = 3) -> SimpleNamespace:
    """Exposes only what _append_decoder_forcings reads."""
    return SimpleNamespace(
        n_step_output=n_step_output,
        num_decoder_forcing_channels={"data": n_df},
    )


def _append(stub: SimpleNamespace, x_dst: torch.Tensor, forcings: dict | None, ensemble: int) -> torch.Tensor:
    return BaseGraphModel._append_decoder_forcings(stub, x_dst, forcings, "data", ensemble)


def test_append_decoder_forcings_is_a_noop_without_forcings() -> None:
    stub = _df_stub()
    x_dst = torch.randn(6, 4)
    assert _append(stub, x_dst, None, 1) is x_dst
    # A dict that does not mention this dataset is equally a no-op.
    assert _append(stub, x_dst, {"other": torch.randn(1, 2, 1, 3, 3)}, 1) is x_dst


def test_append_decoder_forcings_concatenates_time_major_channels() -> None:
    batch, n_step_output, grid, n_df, channels = 2, 2, 3, 3, 4
    stub = _df_stub(n_step_output=n_step_output, n_df=n_df)
    x_dst = torch.randn(batch * 1 * grid, channels)
    df = torch.randn(batch, n_step_output, 1, grid, n_df)

    out = _append(stub, x_dst, {"data": df}, 1)

    assert out.shape == (batch * grid, channels + n_step_output * n_df)
    assert torch.equal(out[:, :channels], x_dst)


def test_append_decoder_forcings_broadcasts_a_single_member() -> None:
    """Forcings describe the grid point and time, so every member gets the same values."""
    batch, n_step_output, grid, n_df, ensemble = 2, 2, 3, 3, 4
    stub = _df_stub(n_step_output=n_step_output, n_df=n_df)
    x_dst = torch.randn(batch * ensemble * grid, 5)
    df = torch.randn(batch, n_step_output, 1, grid, n_df)

    out = _append(stub, x_dst, {"data": df}, ensemble)

    assert out.shape == (batch * ensemble * grid, 5 + n_step_output * n_df)
    # Node order is (batch, ensemble, grid); members of the same sample must agree.
    appended = out[:, 5:].reshape(batch, ensemble, grid, -1)
    for member in range(1, ensemble):
        torch.testing.assert_close(appended[:, 0], appended[:, member])


def test_append_decoder_forcings_rejects_wrong_time_dim() -> None:
    stub = _df_stub(n_step_output=2, n_df=3)
    x_dst = torch.randn(6, 4)
    df = torch.randn(2, 3, 1, 3, 3)  # 3 output times, not 2
    with pytest.raises(AssertionError, match="decoder_forcings time dim"):
        _append(stub, x_dst, {"data": df}, 1)


def test_append_decoder_forcings_rejects_wrong_channel_count() -> None:
    stub = _df_stub(n_step_output=2, n_df=3)
    x_dst = torch.randn(6, 4)
    df = torch.randn(2, 2, 1, 3, 5)  # 5 channels, not 3
    with pytest.raises(AssertionError, match="decoder_forcings channels"):
        _append(stub, x_dst, {"data": df}, 1)
