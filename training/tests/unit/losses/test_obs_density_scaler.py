# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.scalers import ObsDensityScaler
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.utils.enums import TensorDim
from anemoi.training.utils.index_space import IndexSpace
from anemoi.training.utils.masks import NoOutputMask

NODES = "data"
LATS = np.repeat([-60.0, -30.0, 0.0, 10.0, 30.0, 60.0], 2)
LONS = np.tile([0.0, 240.0], 6)
FILE_VARIABLES = ["t_500", "u_250", "q_850", "2t"]


@pytest.fixture
def graph() -> HeteroData:
    graph = HeteroData()
    graph[NODES].x = torch.deg2rad(torch.tensor(np.stack([LATS, LONS], axis=-1), dtype=torch.float32))
    graph[NODES].area_weight = torch.full((len(LATS), 1), 1.0 / len(LATS))
    return graph


@pytest.fixture
def data_indices() -> IndexCollection:
    config = DictConfig({"forcing": ["f"], "diagnostic": ["d"]})
    name_to_index = {"f": 0, "t_500": 1, "u_250": 2, "q_850": 3, "d": 4}
    return IndexCollection(data_config=config, name_to_index=name_to_index)


def write_weights(
    path: Path,
    weights: np.ndarray | None = None,
    variables: list[str] = FILE_VARIABLES,
    lats: np.ndarray = LATS,
    lons: np.ndarray = LONS,
    **extra,
) -> tuple[str, np.ndarray]:
    if weights is None:
        weights = np.random.default_rng(0).uniform(0.2, 4.0, size=(len(variables), len(lats))).astype(np.float32)
    np.savez(
        path,
        variables=np.array(variables),
        weights=weights,
        valid_fraction=np.full_like(weights, 0.1),
        latitudes=lats,
        longitudes=lons,
        mode="bands",
        alpha=np.float64(0.5),
        max_weight=np.float64(10.0),
        first_date="1981-01-01T00:00:00",
        last_date="2024-12-31T18:00:00",
        **extra,
    )
    return str(path), weights


def build_scaler(data_indices: IndexCollection, graph: HeteroData, weights_path: str, **kwargs) -> ObsDensityScaler:
    return ObsDensityScaler(
        data_indices=data_indices,
        graph_data=graph,
        nodes_name=NODES,
        weights_path=weights_path,
        **kwargs,
    )


def output_names(data_indices: IndexCollection) -> list[str]:
    index_to_name = {idx: name for name, idx in data_indices.data.output.name_to_index.items()}
    return [index_to_name[i] for i in data_indices.data.output.full.tolist()]


def test_columns_follow_data_output_and_default(
    tmp_path: Path,
    data_indices: IndexCollection,
    graph: HeteroData,
) -> None:
    path, weights = write_weights(tmp_path / "w.npz")
    dims, values = build_scaler(data_indices, graph, path, default_weight=0.5).get_scaling()

    assert dims == (TensorDim.GRID.value, TensorDim.VARIABLE.value)
    names = output_names(data_indices)
    assert values.shape == (len(LATS), len(names))
    for position, name in enumerate(names):
        expected = weights[FILE_VARIABLES.index(name)] if name in FILE_VARIABLES else np.full(len(LATS), 0.5)
        np.testing.assert_allclose(values[:, position].numpy(), expected)


def test_variable_patterns(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, weights = write_weights(tmp_path / "w.npz")
    values = build_scaler(data_indices, graph, path, variables=["t_*", "u_*"]).get_scaling_values()

    for position, name in enumerate(output_names(data_indices)):
        expected = weights[FILE_VARIABLES.index(name)] if name in ("t_500", "u_250") else np.ones(len(LATS))
        np.testing.assert_allclose(values[:, position].numpy(), expected)


def test_rename(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, weights = write_weights(tmp_path / "w.npz", variables=["t_500", "u_250", "q_850", "dd"])
    values = build_scaler(data_indices, graph, path, rename={"dd": "d"}).get_scaling_values()

    position = output_names(data_indices).index("d")
    np.testing.assert_allclose(values[:, position].numpy(), weights[3])


def test_target_category_variables(tmp_path: Path, graph: HeteroData) -> None:
    config = DictConfig({"forcing": ["f"], "diagnostic": ["d"], "target": ["obs"]})
    data_indices = IndexCollection(data_config=config, name_to_index={"f": 0, "obs": 1, "t_500": 2, "d": 3})
    path, weights = write_weights(tmp_path / "w.npz", variables=["obs", "t_500"])
    values = build_scaler(data_indices, graph, path).get_scaling_values()

    names = output_names(data_indices)
    assert "obs" in names
    assert values.shape == (len(LATS), len(names))
    np.testing.assert_allclose(values[:, names.index("obs")].numpy(), weights[0])
    np.testing.assert_allclose(values[:, names.index("t_500")].numpy(), weights[1])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"lats": LATS[:-1], "lons": LONS[:-1], "weights": np.ones((4, len(LATS) - 1), np.float32)}, ValueError),
        ({"lats": LATS + 1.0}, ValueError),
        ({"lons": LONS + 0.01}, ValueError),
        ({"weights": -np.ones((4, len(LATS)), np.float32)}, ValueError),
        ({"weights": np.full((4, len(LATS)), np.nan, np.float32)}, ValueError),
    ],
)
def test_invalid_file_raises(
    tmp_path: Path,
    data_indices: IndexCollection,
    graph: HeteroData,
    kwargs: dict,
    error: type[Exception],
) -> None:
    path, _ = write_weights(tmp_path / "w.npz", **kwargs)
    with pytest.raises(error):
        build_scaler(data_indices, graph, path)


def test_longitude_wrap_is_accepted(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, _ = write_weights(tmp_path / "w.npz", lons=np.where(LONS > 180, LONS - 360, LONS))
    build_scaler(data_indices, graph, path)


def test_coordinate_check_can_be_disabled(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, _ = write_weights(tmp_path / "w.npz", lats=LATS + 1.0)
    build_scaler(data_indices, graph, path, check_coordinates=False)


def test_missing_or_unreadable_file_raises(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    with pytest.raises(FileNotFoundError):
        build_scaler(data_indices, graph, str(tmp_path / "missing.npz"))
    bad = tmp_path / "bad.npz"
    bad.write_text("not an npz")
    with pytest.raises(ValueError, match="cannot read"):
        build_scaler(data_indices, graph, str(bad))


def test_norm_rejected(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, _ = write_weights(tmp_path / "w.npz")
    with pytest.raises(ValueError, match="norm"):
        build_scaler(data_indices, graph, path, norm="unit-sum")


def test_area_method_mismatch_warns(
    tmp_path: Path,
    data_indices: IndexCollection,
    graph: HeteroData,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path, _ = write_weights(tmp_path / "w.npz", areas_method="voronoi")
    with caplog.at_level(logging.WARNING):
        build_scaler(data_indices, graph, path)
    assert "areas=voronoi" in caplog.text


def build_loss(
    data_indices: IndexCollection,
    graph: HeteroData,
    weights_path: str | None,
) -> torch.nn.Module:
    builders = {
        "node_weights": {
            "_target_": "anemoi.training.losses.scalers.GraphNodeAttributeScaler",
            "nodes_attribute_name": "area_weight",
            "norm": "unit-sum",
        },
    }
    loss_scalers = ["node_weights"]
    if weights_path is not None:
        builders["obs_density"] = {
            "_target_": "anemoi.training.losses.scalers.ObsDensityScaler",
            "weights_path": weights_path,
        }
        loss_scalers.append("obs_density")
    scalers, _ = create_scalers(
        DictConfig(builders),
        data_indices=data_indices,
        graph_data=graph,
        nodes_name=NODES,
        output_mask=NoOutputMask(),
    )
    config = DictConfig({"_target_": "anemoi.training.losses.MSELoss", "scalers": loss_scalers, "ignore_nans": True})
    return get_loss_function(config, scalers=scalers, data_indices=data_indices)


def sample(data_indices: IndexCollection) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    n_var = len(data_indices.data.output.full)
    pred = torch.randn(2, 1, 1, len(LATS), n_var, generator=gen)
    target = torch.randn(2, 1, 1, len(LATS), n_var, generator=gen)
    target[torch.rand(target.shape, generator=gen) < 0.6] = torch.nan
    return pred, target


LAYOUTS = {"pred_layout": IndexSpace.MODEL_OUTPUT, "target_layout": IndexSpace.DATA_OUTPUT}


def test_loss_matches_manual(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, weights = write_weights(tmp_path / "w.npz")
    loss = build_loss(data_indices, graph, path)
    pred, target = sample(data_indices)

    out = loss(pred, target, squash=False, **LAYOUTS)

    names = output_names(data_indices)
    w = np.stack([weights[FILE_VARIABLES.index(n)] if n in FILE_VARIABLES else np.ones(len(LATS)) for n in names], -1)
    err2 = np.nan_to_num((pred - target).numpy() ** 2)  # NaN targets contribute zero
    area = np.full(len(LATS), 1.0 / len(LATS))
    expected = (err2 * area[:, None] * w).sum(axis=3).mean(axis=(0, 1, 2))
    np.testing.assert_allclose(out.detach().numpy(), expected, rtol=1e-5)


def test_grid_sharded_sum_equals_full(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, _ = write_weights(tmp_path / "w.npz")
    loss = build_loss(data_indices, graph, path)
    pred, target = sample(data_indices)

    full = loss(pred, target, squash=False, **LAYOUTS)
    shards = [slice(0, 5), slice(5, 9), slice(9, len(LATS))]
    sharded = sum(loss(pred[..., s, :], target[..., s, :], squash=False, grid_shard_slice=s, **LAYOUTS) for s in shards)
    torch.testing.assert_close(sharded, full)


def test_all_ones_file_is_identity(tmp_path: Path, data_indices: IndexCollection, graph: HeteroData) -> None:
    path, _ = write_weights(tmp_path / "w.npz", weights=np.ones((len(FILE_VARIABLES), len(LATS)), np.float32))
    pred, target = sample(data_indices)

    with_scaler = build_loss(data_indices, graph, path)(pred, target, **LAYOUTS)
    without = build_loss(data_indices, graph, None)(pred, target, **LAYOUTS)
    torch.testing.assert_close(with_scaler, without)
