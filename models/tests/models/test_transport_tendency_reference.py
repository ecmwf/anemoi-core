# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tendency transport models that take their reference from another dataset.

A small model predicts ``hres`` and reads ``lres`` as input only, on the same grid (as after a
spatial preprocessor has projected ``lres`` onto the ``hres`` grid). ``hres`` learns its outputs
``a`` and ``b`` as residuals against the ``lres`` input of the same name, and ``s``, which ``lres``
does not have, as a state. With several snapshots, each output step is taken against the ``lres``
input at the same time.
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.utils.graphs import build_graph
from anemoi.utils.config import DotDict
from tests.models.test_transport_input_only_datasets import _model_config as _input_only_model_config
from tests.models.test_transport_input_only_datasets import _points_on_sphere

NUM_GRID_POINTS = 40
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
VARIABLES = {"lres": ["a", "b", "c"], "hres": ["a", "b", "s", "f"]}
STATISTICS = {
    "lres": {"mean": np.array([10.0, -3.0, 1.0]), "stdev": np.array([4.0, 0.5, 2.0])},
    "hres": {"mean": np.array([1.0, -1.0, 0.5, 2.0]), "stdev": np.array([2.0, 0.25, 1.0, 3.0])},
}
# The values of "s" (learned as a state) and "f" (a forcing) are never read, so they can be missing.
RESIDUAL_STATISTICS = {"mean": np.array([0.1, -0.2, np.nan, np.nan]), "stdev": np.array([0.3, 0.2, np.nan, np.nan])}
TENDENCY_STATISTICS = {"hres": {"lead_times": ["0h"], "0h": RESIDUAL_STATISTICS}}


@pytest.fixture(autouse=True)
def _graphs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the graphs on the CPU, where the model and the data are, also on a GPU node."""
    monkeypatch.setenv("ANEMOI_GRAPHS_FORCE_CPU", "1")


def _graph_config(latitudes: np.ndarray, longitudes: np.ndarray) -> DictConfig:
    edge_attributes = {
        "edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": "unit-std"},
        "edge_dirs": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection", "norm": "unit-std"},
    }

    def edges(source, target, builder):
        return {"source_name": source, "target_name": target, "edge_builders": [builder], "attributes": edge_attributes}

    def grid_nodes():
        return {
            "node_builder": {
                "_target_": "anemoi.graphs.nodes.LatLonNodes",
                "latitudes": latitudes.tolist(),
                "longitudes": longitudes.tolist(),
            },
        }

    cutoff = {"_target_": "anemoi.graphs.edges.CutOffEdges", "cutoff_factor": 0.6}
    return OmegaConf.create(
        {
            "nodes": {
                "lres": grid_nodes(),
                "hres": grid_nodes(),
                "hidden": {"node_builder": {"_target_": "anemoi.graphs.nodes.TriNodes", "resolution": 1}},
            },
            "edges": [
                edges("lres", "hidden", cutoff),
                edges("hres", "hidden", cutoff),
                edges(
                    "hidden",
                    "hidden",
                    {"_target_": "anemoi.graphs.edges.MultiScaleEdges", "x_hops": 1, "scale_resolutions": 1},
                ),
                edges("hidden", "hres", {"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 3}),
            ],
        },
    )


def _model_config(**overrides) -> DictConfig:
    config = _input_only_model_config()
    mapper = config.encoders.grid.mapper
    decoder = config.decoders.grid
    config.condition_on_residual = False
    config.encoders = {
        name: {"source_datasets": [name], "dataset_fusing_strategy": "none", "mapper": mapper}
        for name in ("lres", "hres")
    }
    config.decoders = {"hres": {**decoder, "target_datasets": ["hres"]}}
    config.residual = {"datasets": {"lres": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}}}
    config.residual_reference = {"hres": "lres"}
    config.node_trainable_parameters = {"lres": 0, "hres": 0, "hidden": 0}
    config.bounding = {"datasets": {"hres": []}}
    for key, value in overrides.items():
        config[key] = value
    return config


def _data_indices() -> dict[str, IndexCollection]:
    return {
        "lres": IndexCollection(
            DictConfig({"forcing": [], "diagnostic": [], "target": []}),
            {name: index for index, name in enumerate(VARIABLES["lres"])},
        ),
        "hres": IndexCollection(
            DictConfig({"forcing": ["f"], "diagnostic": ["a", "b", "s"], "target": []}),
            {name: index for index, name in enumerate(VARIABLES["hres"])},
        ),
    }


def _build_model(
    n_step_input: dict[str, int] | None = None, n_step_output: dict[str, int] | None = None, **overrides
) -> AnemoiTransportTendModelEncProcDec:
    latitudes, longitudes = _points_on_sphere(NUM_GRID_POINTS, torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    static_graph_data, dynamic_graph_config = build_graph(
        DotDict(_graph_config(latitudes, longitudes)),
        is_dataset_static={"lres": True, "hres": True},
        hidden_nodes_name="hidden",
    )
    model = AnemoiTransportTendModelEncProcDec(
        model_config=_model_config(**overrides),
        static_graph_data=static_graph_data,
        dynamic_graph_config=dynamic_graph_config,
        data_indices=_data_indices(),
        statistics=STATISTICS,
        n_step_input=n_step_input or {"lres": 1, "hres": 1},
        n_step_output=n_step_output or {"lres": 1, "hres": 1},
    )
    # The model does not keep the graph; the synthetic batches need its grid coordinates.
    model.test_grid_coordinates = static_graph_data["hres"].x.float()
    return model


def _gridded(dataset: str, variables: list[str], data: torch.Tensor, coordinates: torch.Tensor) -> GriddedSource:
    positions = [VARIABLES[dataset].index(name) for name in variables]
    return GriddedSource(
        name=dataset,
        variables=variables,
        layout=GRIDDED_LAYOUT,
        statistics={key: value[positions] for key, value in STATISTICS[dataset].items()},
        values=data,
        coordinates=coordinates,
    )


def _synthetic_batches(model: AnemoiTransportTendModelEncProcDec, n_step: int = 1) -> dict[str, Batch]:
    generator = torch.Generator().manual_seed(2)
    coordinates = model.test_grid_coordinates
    indices = _data_indices()

    def random_source(dataset: str, names: list[str], n_step: int) -> GriddedSource:
        data = torch.randn(1, n_step, 1, NUM_GRID_POINTS, len(names), generator=generator)
        return _gridded(dataset, names, data, coordinates)

    hres_outputs = list(indices["hres"].model.output.ordered_names)
    return {
        "x": Batch(
            {
                "lres": random_source("lres", list(indices["lres"].model.input.ordered_names), n_step),
                "hres": random_source("hres", list(indices["hres"].model.input.ordered_names), n_step),
            },
        ),
        "target_forcing": Batch({"hres": random_source("hres", ["f"], n_step)}),
        "state": Batch({"hres": random_source("hres", hres_outputs, n_step)}),
    }


def _processors() -> tuple[torch.nn.ModuleDict, torch.nn.ModuleDict]:
    """``lres`` leaves ``a`` unnormalised; everything else is normalised with mean and stdev."""
    configs = {"lres": {"default": "mean-std", "none": ["a"]}, "hres": {"default": "mean-std"}}
    normalizers = {name: InputNormalizer(config=DictConfig(config)) for name, config in configs.items()}
    pre = torch.nn.ModuleDict({name: Processors([["normalizer", norm]]) for name, norm in normalizers.items()})
    post = torch.nn.ModuleDict(
        {name: Processors([["normalizer", norm]], inverse=True) for name, norm in normalizers.items()}
    )
    return pre, post


def test_outputs_shared_with_the_reference_dataset_are_residuals() -> None:
    model = _build_model()

    assert model.residual_reference == {"hres": "lres"}
    assert model.residual_variables("hres") == ["a", "b"]


def test_reference_is_the_latest_reference_input_in_its_own_physical_units() -> None:
    model = _build_model()
    batches = _synthetic_batches(model)
    _, post_processors = _processors()

    reference = model.reference_state(batches["x"], None, None, post_processors)

    assert list(reference.keys()) == ["hres"]
    assert reference["hres"].variables == ["a", "b"]
    lres = batches["x"]["lres"].values
    # "a" is not normalised in lres; "b" is normalised with lres's statistics, not with hres's.
    torch.testing.assert_close(reference["hres"].values[..., 0], lres[:, -1:, ..., 0])
    torch.testing.assert_close(reference["hres"].values[..., 1], lres[:, -1:, ..., 1] * 0.5 - 3.0)


def test_residuals_are_taken_against_the_reference_dataset() -> None:
    model = _build_model()
    batches = _synthetic_batches(model)
    pre_processors, post_processors = _processors()
    per_step = model.tendency_statistics("hres", TENDENCY_STATISTICS["hres"])
    reference = model.reference_state(batches["x"], None, None, post_processors)["hres"]
    state = batches["state"]["hres"]

    tendency = model.compute_tendency(
        "hres", state, reference, per_step, pre_processors["hres"], post_processors["hres"]
    )

    physical_state = post_processors["hres"](state, in_place=False).values
    for position, name in enumerate(["a", "b"]):
        index = state.name_to_index[name]
        expected = (
            physical_state[..., index] - reference.values[..., position] - RESIDUAL_STATISTICS["mean"][position]
        ) / (RESIDUAL_STATISTICS["stdev"][position])
        torch.testing.assert_close(tendency.values[..., index], expected.float())
    # "s" has no reference and stays a normalised state.
    index = state.name_to_index["s"]
    torch.testing.assert_close(tendency.values[..., index], state.values[..., index], rtol=0, atol=0)

    rebuilt = model.add_tendency_to_state(
        "hres", reference, tendency, per_step, post_processors["hres"], pre_processors=pre_processors["hres"]
    )
    torch.testing.assert_close(rebuilt.values, state.values)


def test_predict_step_adds_the_sampled_residual_to_the_reference_dataset() -> None:
    model = _build_model()
    batches = _synthetic_batches(model)
    pre_processors, post_processors = _processors()
    template = {"hres": batches["state"]["hres"].template()}
    seen = []
    add_tendency_to_state = model.add_tendency_to_state

    def spy(dataset_name, reference, tendency, *args, **kwargs):
        seen.append((dataset_name, reference.variables, reference.values.clone()))
        return add_tendency_to_state(dataset_name, reference, tendency, *args, **kwargs)

    model.add_tendency_to_state = spy

    out = model.predict_step(
        batches["x"],
        target_template=template,
        pre_processors=pre_processors,
        post_processors=post_processors,
        target_forcing=batches["target_forcing"],
        statistics_tendencies=TENDENCY_STATISTICS,
    )

    assert list(out.keys()) == ["hres"]
    assert out["hres"].variables == ["a", "b", "s"]
    assert torch.isfinite(out["hres"].values).all()
    # predict_step takes inputs in physical units, so the reference is the latest lres input of a and b.
    assert [(name, variables) for name, variables, _ in seen] == [("hres", ["a", "b"])]
    torch.testing.assert_close(seen[0][2], batches["x"]["lres"].values[:, -1:, ..., :2])


def _build_matching_model(n_step_input_lres: int = 2) -> AnemoiTransportTendModelEncProcDec:
    """Two snapshots: each output step of ``hres`` is taken against the ``lres`` input at the same time."""
    return _build_model(
        n_step_input={"lres": n_step_input_lres, "hres": 2},
        n_step_output={"lres": 2, "hres": 2},
        residual={
            "datasets": {"lres": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": "matching"}}
        },
    )


# The second snapshot gets other statistics, so pairing it with the wrong statistics would show.
SECOND_STEP_STATISTICS = {"mean": np.array([-0.4, 0.3, np.nan, np.nan]), "stdev": np.array([0.7, 0.1, np.nan, np.nan])}
TWO_STEP_STATISTICS = {"lead_times": ["0h", "6h"], "0h": RESIDUAL_STATISTICS, "6h": SECOND_STEP_STATISTICS}


def test_matching_reference_keeps_one_reference_step_per_output_step() -> None:
    model = _build_matching_model()
    batches = _synthetic_batches(model, n_step=2)
    pre_processors, post_processors = _processors()
    per_step = model.tendency_statistics("hres", TWO_STEP_STATISTICS)
    reference = model.reference_state(batches["x"], None, None, post_processors)["hres"]
    lres = batches["x"]["lres"].values

    assert reference.time_size == 2
    torch.testing.assert_close(reference.values[..., 0], lres[..., 0])
    torch.testing.assert_close(reference.values[..., 1], lres[..., 1] * 0.5 - 3.0)

    state = batches["state"]["hres"]
    tendency = model.compute_tendency(
        "hres", state, reference, per_step, pre_processors["hres"], post_processors["hres"]
    )

    physical_state = post_processors["hres"](state, in_place=False).values
    for step, statistics in enumerate((RESIDUAL_STATISTICS, SECOND_STEP_STATISTICS)):
        for position, name in enumerate(["a", "b"]):
            index = state.name_to_index[name]
            expected = (
                physical_state[:, step, ..., index]
                - reference.values[:, step, ..., position]
                - statistics["mean"][position]
            ) / statistics["stdev"][position]
            torch.testing.assert_close(tendency.values[:, step, ..., index], expected.float())

    rebuilt = model.add_tendency_to_state(
        "hres", reference, tendency, per_step, post_processors["hres"], pre_processors=pre_processors["hres"]
    )
    torch.testing.assert_close(rebuilt.values, state.values)


def test_predict_step_with_a_matching_reference_adds_each_residual_to_its_snapshot() -> None:
    model = _build_matching_model()
    batches = _synthetic_batches(model, n_step=2)
    pre_processors, post_processors = _processors()
    seen = []
    add_tendency_to_state = model.add_tendency_to_state

    def spy(dataset_name, reference, tendency, *args, **kwargs):
        seen.append(reference.values.clone())
        return add_tendency_to_state(dataset_name, reference, tendency, *args, **kwargs)

    model.add_tendency_to_state = spy

    out = model.predict_step(
        batches["x"],
        target_template={"hres": batches["state"]["hres"].template()},
        pre_processors=pre_processors,
        post_processors=post_processors,
        target_forcing=batches["target_forcing"],
        statistics_tendencies={"hres": TWO_STEP_STATISTICS},
    )

    assert out["hres"].time_size == 2
    assert torch.isfinite(out["hres"].values).all()
    torch.testing.assert_close(seen[0], batches["x"]["lres"].values[..., :2])


def test_several_output_steps_need_a_matching_reference() -> None:
    with pytest.raises(ValueError, match="set model.residual.datasets.lres to a SkipConnection with step: matching"):
        _build_model(n_step_input={"lres": 2, "hres": 2}, n_step_output={"lres": 2, "hres": 2})


def test_matching_reference_needs_one_reference_input_step_per_output_step() -> None:
    with pytest.raises(ValueError, match="has 2 output steps and 'lres' has 1 input steps"):
        _build_matching_model(n_step_input_lres=1)


def test_missing_residual_statistics_are_rejected() -> None:
    model = _build_model()
    statistics = {"lead_times": ["0h"], "0h": {**RESIDUAL_STATISTICS, "stdev": np.array([0.3, np.nan, 1.0, 1.0])}}

    with pytest.raises(ValueError, match=r"'stdev'.*not finite for variables \['b'\]"):
        model.tendency_statistics("hres", statistics)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "residual": {
                    "datasets": {
                        name: {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}
                        for name in ("lres", "hres")
                    }
                }
            },
            "nothing uses model.residual.datasets.hres",
        ),
        ({"residual": {"datasets": {}}}, "needs a residual connection: add model.residual.datasets.lres"),
        ({"condition_on_residual": True}, "cannot be combined with tendency references from other datasets"),
        ({"residual_reference": {"lres": "hres"}}, r"datasets the model does not predict: \['lres'\]"),
        ({"residual_reference": {"hres": "other"}}, "'other', which is not a dataset of the model"),
    ],
)
def test_invalid_references_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build_model(**overrides)


def test_reference_dataset_without_shared_variables_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(VARIABLES, "lres", ["x", "y", "z"])

    with pytest.raises(ValueError, match="none of its output variables is an input of 'lres'"):
        _build_model()
