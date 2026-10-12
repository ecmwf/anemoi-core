# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Transport models that read a dataset without predicting it.

A small transport model predicts a gridded dataset and reads an observation dataset as input only:
the observations have an encoder but no decoder. The observations are encoded from their history
alone, with no noisy target, and they still shape the grid prediction.
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.data import Batch
from anemoi.models.data import TensorLayout
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec
from anemoi.models.preprocessing import Processors
from anemoi.models.preprocessing.normalizer import InputNormalizer

NUM_GRID_POINTS = 60
OBS_ROWS_PER_WINDOW = (17, 21)  # the two input windows of the observation dataset
GRID_VARIABLES = ["a", "b", "f", "d"]  # "f" is a forcing, "d" a diagnostic
OBS_VARIABLES = ["o1", "o2", "fo"]  # "fo" is a forcing
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TABULAR_LAYOUT = TensorLayout(ensemble=0, grid=1, variables=2)
STATISTICS = {
    "grid": {"mean": np.array([1.0, -1.0, 0.5, 2.0]), "stdev": np.array([2.0, 0.5, 1.0, 3.0])},
    "obs": {"mean": np.array([0.5, 1.5, -0.5]), "stdev": np.array([1.5, 2.0, 1.0])},
}
# One output step; the values of the forcing "f" are never read.
TENDENCY_STATISTICS = {
    "grid": {
        "lead_times": ["6h"],
        "6h": {"mean": np.array([0.1, -0.2, 9.0, 9.0]), "stdev": np.array([0.3, 0.2, 9.0, 9.0])},
    },
}
MODEL_CLASSES = {"state": AnemoiTransportModelEncProcDec, "tendency": AnemoiTransportTendModelEncProcDec}


@pytest.fixture(autouse=True)
def _graphs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the graphs on the CPU, where the model and the data are, also on a GPU node."""
    monkeypatch.setenv("ANEMOI_GRAPHS_FORCE_CPU", "1")


def _points_on_sphere(num_points: int, generator: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes in degrees, spread evenly over the sphere."""
    latitudes = torch.rad2deg(torch.asin(2 * torch.rand(num_points, generator=generator) - 1))
    longitudes = 360 * torch.rand(num_points, generator=generator) - 180
    return latitudes.numpy(), longitudes.numpy()


def _graph_config(latitudes: np.ndarray, longitudes: np.ndarray) -> DictConfig:
    edge_attributes = {
        "edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": "unit-std"},
        "edge_dirs": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection", "norm": "unit-std"},
    }

    def edges(source, target, builder):
        return {"source_name": source, "target_name": target, "edge_builders": [builder], "attributes": edge_attributes}

    cutoff = {"_target_": "anemoi.graphs.edges.CutOffEdges", "cutoff_factor": 0.6}
    return OmegaConf.create(
        {
            "nodes": {
                "grid": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.LatLonNodes",
                        "latitudes": latitudes.tolist(),
                        "longitudes": longitudes.tolist(),
                    },
                },
                # Observation nodes change with every sample, so only their edges are configured.
                "obs": {"node_builder": {"_target_": "anemoi.graphs.nodes.LatLonNodes"}, "attributes": {}},
                "hidden": {"node_builder": {"_target_": "anemoi.graphs.nodes.TriNodes", "resolution": 1}},
            },
            "edges": [
                edges("grid", "hidden", cutoff),
                edges("obs", "hidden", cutoff),
                edges(
                    "hidden",
                    "hidden",
                    {"_target_": "anemoi.graphs.edges.MultiScaleEdges", "x_hops": 1, "scale_resolutions": 1},
                ),
                edges("hidden", "grid", {"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 3}),
            ],
        },
    )


def _model_config() -> DictConfig:
    channels = 16
    block = {
        "num_channels": channels,
        "sub_graph_edge_attributes": ["edge_length", "edge_dirs"],
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "num_heads": 2,
        "qk_norm": False,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": {
            "LayerNorm": {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "normalized_shape": channels,
                "condition_shape": 4,
                # Non-zero, so the noise conditioning reaches the output from the first step.
                "zero_init": False,
                "autocast": False,
            },
            "Linear": {"_target_": "torch.nn.Linear"},
            "Activation": {"_target_": "torch.nn.GELU"},
        },
        # The Triton kernels need a GPU.
        "graph_attention_backend": "pyg",
        "edge_pre_mlp": False,
    }

    def mapper(target: str, **extra) -> dict:
        return {"_target_": f"anemoi.models.layers.mapper.{target}", "trainable_size": 0, **block, **extra}

    return OmegaConf.create(
        {
            "model": {
                "hidden_nodes_name": "hidden",
                "latent_skip": True,
                "transport": {
                    "objective": "edm_diffusion",
                    "sigma_data": 1.0,
                    "noise_channels": 8,
                    "noise_cond_dim": 4,
                    "sigma_max": 100.0,
                    "sigma_min": 0.02,
                    "rho": 7.0,
                    "noise_embedder": {
                        "_target_": "anemoi.models.layers.diffusion.SinusoidalEmbeddings",
                        "num_channels": 8,
                        "max_period": 1000,
                    },
                    "inference_defaults": {
                        "sampling_schedule": {
                            "schedule_type": "karras",
                            "sigma_max": 100.0,
                            "sigma_min": 0.02,
                            "rho": 7.0,
                            "num_steps": 3,
                        },
                        "sampler": {"sampler": "heun"},
                    },
                },
            },
            "condition_on_residual": True,
            "processor": {
                "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
                "trainable_size": 0,
                "num_layers": 1,
                **block,
            },
            "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
            "encoders": {
                name: {
                    "source_datasets": [name],
                    "dataset_fusing_strategy": "none",
                    "mapper": mapper("GraphTransformerForwardMapper"),
                }
                for name in ("grid", "obs")
            },
            # Only the grid is predicted.
            "decoders": {
                "grid": {
                    "target_datasets": ["grid"],
                    "target_node_features": ["encoded_data", "target_forcings"],
                    "mapper": mapper("GraphTransformerBackwardMapper", initialise_data_extractor_zero=False),
                },
            },
            # The tendency model takes the latest grid state as its reference.
            "residual": {
                "datasets": {"grid": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}}
            },
            "node_trainable_parameters": {"grid": 0, "obs": 0, "hidden": 0},
            "bounding": {"datasets": {"grid": []}},
        },
    )


def _data_indices() -> dict[str, IndexCollection]:
    return {
        "grid": IndexCollection(
            DictConfig({"forcing": ["f"], "diagnostic": ["d"], "target": []}),
            {name: index for index, name in enumerate(GRID_VARIABLES)},
        ),
        "obs": IndexCollection(
            DictConfig({"forcing": ["fo"], "diagnostic": [], "target": []}),
            {name: index for index, name in enumerate(OBS_VARIABLES)},
        ),
    }


def _build_model(kind: str) -> AnemoiTransportModelEncProcDec:
    latitudes, longitudes = _points_on_sphere(NUM_GRID_POINTS, torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    return MODEL_CLASSES[kind](
        model_config=_model_config(),
        model_graph_config=_graph_config(latitudes, longitudes),
        data_indices=_data_indices(),
        statistics=STATISTICS,
        is_dataset_static={"grid": True, "obs": False},
        n_step_input={"grid": 2, "obs": 2},
        n_step_output={"grid": 1, "obs": 1},
    )


def _gridded(variables: list[str], data: torch.Tensor, coordinates: torch.Tensor) -> GriddedSource:
    positions = [GRID_VARIABLES.index(name) for name in variables]
    return GriddedSource(
        name="grid",
        variables=variables,
        layout=GRIDDED_LAYOUT,
        statistics={key: value[positions] for key, value in STATISTICS["grid"].items()},
        data=data,
        coordinates=coordinates,
    )


def _tabular(variables: list[str], windows: list[torch.Tensor], coordinates: list[torch.Tensor]) -> TabularSource:
    """One sample of observations, one ``(rows, variables)`` tensor per time window."""
    positions = [OBS_VARIABLES.index(name) for name in variables]
    starts = np.cumsum([0, *[window.shape[0] for window in windows]])
    return TabularSource(
        name="obs",
        variables=variables,
        layout=TABULAR_LAYOUT,
        statistics={key: value[positions] for key, value in STATISTICS["obs"].items()},
        data=[torch.cat(windows)[None]],
        coordinates=[torch.cat(coordinates)],
        timedeltas=[torch.cat([torch.full((window.shape[0],), float(step)) for step, window in enumerate(windows)])],
        boundaries=[tuple(slice(int(start), int(stop)) for start, stop in zip(starts[:-1], starts[1:]))],
    )


def _synthetic_batches(model: AnemoiTransportModelEncProcDec) -> dict[str, Batch]:
    """Inputs of both datasets, and the target forcings and a noisy target of the grid, for one sample."""
    generator = torch.Generator().manual_seed(2)
    grid_coordinates = model._graph_data["grid"].x.float()  # radians, as the graph stores them
    indices = _data_indices()

    def obs_coordinates(num_rows):
        latitudes, longitudes = _points_on_sphere(num_rows, generator)
        return torch.deg2rad(torch.stack([torch.as_tensor(latitudes), torch.as_tensor(longitudes)], dim=-1)).float()

    grid_inputs = list(indices["grid"].model.input.ordered_names)
    grid_outputs = list(indices["grid"].model.output.ordered_names)
    obs_inputs = list(indices["obs"].model.input.ordered_names)
    return {
        "x": Batch(
            {
                "grid": _gridded(
                    grid_inputs,
                    torch.randn(1, 2, 1, NUM_GRID_POINTS, len(grid_inputs), generator=generator),
                    grid_coordinates,
                ),
                "obs": _tabular(
                    obs_inputs,
                    [torch.randn(rows, len(obs_inputs), generator=generator) for rows in OBS_ROWS_PER_WINDOW],
                    [obs_coordinates(rows) for rows in OBS_ROWS_PER_WINDOW],
                ),
            },
        ),
        "target_forcing": Batch(
            {"grid": _gridded(["f"], torch.randn(1, 1, 1, NUM_GRID_POINTS, 1, generator=generator), grid_coordinates)},
        ),
        "noisy_target": Batch(
            {
                "grid": _gridded(
                    grid_outputs,
                    torch.randn(1, 1, 1, NUM_GRID_POINTS, len(grid_outputs), generator=generator),
                    grid_coordinates,
                ),
            },
        ),
    }


@pytest.mark.parametrize("kind", list(MODEL_CLASSES))
def test_input_only_dataset_is_encoded_from_its_history(kind: str) -> None:
    model = _build_model(kind)

    assert model.target_datasets == ["grid"]
    # The observations take the width the deterministic model gives them: no noisy-target
    # columns, no flag column, no residual columns.
    assert model.input_dim["obs"] == AnemoiModelEncProcDec._calculate_input_dim(model, "obs")
    assert model.target_dim["obs"] == 0


@pytest.mark.parametrize("kind", list(MODEL_CLASSES))
def test_input_only_observations_shape_the_grid_prediction(kind: str) -> None:
    model = _build_model(kind)
    batches = _synthetic_batches(model)
    sigma = torch.full((1, 1, 1, 1, 1), 0.7)

    out = model(batches["x"], batches["noisy_target"], {"grid": sigma}, target_forcing=batches["target_forcing"])

    assert list(out.keys()) == ["grid"]
    assert out["grid"].data.shape == batches["noisy_target"]["grid"].data.shape
    assert torch.isfinite(out["grid"].data).all()

    weights = torch.randn(out["grid"].data.shape, generator=torch.Generator().manual_seed(3))
    (out["grid"].data * weights).sum().backward()
    obs_gradients = [parameter.grad for parameter in model.encoder["obs"].parameters()]
    assert all(gradient is not None for gradient in obs_gradients)
    assert sum(gradient.abs().sum() for gradient in obs_gradients) > 0


def test_tendency_model_samples_grid_states_with_input_only_observations() -> None:
    model = _build_model("tendency")
    batches = _synthetic_batches(model)
    normalizer = InputNormalizer(config=DictConfig({"default": "mean-std"}))
    pre_processors = torch.nn.ModuleDict({name: Processors([["normalizer", normalizer]]) for name in ("grid", "obs")})
    post_processors = torch.nn.ModuleDict(
        {name: Processors([["normalizer", normalizer]], inverse=True) for name in ("grid", "obs")},
    )

    out = model.predict_step(
        batches["x"],
        target_template={"grid": batches["noisy_target"]["grid"].template()},
        pre_processors=pre_processors,
        post_processors=post_processors,
        n_step_input=model.n_step_input,
        target_forcing=batches["target_forcing"],
        statistics_tendencies=TENDENCY_STATISTICS,
    )

    assert list(out.keys()) == ["grid"]
    assert out["grid"].variables == list(_data_indices()["grid"].model.output.ordered_names)
    assert out["grid"].data.shape == (1, 1, 1, NUM_GRID_POINTS, 3)
    assert torch.isfinite(out["grid"].data).all()
