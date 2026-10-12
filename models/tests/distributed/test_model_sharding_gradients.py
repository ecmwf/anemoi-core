# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Whole models give the same outputs and gradients whether or not they are sharded across a model group.

This is a temporary test to support the development of model sharding for the richer-batch models.
It builds whole models from a hand-written configuration, so it is expected to be replaced or removed
once that work settles.

A small deterministic model and a small EDM transport model, each with a gridded dataset and an
observation dataset, run once on the whole batch without communication and once sharded across the
process group, in float64 on CPU. Two further transport models, one predicting states and one predicting
tendencies, predict the grid only and read the observations as input only. The sharded gradients are
combined the way training combines them: the strategy's hooks multiply every gradient by the group size
except those of parameters named ``trainable`` or ``no_gradscaling``, and DDP then averages over the
ranks.

A sharded batch reaches the observation graph in a different row order, so the per-batch graph must
not depend on row order. Each model runs with pyg-lib's neighbour search and a cap above the number of
observations, and with the scikit-learn search, which keeps the nearest neighbours, and a cap small
enough that observations are dropped.
"""

import os
from copy import deepcopy

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
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportModelEncProcDec
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportTendModelEncProcDec

from ._distributed_runner import _run_distributed_test

DTYPE = torch.float64
NUM_GRID_POINTS = 80
OBS_ROWS_PER_WINDOW = (23, 29)  # the two input windows of the observation dataset
OBS_TARGET_ROWS = 26
GRID_VARIABLES = ["a", "b", "f", "d"]  # "f" is a forcing, "d" a diagnostic
OBS_VARIABLES = ["o1", "o2", "fo"]  # "fo" is a forcing
GRIDDED_LAYOUT = TensorLayout(batch=0, time=1, ensemble=2, grid=3, variables=4)
TABULAR_LAYOUT = TensorLayout(ensemble=0, grid=1, variables=2)
# Parameters whose gradient is already complete on every rank; the strategy does not scale them.
UNSCALED_PARAMETERS = ("trainable", "no_gradscaling")
MODELS = {
    "deterministic": {"model_class": AnemoiModelEncProcDec, "obs_predicted": True},
    "edm_transport": {"model_class": AnemoiTransportModelEncProcDec, "obs_predicted": True},
    "edm_transport_obs_input_only": {"model_class": AnemoiTransportModelEncProcDec, "obs_predicted": False},
    "edm_tendency_obs_input_only": {"model_class": AnemoiTransportTendModelEncProcDec, "obs_predicted": False},
}
NEIGHBOUR_SEARCH = {
    # pyg-lib keeps the first neighbours in row order, so its cap is above the number of observations.
    "pyg_all_neighbours": {"disable_pyg_lib": False, "max_obs_neighbours": 4096},
    # scikit-learn keeps the nearest neighbours. Hidden nodes see up to 7 history observations
    # (deterministic) and up to 9 history and noisy-target observations (transport), so a cap of 2 drops
    # observations in both models.
    "nearest_cropped": {"disable_pyg_lib": True, "max_obs_neighbours": 2},
}


def _points_on_sphere(num_points: int, generator: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes in degrees, spread evenly over the sphere."""
    latitudes = torch.rad2deg(torch.asin(2 * torch.rand(num_points, generator=generator) - 1))
    longitudes = 360 * torch.rand(num_points, generator=generator) - 180
    return latitudes.numpy(), longitudes.numpy()


def _graph_config(grid_latitudes: np.ndarray, grid_longitudes: np.ndarray, max_obs_neighbours: int) -> DictConfig:
    edge_attributes = {
        "edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": "unit-std"},
        "edge_dirs": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection", "norm": "unit-std"},
    }

    def edges(source, target, builder):
        return {"source_name": source, "target_name": target, "edge_builders": [builder], "attributes": edge_attributes}

    knn = {"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 3}
    return OmegaConf.create(
        {
            "nodes": {
                "grid": {
                    "node_builder": {
                        "_target_": "anemoi.graphs.nodes.LatLonNodes",
                        "latitudes": grid_latitudes.tolist(),
                        "longitudes": grid_longitudes.tolist(),
                    },
                },
                # Observation nodes change with every sample, so only their edges are configured.
                "obs": {"node_builder": {"_target_": "anemoi.graphs.nodes.LatLonNodes"}, "attributes": {}},
                "hidden": {"node_builder": {"_target_": "anemoi.graphs.nodes.TriNodes", "resolution": 1}},
            },
            "edges": [
                edges("grid", "hidden", {"_target_": "anemoi.graphs.edges.CutOffEdges", "cutoff_factor": 0.6}),
                edges(
                    "obs",
                    "hidden",
                    {
                        "_target_": "anemoi.graphs.edges.CutOffEdges",
                        "cutoff_factor": 0.6,
                        "max_num_neighbours": max_obs_neighbours,
                    },
                ),
                edges(
                    "hidden",
                    "hidden",
                    {"_target_": "anemoi.graphs.edges.MultiScaleEdges", "x_hops": 1, "scale_resolutions": 1},
                ),
                edges("hidden", "grid", knn),
                edges("hidden", "obs", knn),
            ],
        },
    )


def _model_config(model_class: type, obs_predicted: bool) -> DictConfig:
    transport = issubclass(model_class, AnemoiTransportModelEncProcDec)
    tendency = issubclass(model_class, AnemoiTransportTendModelEncProcDec)
    channels = 16
    layer_norm = (
        {
            "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
            "normalized_shape": channels,
            "condition_shape": 4,
            # Non-zero, so the noise MLP gets a gradient in the first step.
            "zero_init": False,
            "autocast": False,
        }
        if transport
        else {"_target_": "torch.nn.LayerNorm", "normalized_shape": channels}
    )
    block = {
        "num_channels": channels,
        "sub_graph_edge_attributes": ["edge_length", "edge_dirs"],
        "num_chunks": 2,
        "mlp_hidden_ratio": 2,
        "num_heads": 2,
        "qk_norm": False,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": {
            "LayerNorm": layer_norm,
            "Linear": {"_target_": "torch.nn.Linear"},
            "Activation": {"_target_": "torch.nn.GELU"},
        },
        # The Triton kernels have no float64 path.
        "graph_attention_backend": "pyg",
        "edge_pre_mlp": False,
    }

    def mapper(target: str, trainable_size: int, **extra) -> dict:
        return {"_target_": f"anemoi.models.layers.mapper.{target}", "trainable_size": trainable_size, **block, **extra}

    config = {
        "model": {"hidden_nodes_name": "hidden", "latent_skip": True},
        "processor": {
            "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
            "trainable_size": 2,
            "num_layers": 2,
            **block,
        },
        "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
        "encoders": {
            name: {
                "source_datasets": [name],
                "dataset_fusing_strategy": "none",
                "mapper": mapper("GraphTransformerForwardMapper", 2, shard_strategy="edges"),
            }
            for name in ("grid", "obs")
        },
        "decoders": {
            "grid": {
                "target_datasets": ["grid"],
                "target_node_features": ["encoded_data", "target_forcings"],
                "mapper": mapper(
                    "GraphTransformerBackwardMapper",
                    2,
                    shard_strategy="edges",
                    initialise_data_extractor_zero=False,
                ),
            },
            "obs": {
                "target_datasets": ["obs"],
                "target_node_features": ["coordinates", "target_forcings"],
                "mapper": mapper(
                    "GraphTransformerBackwardMapper",
                    2,
                    shard_strategy="edges",
                    initialise_data_extractor_zero=False,
                ),
            },
        },
        "residual": {"datasets": {"grid": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}}},
        "node_trainable_parameters": {"grid": 2, "obs": 0, "hidden": 2},
        "bounding": {"datasets": {"grid": [], "obs": []}},
    }
    if transport:
        config["model"]["transport"] = {
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
        }
        if tendency:
            # The tendency model takes the latest grid state as its reference and is conditioned on it.
            config["condition_on_residual"] = True
        else:
            # The transport network predicts the corrupted target itself, without a skip connection.
            config["residual"] = {"datasets": {}}
    if not obs_predicted:
        del config["decoders"]["obs"]
        del config["bounding"]["datasets"]["obs"]
    return OmegaConf.create(config)


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


def _statistics() -> dict[str, dict[str, np.ndarray]]:
    def stats(num_variables: int) -> dict[str, np.ndarray]:
        return {
            "mean": np.zeros(num_variables),
            "stdev": np.ones(num_variables),
            "minimum": -np.ones(num_variables),
            "maximum": np.ones(num_variables),
        }

    return {"grid": stats(len(GRID_VARIABLES)), "obs": stats(len(OBS_VARIABLES))}


def _build_model(model_class: type, obs_predicted: bool, max_obs_neighbours: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(0)
    latitudes, longitudes = _points_on_sphere(NUM_GRID_POINTS, generator)
    torch.manual_seed(1)
    model = model_class(
        model_config=_model_config(model_class, obs_predicted),
        model_graph_config=_graph_config(latitudes, longitudes, max_obs_neighbours),
        data_indices=_data_indices(),
        statistics=_statistics(),
        is_dataset_static={"grid": True, "obs": False},
        n_step_input={"grid": 2, "obs": 2},
        n_step_output={"grid": 1, "obs": 1},
    )
    return model.to(DTYPE)


def _variable_names(data_indices: IndexCollection, positions) -> list[str]:
    index_to_name = {index: name for name, index in data_indices.name_to_index.items()}
    return [index_to_name[int(position)] for position in positions]


def _gridded(variables: list[str], data: torch.Tensor, coordinates: torch.Tensor) -> GriddedSource:
    statistics = {
        key: value[[GRID_VARIABLES.index(name) for name in variables]] for key, value in _statistics()["grid"].items()
    }
    return GriddedSource(
        name="grid",
        variables=variables,
        layout=GRIDDED_LAYOUT,
        statistics=statistics,
        data=data,
        coordinates=coordinates,
    )


def _tabular(variables: list[str], windows: list[torch.Tensor], coordinates: list[torch.Tensor]) -> TabularSource:
    """One sample of observations, one ``(rows, variables)`` tensor per time window."""
    statistics = {
        key: value[[OBS_VARIABLES.index(name) for name in variables]] for key, value in _statistics()["obs"].items()
    }
    starts = np.cumsum([0, *[window.shape[0] for window in windows]])
    return TabularSource(
        name="obs",
        variables=variables,
        layout=TABULAR_LAYOUT,
        statistics=statistics,
        data=[torch.cat(windows)[None]],
        coordinates=[torch.cat(coordinates)],
        timedeltas=[torch.cat([torch.full((window.shape[0],), float(step)) for step, window in enumerate(windows)])],
        boundaries=[tuple(slice(int(start), int(stop)) for start, stop in zip(starts[:-1], starts[1:]))],
    )


def _shard_tabular(source: TabularSource, rank: int, world_size: int) -> TabularSource:
    """This rank's share of every time window, as the observation reader splits it."""
    data, coordinates, timedeltas = source.data[0], source.coordinates[0], source.timedeltas[0]
    keep, boundaries, window_sizes, start = [], [], [], 0
    for window in source.boundaries[0]:
        sizes = get_balanced_partition_sizes(window.stop - window.start, world_size)
        first = window.start + sum(sizes[:rank])
        keep.append(torch.arange(first, first + sizes[rank]))
        boundaries.append(slice(start, start + sizes[rank]))
        window_sizes.append(sizes)
        start += sizes[rank]
    rows = torch.cat(keep)
    return source.clone(
        data=[data[:, rows]],
        coordinates=[coordinates[rows]],
        timedeltas=[timedeltas[rows]],
        boundaries=[tuple(boundaries)],
        shard_sizes=[window_sizes],
    )


def _local_rows(source, rank: int, world_size: int) -> torch.Tensor:
    """Indices of this rank's rows in the whole source's flattened node order."""
    if isinstance(source, TabularSource):
        rows = []
        for window in source.boundaries[0]:
            sizes = get_balanced_partition_sizes(window.stop - window.start, world_size)
            first = window.start + sum(sizes[:rank])
            rows.append(torch.arange(first, first + sizes[rank]))
        return torch.cat(rows)
    sizes = get_balanced_partition_sizes(source.grid_size, world_size)
    return torch.arange(sum(sizes[:rank]), sum(sizes[: rank + 1]))


def _synthetic_batches(model: torch.nn.Module) -> dict[str, Batch]:
    """Inputs, target forcings and the noisy target of one sample, on the whole grid.

    The target forcings and the noisy target cover the predicted datasets only.
    """
    generator = torch.Generator().manual_seed(2)
    grid_coordinates = model._graph_data["grid"].x.float()  # radians, as the graph stores them
    indices = _data_indices()

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=DTYPE)

    def obs_coordinates(num_rows):
        latitudes, longitudes = _points_on_sphere(num_rows, generator)
        return torch.deg2rad(torch.stack([torch.as_tensor(latitudes), torch.as_tensor(longitudes)], dim=-1)).float()

    grid_inputs = _variable_names(indices["grid"], indices["grid"].data.input.full)
    grid_outputs = _variable_names(indices["grid"], indices["grid"].data.output.full)
    obs_inputs = _variable_names(indices["obs"], indices["obs"].data.input.full)
    obs_outputs = _variable_names(indices["obs"], indices["obs"].data.output.full)
    history_coordinates = [obs_coordinates(rows) for rows in OBS_ROWS_PER_WINDOW]
    target_coordinates = [obs_coordinates(OBS_TARGET_ROWS)]

    batches = {
        "x": Batch(
            {
                "grid": _gridded(grid_inputs, randn(1, 2, 1, NUM_GRID_POINTS, len(grid_inputs)), grid_coordinates),
                "obs": _tabular(
                    obs_inputs,
                    [randn(rows, len(obs_inputs)) for rows in OBS_ROWS_PER_WINDOW],
                    history_coordinates,
                ),
            },
        ),
        "target_forcing": Batch(
            {
                "grid": _gridded(["f"], randn(1, 1, 1, NUM_GRID_POINTS, 1), grid_coordinates),
                "obs": _tabular(["fo"], [randn(OBS_TARGET_ROWS, 1)], target_coordinates),
            },
        ),
        "noisy_target": Batch(
            {
                "grid": _gridded(grid_outputs, randn(1, 1, 1, NUM_GRID_POINTS, len(grid_outputs)), grid_coordinates),
                "obs": _tabular(obs_outputs, [randn(OBS_TARGET_ROWS, len(obs_outputs))], target_coordinates),
            },
        ),
    }
    for key in ("target_forcing", "noisy_target"):
        batches[key] = Batch({name: batches[key][name] for name in model.target_datasets})
    return batches


def _shard_batch(batch: Batch, rank: int, world_size: int, group) -> Batch:
    return Batch(
        {
            name: _shard_tabular(source, rank, world_size) if source.is_tabular else source.shard(group)
            for name, source in batch.items()
        },
    )


def _forward(model, batches: dict[str, Batch], transport: bool, group) -> Batch:
    if transport:
        sigma = torch.full((1, 1, 1, 1, 1), 0.7, dtype=DTYPE)
        return model(
            batches["x"],
            batches["noisy_target"],
            {name: sigma for name in batches["noisy_target"].dataset_names},
            model_comm_group=group,
            target_forcing=batches["target_forcing"],
        )
    return model(
        batches["x"],
        batches["target_forcing"],
        # What to predict: the output variables on the target nodes, as the task's target template.
        {name: source.template() for name, source in batches["noisy_target"].items()},
        model_comm_group=group,
    )


def _flat_output(source) -> torch.Tensor:
    return source.flatten().data


def _loss(outputs: Batch, weights: dict[str, torch.Tensor], rows: dict[str, torch.Tensor] | None) -> torch.Tensor:
    """Weighted sum of the outputs; ``rows`` picks this rank's weights when the outputs are sharded."""
    total = 0
    for name, source in outputs.items():
        out = _flat_output(source)
        weight = weights[name] if rows is None else weights[name][rows[name]]
        total = total + (out * weight).sum()
    return total


def _check_sharded_model(*, rank, world_size, device, group, model, keep_batch_sharded, neighbour_search):
    del device  # CPU only: the Triton kernels have no float64 path
    # Read by anemoi-graphs whenever it builds a graph: the graph stays on the CPU, also on a GPU node.
    os.environ["ANEMOI_GRAPHS_FORCE_CPU"] = "1"
    if neighbour_search["disable_pyg_lib"]:
        # Read by anemoi-graphs whenever it builds edges; this process is one spawned rank.
        os.environ["ANEMOI_GRAPHS_DISABLE_PYG_LIB"] = "1"
    max_obs_neighbours = neighbour_search["max_obs_neighbours"]
    transport = issubclass(model["model_class"], AnemoiTransportModelEncProcDec)
    reference_model = _build_model(model["model_class"], model["obs_predicted"], max_obs_neighbours)
    sharded_model = deepcopy(reference_model)
    batches = _synthetic_batches(reference_model)

    # Reference: the whole batch on one rank, no communication.
    reference_out = _forward(reference_model, batches, transport, group=None)
    generator = torch.Generator().manual_seed(3)
    weights = {
        name: torch.randn(_flat_output(source).shape, generator=generator, dtype=DTYPE)
        for name, source in reference_out.items()
    }
    _loss(reference_out, weights, rows=None).backward()

    # Sharded: the same weights, the batch whole or split across the group.
    rows = None
    if keep_batch_sharded:
        batches = {key: _shard_batch(batch, rank, world_size, group) for key, batch in batches.items()}
        rows = {name: _local_rows(source, rank, world_size) for name, source in reference_out.items()}
    sharded_out = _forward(sharded_model, batches, transport, group=group)
    _loss(sharded_out, weights, rows).backward()

    for name, source in sharded_out.items():
        expected = _flat_output(reference_out[name]).detach()
        if rows is not None:
            expected = expected[rows[name]]
        torch.testing.assert_close(_flat_output(source).detach(), expected, rtol=1e-12, atol=1e-12)

    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in sharded_model.named_parameters():
        expected = reference_parameters[name].grad
        if expected is None:
            assert parameter.grad is None, f"{name} has a gradient only when sharded."
            continue
        assert parameter.grad is not None, f"{name} has no gradient when sharded."
        gradient = parameter.grad.clone()
        if not any(marker in name for marker in UNSCALED_PARAMETERS):
            gradient *= world_size
        torch.distributed.all_reduce(gradient, group=group)
        gradient /= world_size
        torch.testing.assert_close(
            gradient, expected, rtol=1e-10, atol=1e-12, msg=lambda message, name=name: f"{name}: {message}"
        )


@pytest.mark.distributed
@pytest.mark.parametrize("neighbour_search", list(NEIGHBOUR_SEARCH))
@pytest.mark.parametrize("keep_batch_sharded", [False, True], ids=["batch_gathered", "batch_sharded"])
@pytest.mark.parametrize("model", list(MODELS))
def test_sharded_model_matches_whole_model(
    distributed_backend, distributed_world_size, model, keep_batch_sharded, neighbour_search
):
    if distributed_backend != "gloo":
        pytest.skip("Float64 comparison runs on CPU only.")
    _run_distributed_test(
        _check_sharded_model,
        backend=distributed_backend,
        world_size=distributed_world_size,
        model=MODELS[model],
        keep_batch_sharded=keep_batch_sharded,
        neighbour_search=NEIGHBOUR_SEARCH[neighbour_search],
    )
