# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A real, tiny :class:`AnemoiModelEncProcDecHierarchical`, built from configuration and run on the CPU.

The hierarchical model is otherwise only exercised by the slow training integration tests, so
its network construction and its forward pass (encode onto the first level, up the levels,
process at the last level, back down with skip connections, decode) had no unit coverage.
Here it is built through :class:`AnemoiModelInterface` from a graph recipe alone -- a fixed
``LatLonNodes`` grid, two or three ``TriNodes`` hidden levels of increasing coarseness,
``KNNEdges`` everywhere -- with GNN processors and data mappers and graph-transformer level
mappers at 8 channels, so the tests together run in a few seconds.

The main fixture has two datasets: a gridded one, which lives in the static graph, and a
tabular one, whose encoder and decoder edges are built per sample. The tabular decoder
exercises the batched-source-coordinate branch of the hierarchical decoder loop.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.data.batch import Batch
from anemoi.models.data.sample import create_batched_struct
from anemoi.models.data.sources import GriddedSource
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.layers.mapper import GNNBackwardMapper
from anemoi.models.layers.mapper import GNNForwardMapper
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper
from anemoi.models.layers.processor import GNNProcessor
from anemoi.models.models import AnemoiModelEncProcDecHierarchical
from anemoi.models.utils.config import COORDS_DIM

CHANNELS = 8
N_LAT, N_LON = 5, 6
N_STEP_INPUT = 2
N_STEP_OUTPUT = 1

GRIDDED = "era5"
TABULAR = "obs"

VARIABLES = ["cos_jd", "lsm", "t2m", "q", "tp"]
FORCING = ["cos_jd", "lsm"]
DIAGNOSTIC = ["tp"]

# Trainable node parameters per hidden level. They differ on purpose: the upscale mappers size
# their destination embedding from the destination level's count, not the first level's.
HIDDEN_TRAINABLE = {"hidden_1": 2, "hidden_2": 3, "hidden_3": 1}
# TriNodes resolution per level: 42, 12 and 12 nodes. The last two coincide in size, which is
# harmless: the levels are separate node sets with their own edges.
HIDDEN_RESOLUTION = {"hidden_1": 1, "hidden_2": 0, "hidden_3": 0}

LAYER_KERNELS = {
    "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
    "Linear": {"_target_": "torch.nn.Linear"},
    "Activation": {"_target_": "torch.nn.GELU"},
}
EDGE_ATTRIBUTES = {"edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": "unit-max"}}
GRIDDED_LAYOUT = ("time", "ensemble", "grid", "variables")
TABULAR_LAYOUT = ("ensemble", "grid", "variables")


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------


def _grid_latlons() -> tuple[np.ndarray, np.ndarray]:
    lats, lons = np.meshgrid(np.linspace(-60.0, 60.0, N_LAT), np.linspace(0.0, 300.0, N_LON), indexing="ij")
    return lats.ravel().astype(np.float32), lons.ravel().astype(np.float32)


def _edge(src: str, dst: str, k: int) -> dict:
    return {
        "source_name": src,
        "target_name": dst,
        "edge_builders": [{"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": k}],
        "attributes": dict(EDGE_ATTRIBUTES),
    }


def _graph_config(levels: list[str], datasets: dict[str, bool]) -> dict:
    """Data nodes, one ``TriNodes`` set per level, and the edges the hierarchical model looks up.

    The datasets connect to the first level only. Consecutive levels are joined both ways, and
    every level has its own self-edges (the last level's feed the main processor, the others'
    the level processors). A tabular dataset has no node group; its edges go to the dynamic
    graph config.
    """
    lats, lons = _grid_latlons()
    nodes = {
        name: {
            "node_builder": {
                "_target_": "anemoi.graphs.nodes.LatLonNodes",
                "latitudes": lats.tolist(),
                "longitudes": lons.tolist(),
            },
            "attributes": {},
        }
        for name, is_static in datasets.items()
        if is_static
    }
    for level in levels:
        nodes[level] = {
            "node_builder": {"_target_": "anemoi.graphs.nodes.TriNodes", "resolution": HIDDEN_RESOLUTION[level]},
            "attributes": {},
        }

    # Each kind of edge set has its own neighbour count, so that a provider wired to the wrong
    # edge set shows up as a different edge count even between levels of the same size.
    edges = []
    for name in datasets:
        edges.append(_edge(name, levels[0], 3))
        edges.append(_edge(levels[0], name, 4))
    for fine, coarse in zip(levels[:-1], levels[1:]):
        edges.append(_edge(fine, coarse, 2))
        edges.append(_edge(coarse, fine, 5))
    for index, level in enumerate(levels):
        edges.append(_edge(level, level, 6 + index))

    return {"overwrite": True, "nodes": nodes, "edges": edges, "post_processors": []}


def _gnn_mapper(target: str, *, trainable_size: int) -> dict:
    return {
        "_target_": f"anemoi.models.layers.mapper.{target}",
        "num_channels": CHANNELS,
        "trainable_size": trainable_size,
        "sub_graph_edge_attributes": list(EDGE_ATTRIBUTES),
        "num_chunks": 1,
        "mlp_extra_layers": 0,
        "mlp_hidden_ratio": 1.0,
        "mlp_implementation": "mlp",
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": dict(LAYER_KERNELS),
    }


def _graph_transformer_mapper(target: str, *, trainable_size: int) -> dict:
    """A graph-transformer mapper on the ``pyg`` attention backend, the one that runs without triton.

    Level mappers carry no ``num_channels``: the model sizes them per level, as in the packaged
    hierarchical config. The tabular decoder adds one, since it is a data mapper.
    """
    config = {
        "_target_": f"anemoi.models.layers.mapper.{target}",
        "trainable_size": trainable_size,
        "sub_graph_edge_attributes": list(EDGE_ATTRIBUTES),
        "num_chunks": 1,
        "mlp_hidden_ratio": 1,
        "mlp_implementation": "mlp",
        "num_heads": 2,
        "qk_norm": False,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": dict(LAYER_KERNELS),
        "shard_strategy": "edges",
        "graph_attention_backend": "pyg",
        "edge_pre_mlp": False,
    }
    if target == "GraphTransformerBackwardMapper":
        config["initialise_data_extractor_zero"] = False
    return config


def _model_config(
    levels: list[str],
    datasets: dict[str, bool],
    *,
    level_processing: bool,
    input_only: tuple[str, ...] = (),
    output_only: tuple[str, ...] = (),
) -> dict:
    """Model config; every dataset is encoded and decoded unless named in ``input_only``/``output_only``."""
    # A DynamicGraphProvider cannot carry trainable edge parameters, so tabular datasets ask for none.
    trainable = {name: 2 if is_static else 0 for name, is_static in datasets.items()}
    tabular_decoder = {**_graph_transformer_mapper("GraphTransformerBackwardMapper", trainable_size=0)}
    tabular_decoder["num_channels"] = CHANNELS
    return {
        "num_channels": CHANNELS,
        "cpu_offload": False,
        "keep_batch_sharded": False,
        "model": {
            "_target_": "anemoi.models.models.AnemoiModelEncProcDecHierarchical",
            "hidden_nodes_name": list(levels),
            "latent_skip": True,
        },
        "layer_kernels": dict(LAYER_KERNELS),
        "enable_hierarchical_level_processing": level_processing,
        "level_process_num_layers": 2,
        "processor": {
            "_target_": "anemoi.models.layers.processor.GNNProcessor",
            "num_channels": CHANNELS,
            "trainable_size": 2,
            "sub_graph_edge_attributes": list(EDGE_ATTRIBUTES),
            "num_layers": 1,
            "num_chunks": 1,
            "mlp_extra_layers": 0,
            "mlp_hidden_ratio": 1.0,
            "mlp_implementation": "mlp",
            "cpu_offload": False,
            "gradient_checkpointing": False,
            "layer_kernels": dict(LAYER_KERNELS),
        },
        "upscale_mapper": _graph_transformer_mapper("GraphTransformerForwardMapper", trainable_size=2),
        "downscale_mapper": _graph_transformer_mapper("GraphTransformerBackwardMapper", trainable_size=2),
        "encoders": {
            name: {
                "source_datasets": [name],
                "dataset_fusing_strategy": "none",
                "mapper": _gnn_mapper("GNNForwardMapper", trainable_size=trainable[name]),
            }
            for name in datasets
            if name not in output_only
        },
        "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
        "decoders": {
            name: (
                {
                    "target_datasets": [name],
                    "target_node_features": ["encoded_data"],
                    "mapper": _gnn_mapper("GNNBackwardMapper", trainable_size=trainable[name]),
                }
                if is_static and name not in output_only
                else {
                    # nothing encoded to decode onto: start from the output nodes' coordinates
                    "target_datasets": [name],
                    "target_node_features": ["coordinates"] if is_static else ["coordinates", "target_forcings"],
                    "mapper": _gnn_mapper("GNNBackwardMapper", trainable_size=trainable[name])
                    if is_static
                    else tabular_decoder,
                }
            )
            for name, is_static in datasets.items()
            if name not in input_only
        },
        "residual": {
            "datasets": {
                name: {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}
                for name, is_static in datasets.items()
                if is_static and name not in (*input_only, *output_only)
            }
        },
        "node_trainable_parameters": {
            **{name: 2 for name, is_static in datasets.items() if is_static},
            **{level: HIDDEN_TRAINABLE[level] for level in levels},
        },
        "attributes": {"edges": list(EDGE_ATTRIBUTES), "nodes": []},
        "bounding": {"datasets": {}},
    }


def _data_config(datasets: dict[str, bool]) -> dict:
    return {
        "datasets": {
            name: {
                "forcing": list(FORCING),
                "diagnostic": list(DIAGNOSTIC),
                "processors": {
                    "normalizer": {
                        "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
                        "config": {"default": "mean-std"},
                    }
                },
            }
            for name in datasets
        }
    }


def _statistics() -> dict:
    rng = np.random.default_rng(0)
    stdev = rng.uniform(0.5, 2.0, len(VARIABLES)).astype(np.float32)
    mean = rng.uniform(-1.0, 1.0, len(VARIABLES)).astype(np.float32)
    return {"mean": mean, "stdev": stdev, "minimum": mean - 3 * stdev, "maximum": mean + 3 * stdev}


def build_interface(
    levels: list[str], datasets: dict[str, bool], *, level_processing: bool = True, **routing: tuple[str, ...]
) -> AnemoiModelInterface:
    """Build a real hierarchical interface over ``{dataset_name: is_static_grid}``.

    The graph is built on the CPU (``ANEMOI_GRAPHS_FORCE_CPU``) so it matches the model's
    parameters, which start there.
    """
    torch.manual_seed(0)
    config = OmegaConf.create(
        {
            "data": _data_config(datasets),
            "model": _model_config(levels, datasets, level_processing=level_processing, **routing),
            "graph": _graph_config(levels, datasets),
        }
    )
    name_to_index = {name: i for i, name in enumerate(VARIABLES)}
    data_indices = {name: IndexCollection(config.data.datasets[name], name_to_index=name_to_index) for name in datasets}

    previous = os.environ.get("ANEMOI_GRAPHS_FORCE_CPU")
    os.environ["ANEMOI_GRAPHS_FORCE_CPU"] = "1"
    try:
        return AnemoiModelInterface(
            config=config,
            n_step_input=dict.fromkeys(datasets, N_STEP_INPUT),
            n_step_output=dict.fromkeys(datasets, N_STEP_OUTPUT),
            statistics={name: _statistics() for name in datasets},
            data_indices=data_indices,
            metadata={"metadata_inference": {name: {} for name in datasets}},
            data_types={name: "gridded" if is_static else "tabular" for name, is_static in datasets.items()},
        )
    finally:
        if previous is None:
            os.environ.pop("ANEMOI_GRAPHS_FORCE_CPU", None)
        else:
            os.environ["ANEMOI_GRAPHS_FORCE_CPU"] = previous


# --------------------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------------------


def _input_names(model_interface: AnemoiModelInterface, name: str) -> list[str]:
    return model_interface.data_indices[name].model.input.ordered_names


def _output_names(model_interface: AnemoiModelInterface, name: str) -> list[str]:
    return model_interface.data_indices[name].model.output.ordered_names


def _forcing_names(model_interface: AnemoiModelInterface, name: str) -> list[str]:
    return [variable for variable in _input_names(model_interface, name) if variable in FORCING]


def _gridded_payload(n_step: int, variables: list[str], seed: int) -> dict:
    lats, lons = _grid_latlons()
    generator = torch.Generator().manual_seed(seed)
    return {
        "data_type": "gridded",
        "values": torch.randn(n_step, 1, len(lats), len(variables), generator=generator),
        "variables": list(variables),
        "latitudes": lats,
        "longitudes": lons,
        "layout": GRIDDED_LAYOUT,
    }


def _tabular_payload(nodes_per_step: list[int], variables: list[str], seed: int) -> dict:
    """Observations with time folded into the node axis; slot sizes are deliberately unequal."""
    rng = np.random.default_rng(seed)
    total = int(sum(nodes_per_step))
    boundaries, timedeltas, offset = [], [], 0
    for step, count in enumerate(nodes_per_step):
        boundaries.append((offset, offset + count))
        timedeltas.extend([float((step - len(nodes_per_step) + 1) * 3600)] * count)
        offset += count
    generator = torch.Generator().manual_seed(seed)
    return {
        "data_type": "tabular",
        "values": torch.randn(1, total, len(variables), generator=generator),
        "variables": list(variables),
        "latitudes": rng.uniform(-70.0, 70.0, total).astype(np.float32),
        "longitudes": rng.uniform(0.0, 350.0, total).astype(np.float32),
        "timedeltas": np.asarray(timedeltas, dtype=np.float32),
        "boundaries": boundaries,
        "layout": TABULAR_LAYOUT,
    }


def _sample(model_interface: AnemoiModelInterface, name: str, is_static: bool, seed: int) -> tuple[dict, dict]:
    """An input payload and its decoder conditioning (the forcings at the output time)."""
    if is_static:
        return (
            _gridded_payload(N_STEP_INPUT, _input_names(model_interface, name), seed),
            _gridded_payload(N_STEP_OUTPUT, _forcing_names(model_interface, name), seed + 100),
        )
    return (
        _tabular_payload([7, 5], _input_names(model_interface, name), seed),
        _tabular_payload([4], _forcing_names(model_interface, name), seed + 100),
    )


def _template(payload: dict, variables: list[str]) -> dict:
    geometry = {key: value for key, value in payload.items() if key != "values"}
    geometry["variables"] = list(variables)
    return geometry


def _statistics_for(model_interface: AnemoiModelInterface, name: str, variables: list[str]) -> dict:
    positions = [model_interface.data_indices[name].name_to_index[variable] for variable in variables]
    return {key: values[positions] for key, values in model_interface.statistics[name].items()}


def forward_inputs(
    model_interface: AnemoiModelInterface, datasets: dict[str, bool], *, batch_size: int
) -> tuple[Batch, Batch, dict]:
    """The three positional arguments of ``forward``, for ``batch_size`` independent samples.

    Every sample of a tabular dataset decodes onto the same output geometry here, since a
    template is built from one payload and repeated over the batch.
    """
    inputs, forcings = [], []
    geometry = {}
    for index in range(batch_size):
        sample_inputs, sample_forcings = {}, {}
        for name, is_static in datasets.items():
            sample, forcing = _sample(model_interface, name, is_static, seed=10 * index + 1)
            if not is_static:
                # one output geometry for the whole batch
                forcing = _sample(model_interface, name, is_static, seed=1)[1]
            sample_inputs[name] = create_batched_struct(
                statistics=_statistics_for(model_interface, name, sample["variables"]), **sample
            )
            sample_forcings[name] = create_batched_struct(
                statistics=_statistics_for(model_interface, name, forcing["variables"]), **forcing
            )
            geometry[name] = _template(forcing, _output_names(model_interface, name))
        inputs.append(sample_inputs)
        forcings.append(sample_forcings)

    batch = Batch.collate(inputs)
    templates = model_interface._get_template(geometry, ensemble_size=1, batch_size=batch_size, device=batch.device)
    return batch, Batch.collate(forcings), templates


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

TWO_LEVELS = ["hidden_1", "hidden_2"]
THREE_LEVELS = ["hidden_1", "hidden_2", "hidden_3"]
GRIDDED_ONLY = {GRIDDED: True}
MIXED = {GRIDDED: True, TABULAR: False}


@pytest.fixture(scope="module")
def three_level_interface() -> AnemoiModelInterface:
    return build_interface(THREE_LEVELS, MIXED, level_processing=True)


@pytest.fixture(scope="module")
def two_level_interface() -> AnemoiModelInterface:
    return build_interface(TWO_LEVELS, GRIDDED_ONLY, level_processing=False)


# --------------------------------------------------------------------------------------
# _build_networks
# --------------------------------------------------------------------------------------


def test_build_networks_creates_one_mapper_pair_and_two_processors_per_level(three_level_interface) -> None:
    """Every level but the last gets an upscale and a downscale mapper and, with level
    processing on, a processor on the way up and one on the way down. Upscale mappers are keyed
    by their source level and downscale mappers by their destination level, which is how
    ``forward`` looks them up.
    """
    model = three_level_interface.model
    assert isinstance(model, AnemoiModelEncProcDecHierarchical)
    assert len(model.hidden_names) == 3
    assert model.level_process is True

    lower_levels = THREE_LEVELS[:-1]
    for modules in (
        model.upscale,
        model.upscale_graph_providers,
        model.downscale,
        model.downscale_graph_providers,
        model.down_level_processor,
        model.down_level_processor_graph_providers,
        model.up_level_processor,
        model.up_level_processor_graph_providers,
    ):
        assert list(modules) == lower_levels

    assert all(isinstance(model.upscale[level], GraphTransformerForwardMapper) for level in lower_levels)
    assert all(isinstance(model.downscale[level], GraphTransformerBackwardMapper) for level in lower_levels)
    assert isinstance(model.processor, GNNProcessor)

    # Data mappers touch the first level only.
    assert set(model.encoder) == {GRIDDED, TABULAR}
    assert isinstance(model.encoder[GRIDDED], GNNForwardMapper)
    assert isinstance(model.decoder[GRIDDED], GNNBackwardMapper)
    assert isinstance(model.decoder[TABULAR], GraphTransformerBackwardMapper)
    assert set(model.encoder_graph_provider) == set(model.decoder_graph_provider) == {GRIDDED, TABULAR}


def test_build_networks_doubles_the_latent_width_at_each_level(three_level_interface) -> None:
    """The latent is ``num_channels`` wide at the first level and doubles at each level up; every
    level module is sized from that, not from the ``num_channels`` placeholder in its config.
    """
    model = three_level_interface.model
    assert model.hidden_dims == {"hidden_1": CHANNELS, "hidden_2": 2 * CHANNELS, "hidden_3": 4 * CHANNELS}

    for src, dst in zip(THREE_LEVELS[:-1], THREE_LEVELS[1:]):
        upscale = model.upscale[src]
        assert upscale.in_channels_src == model.hidden_dims[src]
        assert upscale.hidden_dim == model.hidden_dims[dst]
        # the destination level starts from its coordinates and its own trainable parameters
        assert upscale.in_channels_dst == COORDS_DIM + HIDDEN_TRAINABLE[dst]

        downscale = model.downscale[src]
        assert downscale.in_channels_src == model.hidden_dims[dst]
        assert downscale.in_channels_dst == model.hidden_dims[src]
        assert downscale.out_channels_dst == model.hidden_dims[src]

        for processors in (model.down_level_processor, model.up_level_processor):
            processor = processors[src]
            assert processor.num_channels == model.hidden_dims[src]
            # level processors take ``level_process_num_layers``, not the main processor's ``num_layers``
            assert processor.num_layers == 2

    assert model.processor.num_channels == model.hidden_dims["hidden_3"]
    assert model.processor.num_layers == 1


def test_build_networks_wires_each_provider_to_its_own_edges(three_level_interface) -> None:
    """Each graph provider must be built from the edge set it is used on: data -> first level,
    level i -> i + 1 for upscaling, i + 1 -> i for downscaling, and each level's self-edges.
    """
    model = three_level_interface.model
    graph = three_level_interface.graph_data

    def assert_built_from(provider, src: str, dst: str) -> None:
        edges = graph[(src, "to", dst)].edge_index
        # the provider keeps the edges sorted by destination, and the node counts of both ends
        assert provider.edge_inc.flatten().tolist() == [graph[src].num_nodes, graph[dst].num_nodes]
        assert set(map(tuple, provider.edge_index_base.T.tolist())) == set(map(tuple, edges.T.tolist()))
        assert provider.edge_index_base.shape == edges.shape

    for src, dst in zip(THREE_LEVELS[:-1], THREE_LEVELS[1:]):
        assert_built_from(model.upscale_graph_providers[src], src, dst)
        assert_built_from(model.downscale_graph_providers[src], dst, src)
        assert_built_from(model.down_level_processor_graph_providers[src], src, src)
        assert_built_from(model.up_level_processor_graph_providers[src], src, src)
    assert list(model.processor_graph_provider) == ["hidden_3"]
    assert_built_from(model.processor_graph_provider["hidden_3"], "hidden_3", "hidden_3")
    assert_built_from(model.encoder_graph_provider[GRIDDED], GRIDDED, "hidden_1")
    assert_built_from(model.decoder_graph_provider[GRIDDED], "hidden_1", GRIDDED)


def test_build_networks_without_level_processing_has_no_level_processors(two_level_interface) -> None:
    """With level processing off only the main processor exists, and ``forward`` must not look
    for the per-level ones.
    """
    model = two_level_interface.model
    assert len(model.hidden_names) == 2
    assert model.level_process is False
    for name in (
        "down_level_processor",
        "up_level_processor",
        "down_level_processor_graph_providers",
        "up_level_processor_graph_providers",
    ):
        assert not hasattr(model, name)
    assert list(model.upscale) == list(model.downscale) == ["hidden_1"]


# --------------------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "datasets"),
    [("three_level_interface", MIXED), ("two_level_interface", GRIDDED_ONLY)],
    ids=["three-levels-mixed-level-processing", "two-levels-gridded"],
)
def test_forward_returns_finite_output_on_the_target_geometry(fixture, datasets, request) -> None:
    """A batch of two goes through every level and comes back as one source per decoded
    dataset, with the output variables, on the template's nodes, and finite.
    """
    model_interface = request.getfixturevalue(fixture)
    batch_size = 2
    batch, forcings, templates = forward_inputs(model_interface, datasets, batch_size=batch_size)

    with torch.no_grad():
        out = model_interface.model(batch, forcings, templates)

    assert set(out.keys()) == set(datasets)
    for name, is_static in datasets.items():
        source = out[name]
        assert source.variables == _output_names(model_interface, name)
        assert torch.all(torch.isfinite(source.values if is_static else torch.cat(source.values, dim=-2)))
        if is_static:
            assert isinstance(source, GriddedSource)
            # (batch, time, ensemble, grid, variables)
            assert source.values.shape == (batch_size, N_STEP_OUTPUT, 1, N_LAT * N_LON, len(source.variables))
        else:
            assert isinstance(source, TabularSource)
            assert len(source.values) == batch_size
            for sample_data in source.values:
                # (ensemble, nodes, variables): the four output nodes of the template
                assert sample_data.shape == (1, 4, len(source.variables))


def test_forward_decodes_only_the_datasets_in_the_template(three_level_interface) -> None:
    """The template, not the input batch, decides what is decoded."""
    batch, forcings, templates = forward_inputs(three_level_interface, MIXED, batch_size=1)

    with torch.no_grad():
        out = three_level_interface.model(batch, forcings, {GRIDDED: templates[GRIDDED]})

    assert set(out.keys()) == {GRIDDED}


def test_forward_samples_in_a_batch_are_independent(two_level_interface) -> None:
    """The hidden latents are repeated per sample and the graphs batched; a sample's output must
    not depend on the other samples in the batch.
    """
    batch, forcings, templates = forward_inputs(two_level_interface, GRIDDED_ONLY, batch_size=2)
    single_batch, single_forcings, single_templates = forward_inputs(two_level_interface, GRIDDED_ONLY, batch_size=1)

    with torch.no_grad():
        pair = two_level_interface.model(batch, forcings, templates)[GRIDDED].values
        single = two_level_interface.model(single_batch, single_forcings, single_templates)[GRIDDED].values

    torch.testing.assert_close(pair[:1], single, rtol=1e-5, atol=1e-5)


def test_forward_rejects_mixed_ensemble_sizes(three_level_interface) -> None:
    """All datasets go through one latent, so they must agree on the number of members."""
    batch, forcings, templates = forward_inputs(three_level_interface, MIXED, batch_size=1)
    gridded = batch[GRIDDED]
    two_members = gridded.clone(values=torch.cat([gridded.values, gridded.values], dim=gridded.layout.axis("ensemble")))
    batch = batch.replace(GRIDDED, two_members)

    with pytest.raises(ValueError, match="same ensemble size"):
        three_level_interface.model(batch, forcings, templates)


def test_forward_skips_a_dataset_absent_from_the_batch(three_level_interface) -> None:
    """A dataset with an encoder may be missing from a batch (e.g. no observations in the window):
    its encoder is skipped and the latent is built from the datasets that are there.
    """
    batch, forcings, templates = forward_inputs(three_level_interface, MIXED, batch_size=1)
    gridded_only = Batch({GRIDDED: batch[GRIDDED]})

    with torch.no_grad():
        out = three_level_interface.model(gridded_only, forcings, {GRIDDED: templates[GRIDDED]})

    assert set(out.keys()) == {GRIDDED}
    assert torch.all(torch.isfinite(out[GRIDDED].values))


def test_input_only_and_output_only_datasets() -> None:
    """An input-only dataset gets an encoder and no decoder, an output-only one a decoder and no
    encoder; ``forward`` decodes only datasets that have a decoder, even if the template names more.
    """
    datasets = {GRIDDED: True, "aux": True, "out": True}
    model_interface = build_interface(
        TWO_LEVELS, datasets, level_processing=False, input_only=("aux",), output_only=("out",)
    )
    model = model_interface.model
    assert set(model.encoder_graph_provider) == {GRIDDED, "aux"}
    assert set(model.decoder_graph_provider) == {GRIDDED, "out"}

    batch, forcings, templates = forward_inputs(model_interface, datasets, batch_size=1)
    with torch.no_grad():
        out = model(batch, forcings, templates)

    assert set(out.keys()) == {GRIDDED, "out"}
    for name in (GRIDDED, "out"):
        assert out[name].values.shape[-2] == N_LAT * N_LON
        assert torch.all(torch.isfinite(out[name].values))


@pytest.mark.parametrize(
    ("levels", "datasets", "level_processing"),
    [(THREE_LEVELS, MIXED, True), (TWO_LEVELS, GRIDDED_ONLY, False)],
    ids=["three-levels-mixed-level-processing", "two-levels-gridded"],
)
def test_backward_reaches_every_trainable_parameter(levels, datasets, level_processing) -> None:
    """No dead branch: every level mapper, level processor, data mapper, trainable node and edge
    parameter takes part in the output, so each gets a non-zero gradient from one backward pass.
    """
    model_interface = build_interface(levels, datasets, level_processing=level_processing)
    batch, forcings, templates = forward_inputs(model_interface, datasets, batch_size=2)

    out = model_interface.model(batch, forcings, templates)
    loss = sum(
        (source.values if isinstance(source, GriddedSource) else torch.cat(source.values, dim=-2)).square().mean()
        for source in out.values()
    )
    loss.backward()

    params = dict(model_interface.model.named_parameters())
    assert params
    missing = [name for name, param in params.items() if param.requires_grad and param.grad is None]
    zero = [
        name
        for name, param in params.items()
        if param.requires_grad and param.grad is not None and not torch.any(param.grad != 0)
    ]
    assert not missing, f"parameters without a gradient: {missing}"
    assert not zero, f"parameters with an all-zero gradient: {zero}"


def _level_mapper(target: str) -> dict:
    """A level mapper of type ``target``, without ``num_channels``: the model sizes it per level."""
    if target.startswith("GNN"):
        config = _gnn_mapper(target, trainable_size=2)
        del config["num_channels"]
        return config
    return _graph_transformer_mapper(target, trainable_size=2)


def _interface_with_level_mappers(
    monkeypatch: pytest.MonkeyPatch, upscale: str, downscale: str, levels: list[str]
) -> AnemoiModelInterface:
    build_model_config = _model_config

    def with_level_mappers(*args, **kwargs):
        config = build_model_config(*args, **kwargs)
        config["upscale_mapper"] = _level_mapper(upscale)
        config["downscale_mapper"] = _level_mapper(downscale)
        return config

    monkeypatch.setattr(sys.modules[__name__], "_model_config", with_level_mappers)
    return build_interface(levels, GRIDDED_ONLY, level_processing=False)


@pytest.mark.parametrize("levels", [TWO_LEVELS, THREE_LEVELS], ids=["two-levels", "three-levels"])
@pytest.mark.parametrize(
    ("upscale", "downscale"),
    [
        ("GNNForwardMapper", "GNNBackwardMapper"),
        ("GNNForwardMapper", "GraphTransformerBackwardMapper"),
        ("GraphTransformerForwardMapper", "GNNBackwardMapper"),
    ],
    ids=["gnn-gnn", "gnn-up-gt-down", "gt-up-gnn-down"],
)
def test_forward_with_gnn_level_mappers(
    monkeypatch: pytest.MonkeyPatch, upscale: str, downscale: str, levels: list[str]
) -> None:
    """The schema accepts a GNN mapper as ``upscale_mapper`` and ``downscale_mapper``, alone or mixed.

    ``forward`` keeps what the upscale mapper returns for its source level and hands it to the
    downscale mapper as destination features. A graph-transformer mapper passes its source input
    through, so that is the level's latent; a GNN one returns its updated source latent at the
    next level's width, and the downscale mapper is built for that width.
    """
    model_interface = _interface_with_level_mappers(monkeypatch, upscale, downscale, levels)
    batch, forcings, templates = forward_inputs(model_interface, GRIDDED_ONLY, batch_size=1)

    with torch.no_grad():
        out = model_interface.model(batch, forcings, templates)

    assert out[GRIDDED].values.shape[-2] == N_LAT * N_LON
    assert torch.all(torch.isfinite(out[GRIDDED].values))


def test_backward_reaches_the_updated_source_latent_of_gnn_level_mappers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source-node update of a GNN upscale mapper feeds the downscale mapper, so it is trained.

    Dropping the updated source latent would leave those weights without a gradient (and DDP with
    unused parameters).
    """
    model_interface = _interface_with_level_mappers(monkeypatch, "GNNForwardMapper", "GNNBackwardMapper", THREE_LEVELS)
    batch, forcings, templates = forward_inputs(model_interface, GRIDDED_ONLY, batch_size=2)

    out = model_interface.model(batch, forcings, templates)
    out[GRIDDED].values.square().mean().backward()

    upscale = dict(model_interface.model.upscale.named_parameters())
    assert upscale
    without_gradient = [name for name, param in upscale.items() if param.grad is None or not torch.any(param.grad != 0)]
    assert not without_gradient, f"upscale parameters without a gradient: {without_gradient}"


# --------------------------------------------------------------------------------------
# predict_step
# --------------------------------------------------------------------------------------


def test_predict_step_round_trip(three_level_interface) -> None:
    """``predict_step``, the inference entry point, runs the hierarchical model end to end on
    plain sample dicts: pre-processing, ``forward`` and post-processing, without gradients.
    """
    samples, forcings = {}, {}
    for name, is_static in MIXED.items():
        samples[name], forcings[name] = _sample(three_level_interface, name, is_static, seed=1)
    templates = {name: _template(forcings[name], _output_names(three_level_interface, name)) for name in MIXED}

    out = three_level_interface.predict_step(samples, target_template=templates, target_forcing=forcings)

    assert set(out) == set(MIXED)
    gridded = out[GRIDDED]
    assert gridded["variables"] == _output_names(three_level_interface, GRIDDED)
    assert gridded["values"].shape == (N_STEP_OUTPUT, 1, N_LAT * N_LON, len(gridded["variables"]))
    assert not gridded["values"].requires_grad
    assert torch.all(torch.isfinite(gridded["values"]))

    tabular = out[TABULAR]
    assert tabular["values"].shape == (1, 4, len(tabular["variables"]))
    assert torch.all(torch.isfinite(tabular["values"]))
