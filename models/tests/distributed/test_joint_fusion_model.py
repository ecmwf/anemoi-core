# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Joint fusion of several observation (tabular) datasets through one encoder, on a whole model.

An observations-only model reads two observation datasets with the same variables through one
``joint`` encoder and decodes both. The oracle: once the two per-dataset input projections are tied,
a joint encoder sees one bag of source nodes, so moving observations from one fused dataset to the
other must not change any output. That holds only if the merged index space pairs every edge with the
right row, sample and dataset. The same model must also agree with itself when sharded.
"""

import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.data import Batch
from anemoi.models.data.sources import TabularSource
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.models.utils.graphs import build_graph
from anemoi.utils.config import DotDict

from ._distributed_runner import _run_distributed_test
from .test_model_sharding_gradients import DTYPE
from .test_model_sharding_gradients import TABULAR_LAYOUT
from .test_model_sharding_gradients import _local_rows
from .test_model_sharding_gradients import _points_on_sphere
from .test_model_sharding_gradients import _shard_tabular

DATASETS = ("obs_a", "obs_b")
VARIABLES = ["o1", "o2", "fo"]  # "fo" is a forcing
INPUTS, OUTPUTS = ["o1", "o2", "fo"], ["o1", "o2"]
STATISTICS = {
    key: np.full(3, value) for key, value in (("mean", 0.0), ("stdev", 1.0), ("minimum", -1.0), ("maximum", 1.0))
}


def _model_config(strategy: str = "joint") -> DictConfig:
    block = {
        "num_channels": 16,
        "sub_graph_edge_attributes": ["edge_length", "edge_dirs"],
        "num_chunks": 2,
        "mlp_hidden_ratio": 2,
        "num_heads": 2,
        "qk_norm": False,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": {
            "LayerNorm": {"_target_": "torch.nn.LayerNorm", "normalized_shape": 16},
            "Linear": {"_target_": "torch.nn.Linear"},
            "Activation": {"_target_": "torch.nn.GELU"},
        },
        "graph_attention_backend": "pyg",
        "edge_pre_mlp": False,
        "shard_strategy": "edges",
    }

    def mapper(target: str, **extra) -> dict:
        return {"_target_": f"anemoi.models.layers.mapper.{target}", "trainable_size": 0, **block, **extra}

    processor = {k: v for k, v in block.items() if k != "shard_strategy"}
    return OmegaConf.create(
        {
            "model": {"hidden_nodes_name": "hidden", "latent_skip": True},
            "processor": {
                "_target_": "anemoi.models.layers.processor.GraphTransformerProcessor",
                "num_layers": 2,
                **processor,
                "trainable_size": 2,
            },
            "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
            "encoders": {
                "obs": {
                    "source_datasets": list(DATASETS),
                    "dataset_fusing_strategy": strategy,
                    "node_embedder": {
                        "_target_": "anemoi.models.layers.node_embedder.LinearNodeEmbedder",
                        "num_channels": 8,
                    },
                    "mapper": mapper("GraphTransformerForwardMapper"),
                },
            },
            "decoders": {
                name: {
                    "target_datasets": [name],
                    "target_node_features": ["coordinates", "target_forcings"],
                    "mapper": mapper("GraphTransformerBackwardMapper", initialise_data_extractor_zero=False),
                }
                for name in DATASETS
            },
            "residual": {"datasets": {}},
            "node_trainable_parameters": {"obs_a": 0, "obs_b": 0, "hidden": 2},
            "bounding": {"datasets": {name: [] for name in DATASETS}},
        },
    )


def _graph_config() -> DictConfig:
    # Unnormalised: a normalisation over each dataset's own edges would change when rows move.
    attributes = {
        "edge_length": {"_target_": "anemoi.graphs.edges.attributes.EdgeLength", "norm": None},
        "edge_dirs": {"_target_": "anemoi.graphs.edges.attributes.EdgeDirection", "norm": None},
    }
    # Keep every neighbour inside the cut-off, so the graph does not depend on row order.
    cutoff = {"_target_": "anemoi.graphs.edges.CutOffEdges", "cutoff_factor": 0.6, "max_num_neighbours": 4096}
    knn = {"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 3}

    def edges(source, target, builder):
        return {"source_name": source, "target_name": target, "edge_builders": [builder], "attributes": attributes}

    multiscale = {"_target_": "anemoi.graphs.edges.MultiScaleEdges", "x_hops": 1, "scale_resolutions": 1}
    return OmegaConf.create(
        {
            "nodes": {
                **{
                    name: {"node_builder": {"_target_": "anemoi.graphs.nodes.LatLonNodes"}, "attributes": {}}
                    for name in DATASETS
                },
                "hidden": {"node_builder": {"_target_": "anemoi.graphs.nodes.TriNodes", "resolution": 1}},
            },
            "edges": [
                *(edges(name, "hidden", cutoff) for name in DATASETS),
                edges("hidden", "hidden", multiscale),
                *(edges("hidden", name, knn) for name in DATASETS),
            ],
        },
    )


def _build_model(strategy: str = "joint") -> AnemoiModelEncProcDec:
    indices = IndexCollection(
        DictConfig({"forcing": ["fo"], "diagnostic": [], "target": []}),
        {name: index for index, name in enumerate(VARIABLES)},
    )
    torch.manual_seed(1)
    static_graph_data, dynamic_graph_config = build_graph(
        DotDict(_graph_config()),
        is_dataset_static={name: False for name in DATASETS},
        hidden_nodes_name="hidden",
    )
    model = AnemoiModelEncProcDec(
        model_config=_model_config(strategy),
        static_graph_data=static_graph_data,
        dynamic_graph_config=dynamic_graph_config,
        data_indices={name: indices for name in DATASETS},
        statistics={name: STATISTICS for name in DATASETS},
        n_step_input={name: 2 for name in DATASETS},
        n_step_output={name: 1 for name in DATASETS},
    )
    return model.to(DTYPE)


def _tabular(name: str, variables: list[str], samples: list[list[tuple[torch.Tensor, torch.Tensor]]]) -> TabularSource:
    """Observations of several samples, each a list of ``(values, coordinates)`` per time window."""
    data, coordinates, timedeltas, boundaries = [], [], [], []
    for windows in samples:
        starts = np.cumsum([0, *[values.shape[0] for values, _ in windows]])
        data.append(torch.cat([values for values, _ in windows])[None])
        coordinates.append(torch.cat([coords for _, coords in windows]))
        timedeltas.append(
            torch.cat([torch.full((values.shape[0],), float(step)) for step, (values, _) in enumerate(windows)])
        )
        boundaries.append(tuple(slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])))
    return TabularSource(
        name=name,
        variables=variables,
        layout=TABULAR_LAYOUT,
        statistics={key: value[[VARIABLES.index(v) for v in variables]] for key, value in STATISTICS.items()},
        data=data,
        coordinates=coordinates,
        timedeltas=timedeltas,
        boundaries=boundaries,
    )


def _observations(rows_per_window: tuple[int, ...], seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Random observations of every variable, one ``(values, coordinates)`` pair per time window."""
    generator = torch.Generator().manual_seed(seed)
    windows = []
    for rows in rows_per_window:
        latitudes, longitudes = _points_on_sphere(rows, generator)
        coords = torch.deg2rad(torch.stack([torch.as_tensor(latitudes), torch.as_tensor(longitudes)], dim=-1)).float()
        windows.append((torch.randn(rows, len(VARIABLES), generator=generator, dtype=DTYPE), coords))
    return windows


def _columns(samples, variables: list[str]):
    """The same observations restricted to ``variables``."""
    columns = [VARIABLES.index(v) for v in variables]
    return [[(values[:, columns], coords) for values, coords in windows] for windows in samples]


def _concat(*sources: list[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Window by window, the observations of several datasets as one."""
    return [(torch.cat([v for v, _ in parts]), torch.cat([c for _, c in parts])) for parts in zip(*sources)]


def _empty_like(windows):
    return [(values[:0], coords[:0]) for values, coords in windows]


def _batches(history: dict[str, list[list]], targets: dict[str, list[list]], shard=None) -> tuple[Batch, Batch, dict]:
    """Inputs, target forcings and target template, from per-dataset per-sample windows.

    ``shard`` (a ``(rank, world_size)`` pair) keeps this rank's rows only, as the reader does.
    """

    def source(name, variables, samples):
        tabular = _tabular(name, variables, _columns(samples, variables))
        return tabular if shard is None else _shard_tabular(tabular, *shard)

    x = Batch({name: source(name, INPUTS, history[name]) for name in DATASETS})
    forcing = Batch({name: source(name, ["fo"], targets[name]) for name in DATASETS})
    template = {name: source(name, OUTPUTS, targets[name]).template() for name in DATASETS}
    return x, forcing, template


def _tie_projections(model: AnemoiModelEncProcDec) -> None:
    projections = model.encoder_src_projection["obs"]
    projections["obs_b"].load_state_dict(projections["obs_a"].state_dict())


def _outputs(model, batches, group=None) -> dict[str, list[torch.Tensor]]:
    out = model(*batches[:2], batches[2], model_comm_group=group)
    return {name: [d for d in out[name].data] for name in DATASETS}


@pytest.fixture(autouse=True)
def _graphs_on_cpu(monkeypatch) -> None:
    monkeypatch.setenv("ANEMOI_GRAPHS_FORCE_CPU", "1")


def test_joint_encoder_contributes_one_latent_and_projects_each_source() -> None:
    model = _build_model()
    assert model.latent_aggregator.source_channels == {"obs": 16}
    assert set(model.encoder_src_projection["obs"]) == set(DATASETS)


def test_sequential_encoder_still_contributes_one_latent_per_source() -> None:
    model = _build_model("sequential")
    assert model.latent_aggregator.source_channels == {"obs_a": 16, "obs_b": 16}


@pytest.mark.parametrize(
    "rows", [((5, 7), (3, 4)), ((0, 0), (3, 4)), ((0, 0), (0, 0))], ids=["both", "a_empty", "none"]
)
def test_moving_rows_between_fused_datasets_changes_nothing(rows) -> None:
    """Two samples, ragged across datasets and samples, decoded for both datasets."""
    model = _build_model()
    _tie_projections(model)

    rows_a, rows_b = rows
    samples_a = [_observations(rows_a, seed=10), _observations((2, 6), seed=11)]
    samples_b = [_observations(rows_b, seed=20), _observations((4, 1), seed=21)]
    targets = {name: [_observations((3,), 30 + i), _observations((2,), 40 + i)] for i, name in enumerate(DATASETS)}

    split = _batches({"obs_a": samples_a, "obs_b": samples_b}, targets)
    moved = _batches(
        {
            "obs_a": [_concat(a, b) for a, b in zip(samples_a, samples_b)],
            "obs_b": [_empty_like(b) for b in samples_b],
        },
        targets,
    )

    with torch.no_grad():
        expected, actual = _outputs(model, split), _outputs(model, moved)
    for name in DATASETS:
        for sample, (a, b) in enumerate(zip(actual[name], expected[name], strict=True)):
            assert a.shape[1] == targets[name][sample][0][0].shape[0]
            # The edge attributes are computed from float32 coordinates in a different batch context, so
            # they agree to float32 round-off only (outputs then to ~1e-9; a mis-indexed edge gives ~1e-1).
            torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8, msg=f"{name}, sample {sample}")


def test_untied_projections_make_the_datasets_distinguishable() -> None:
    """Guards the oracle above: without tied projections, moving rows does change the outputs."""
    model = _build_model()
    samples = {name: [_observations((5, 7), seed=10 + i)] for i, name in enumerate(DATASETS)}
    targets = {name: [_observations((3,), seed=50)] for name in DATASETS}
    split = _batches(samples, targets)
    moved = _batches(
        {"obs_a": [_concat(samples["obs_a"][0], samples["obs_b"][0])], "obs_b": [_empty_like(samples["obs_b"][0])]},
        targets,
    )
    with torch.no_grad():
        assert not torch.allclose(_outputs(model, split)["obs_a"][0], _outputs(model, moved)["obs_a"][0])


def _check_sharded_joint_model(*, rank, world_size, device, group):
    del device
    os.environ["ANEMOI_GRAPHS_FORCE_CPU"] = "1"
    reference = _build_model()
    sharded = deepcopy(reference)
    history = {name: [_observations((11, 9), seed=60 + i)] for i, name in enumerate(DATASETS)}
    targets = {name: [_observations((7,), seed=70 + i)] for i, name in enumerate(DATASETS)}
    whole = _batches(history, targets)

    reference_out = reference(*whole[:2], whole[2], model_comm_group=None)
    weights = {name: torch.randn(reference_out[name].flatten().data.shape, dtype=DTYPE) for name in DATASETS}
    sum((reference_out[n].flatten().data * weights[n]).sum() for n in DATASETS).backward()

    x, forcing, template = _batches(history, targets, shard=(rank, world_size))
    sharded_out = sharded(x, forcing, template, model_comm_group=group)
    rows = {name: _local_rows(reference_out[name], rank, world_size) for name in DATASETS}
    sum((sharded_out[n].flatten().data * weights[n][rows[n]]).sum() for n in DATASETS).backward()

    for name in DATASETS:
        torch.testing.assert_close(
            sharded_out[name].flatten().data.detach(), reference_out[name].flatten().data.detach()[rows[name]]
        )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in sharded.named_parameters():
        expected = reference_parameters[name].grad
        if expected is None:
            assert parameter.grad is None, name
            continue
        gradient = parameter.grad.clone()
        if not any(marker in name for marker in ("trainable", "no_gradscaling")):
            gradient *= world_size
        torch.distributed.all_reduce(gradient, group=group)
        torch.testing.assert_close(gradient / world_size, expected, rtol=1e-10, atol=1e-12, msg=name)


@pytest.mark.distributed
def test_sharded_joint_obs_model_matches_whole_model(distributed_backend, distributed_world_size):
    if distributed_backend != "gloo":
        pytest.skip("Float64 comparison runs on CPU only.")
    _run_distributed_test(_check_sharded_joint_model, backend=distributed_backend, world_size=distributed_world_size)
