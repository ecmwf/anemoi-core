# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared construction helpers for tests that build REAL tiny transport models.

Several test modules in this directory (e.g. ``test_transport_conditioning_only.py`` and
``test_transport_residual_forward.py``) instantiate :class:`AnemoiTransportModelEncProcDec` and its
subclasses through their real ``__init__`` with a real tiny graph, real graph-transformer
encoders/decoders/processor, a real ``model_config`` ``DictConfig``, real ``IndexCollection``\\ s and
statistics dicts. The pieces below are the boring, mechanical construction of those inputs — every
choice that actually matters for a given test (dataset names, grid sizes, variable roles, hidden mesh
size, ``conditioning_only_datasets`` / ``history_less_datasets`` / ``residual_prediction`` / ...) is
still supplied explicitly by the calling test module. Nothing here picks a default for those.

All helpers are exposed as factory fixtures (a fixture that returns a builder function) rather than
plain module-level functions, so that test modules never need to import this conftest module
directly — fixture injection is what pytest guarantees works correctly across sibling test files.
"""

import numpy as np
import pytest
import scipy.sparse
import torch
from omegaconf import DictConfig
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection


def _coords(n: int, offset: float) -> torch.Tensor:
    lats = torch.linspace(-1.0, 1.0, n) + offset
    lons = torch.linspace(0.0, 2.0, n) - offset
    return torch.stack((lats, lons), dim=-1).float()


def _dense_bipartite_edges(n_src: int, n_dst: int) -> torch.Tensor:
    src = torch.arange(n_src).repeat_interleave(n_dst)
    dst = torch.arange(n_dst).repeat(n_src)
    return torch.stack((src, dst), dim=0)


@pytest.fixture
def index_collection_factory():
    """Build an IndexCollection from a name->index map plus explicit forcing/diagnostic/target roles."""

    def _build(
        name_to_index: dict[str, int],
        *,
        forcing: list[str] = (),
        diagnostic: list[str] = (),
        target: list[str] = (),
    ) -> IndexCollection:
        cfg = DictConfig({"forcing": list(forcing), "diagnostic": list(diagnostic), "target": list(target)})
        return IndexCollection(cfg, dict(name_to_index))

    return _build


@pytest.fixture
def statistics_factory():
    """Build a zero-mean mean/stdev/minimum/maximum statistics dict.

    Pass an int ``n`` for an all-ones stdev of length ``n``, or an explicit per-channel stdev list.
    """

    def _build(stdev: int | list[float]) -> dict:
        stdev_arr = np.ones(stdev, dtype=np.float32) if isinstance(stdev, int) else np.asarray(stdev, dtype=np.float32)
        n = stdev_arr.shape[0]
        return {
            "mean": np.zeros(n, dtype=np.float32),
            "stdev": stdev_arr,
            "minimum": np.zeros(n, dtype=np.float32),
            "maximum": np.ones(n, dtype=np.float32),
        }

    return _build


@pytest.fixture
def tiny_graph_factory():
    """Build a tiny HeteroData graph: one node set per (name, size), dense bipartite edges.

    ``node_sizes`` is an ordered ``{name: n}`` mapping; each node set gets its own coordinate offset
    (0.0, 0.1, 0.2, ... in insertion order) so that distinct grids never collide. ``edge_pairs`` is the
    explicit list of ``(src, dst)`` node-name pairs to connect (dense src x dst edges); it is exactly
    the graph topology, so callers must state it, not have it inferred.
    """

    def _build(node_sizes: dict[str, int], edge_pairs: list[tuple[str, str]], *, seed: int = 7) -> HeteroData:
        graph = HeteroData()
        for i, (name, n) in enumerate(node_sizes.items()):
            offset = i * 0.1
            graph[name].x = _coords(n, offset)
            graph[name].num_nodes = n

        generator = torch.Generator().manual_seed(seed)
        for src, dst in edge_pairs:
            edge_index = _dense_bipartite_edges(node_sizes[src], node_sizes[dst])
            store = graph[(src, "to", dst)]
            store.edge_index = edge_index
            store.edge_length = torch.rand((edge_index.shape[1], 1), generator=generator)
        return graph

    return _build


@pytest.fixture
def layer_kernels_factory():
    """Build the ``layer_kernels`` config block (ConditionalLayerNorm + autocast Q/K norms)."""

    def _build(num_channels: int, noise_cond_dim: int) -> dict:
        return {
            "LayerNorm": {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "normalized_shape": num_channels,
                "condition_shape": noise_cond_dim,
                "zero_init": True,
                "autocast": False,
            },
            "Linear": {"_target_": "torch.nn.Linear"},
            "Activation": {"_target_": "torch.nn.GELU"},
            "QueryNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
            "KeyNorm": {"_target_": "anemoi.models.layers.normalization.AutocastLayerNorm", "bias": False},
        }

    return _build


@pytest.fixture
def transport_model_config_factory(layer_kernels_factory):
    """Build the full ``model_config`` DictConfig for a tiny real transport model.

    ``model_extra`` is merged into the ``model.model`` section and is where the calling test states
    the dataset-role knobs that make each test what it is (``conditioning_only_datasets``,
    ``history_less_datasets``, ``residual_prediction``, ``direct_prediction``, ...) — there is no
    default for any of them here. ``residual`` overrides the residual-connection block; it defaults to
    a plain ``SkipConnection`` (the conditioning-only tests' case), while a residual model test passes
    an explicit ``InterpolationConnection`` block instead.
    """

    def _build(
        *,
        num_channels: int,
        noise_channels: int,
        noise_cond_dim: int,
        model_extra: dict,
        residual: dict | None = None,
    ) -> DictConfig:
        layer_kernels = layer_kernels_factory(num_channels, noise_cond_dim)
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
        model_section = {
            "hidden_nodes_name": "hidden",
            "latent_skip": True,
            **model_extra,
            "transport": {
                "objective": "edm_diffusion",
                "sigma_data": 1.0,
                "noise_channels": noise_channels,
                "noise_cond_dim": noise_cond_dim,
                "noise_embedder": {
                    "_target_": "anemoi.models.layers.diffusion.SinusoidalEmbeddings",
                    "num_channels": noise_channels,
                    "max_period": 1000,
                },
            },
        }
        return DictConfig(
            {
                "model": {
                    "num_channels": num_channels,
                    "cpu_offload": False,
                    "keep_batch_sharded": False,
                    "model": model_section,
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
                    "residual": residual or {"_target_": "anemoi.models.layers.residual.SkipConnection"},
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

    return _build


@pytest.fixture
def interpolation_npz_factory(tmp_path):
    """Write a dense interpolation matrix to a sparse ``.npz`` file and return its path."""

    def _build(matrix: np.ndarray, filename: str = "source_to_target.npz") -> str:
        path = tmp_path / filename
        scipy.sparse.save_npz(path, scipy.sparse.csr_matrix(matrix))
        return str(path)

    return _build
