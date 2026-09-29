# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.layers.neighbourhood_attention import NeighbourhoodAttentionWrapper
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.models.schemas.models import BaseModelSchema

DATA_GRID = ReducedGrid.octahedral(8)
HIDDEN_GRID = ReducedGrid.octahedral(4)
NUM_CHANNELS = 32

TRANSFORMER = {
    "num_channels": NUM_CHANNELS,
    "num_chunks": 1,
    "num_heads": 4,
    "mlp_hidden_ratio": 2,
    "window_size": None,
    "dropout_p": 0.0,
    "attention_implementation": "neighbourhood",
    "softcap": 0.0,
    "use_alibi_slopes": False,
    "cpu_offload": False,
    "gradient_checkpointing": False,
    "layer_kernels": {},
}


def grid_coords(grid: ReducedGrid) -> torch.Tensor:
    """Latitude and longitude in radians of the points of ``grid``, in grid order."""
    rows, positions = grid.rows_and_positions
    lat = torch.deg2rad(torch.tensor(grid.row_latitudes, dtype=torch.float64))[rows]
    lengths = torch.tensor(grid.row_lengths, dtype=torch.float64)[rows]
    lon = 2 * math.pi * positions / lengths
    return torch.stack([lat, lon], dim=1).float()


def _graph(data_order: torch.Tensor | None = None) -> HeteroData:
    graph = HeteroData()
    for name, grid in (("data", DATA_GRID), ("hidden", HIDDEN_GRID)):
        graph[name].x = grid_coords(grid)
        graph[name].num_nodes = grid.num_points
    if data_order is not None:
        graph["data"].x = graph["data"].x[data_order]
    return graph


def _model_config(neighbourhood: dict) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "model": {
                    "_target_": "anemoi.models.models.AnemoiModelEncProcDec",
                    "hidden_nodes_name": "hidden",
                    "latent_skip": True,
                },
                "node_trainable_parameters": {"data": 0, "hidden": 0},
                "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
                "processor": {
                    "_target_": "anemoi.models.layers.processor.TransformerProcessor",
                    "num_layers": 2,
                    "qk_norm": False,
                    "neighbourhood": {**neighbourhood, "kernel_size": [3, 5]},
                    **TRANSFORMER,
                },
                "encoders": {
                    "0": {
                        "source_datasets": ["data"],
                        "dataset_fusing_strategy": "not_supported",
                        "mapper": {
                            "_target_": "anemoi.models.layers.mapper.TransformerForwardMapper",
                            "neighbourhood": {**neighbourhood, "kernel_size": [5, 7]},
                            **TRANSFORMER,
                        },
                    },
                },
                "decoders": {
                    "0": {
                        "target_datasets": ["data"],
                        "target_node_features": ["encoded_data"],
                        "mapper": {
                            "_target_": "anemoi.models.layers.mapper.TransformerBackwardMapper",
                            "neighbourhood": {**neighbourhood, "kernel_size": [3, 5]},
                            "use_rotary_embeddings": False,
                            **TRANSFORMER,
                        },
                    },
                },
                "residual": {
                    "datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection", "step": -1}},
                },
                "bounding": {"datasets": {"data": []}},
                "output_mask": {"datasets": {"data": {"_target_": "anemoi.training.utils.masks.NoOutputMask"}}},
            },
        }
    )


def _data_indices() -> dict[str, IndexCollection]:
    data_config = DictConfig({"forcing": ["force"], "diagnostic": ["diag"], "target": []})
    name_to_index = {"prog0": 0, "prog1": 1, "force": 2, "diag": 3}
    return {"data": IndexCollection(data_config, name_to_index)}


def _build(graph: HeteroData | None = None, neighbourhood: dict | None = None) -> AnemoiModelEncProcDec:
    config = _model_config(neighbourhood or {"grid": "octahedral", "backend": "sdpa"})
    # Checks the configuration as training does before building the model.
    BaseModelSchema(**OmegaConf.to_container(config.model))
    return AnemoiModelEncProcDec(
        model_config=config,
        data_indices=_data_indices(),
        statistics={"data": None},
        n_step_input=2,
        n_step_output=1,
        graph_data=graph if graph is not None else _graph(),
    )


def _inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    num_input_vars = len(_data_indices()["data"].model.input)
    return {"data": torch.randn(batch_size, 2, 1, DATA_GRID.num_points, num_input_vars)}


def test_every_attention_layer_uses_the_grid_neighbourhood():
    model = _build()
    wrappers = [m for m in model.modules() if isinstance(m, NeighbourhoodAttentionWrapper)]
    # Encoder and decoder blocks each hold a self attention layer they replace with cross attention.
    grids = {(w.neighbourhood.query_grid, w.neighbourhood.key_grid) for w in wrappers}
    assert (HIDDEN_GRID, DATA_GRID) in grids  # encoder
    assert (HIDDEN_GRID, HIDDEN_GRID) in grids  # processor
    assert (DATA_GRID, HIDDEN_GRID) in grids  # decoder
    assert all(isinstance(layer.attention.attention, NeighbourhoodAttentionWrapper) for layer in model.processor.proc)


def test_forward_shape_and_gradients():
    torch.manual_seed(0)
    model = _build()
    out = model(_inputs())["data"]
    num_output_vars = len(_data_indices()["data"].model.output)
    assert out.shape == (2, 1, 1, DATA_GRID.num_points, num_output_vars)

    out.pow(2).mean().backward()
    attention = [(name, p) for name, p in model.named_parameters() if ".attention.lin_" in name]
    assert {name.split(".")[0] for name, _ in attention} == {"encoder", "processor", "decoder"}
    missing = [name for name, p in attention if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, f"Attention parameters without gradient: {missing}"


def test_data_nodes_in_another_order_give_the_same_forecast():
    torch.manual_seed(0)
    in_order = _build()
    perm = torch.randperm(DATA_GRID.num_points, generator=torch.Generator().manual_seed(0))
    out_of_order = _build(graph=_graph(data_order=perm))
    # Same weights; the node coordinates saved with the model stay as each graph has them.
    out_of_order.load_state_dict(dict(in_order.named_parameters()), strict=False)

    x = _inputs()
    expected = in_order(x)["data"][..., perm, :]
    got = out_of_order({"data": x["data"][..., perm, :]})["data"]
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_rejects_nodes_that_do_not_form_the_configured_grid():
    with pytest.raises(InstantiationException, match="do not form one of the HEALPix"):
        _build(neighbourhood={"grid": "healpix", "backend": "sdpa"})
