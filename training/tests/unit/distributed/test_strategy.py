# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.training.distributed.strategy import register_gradient_scaling_hooks

GLOBAL_DEFAULT_ATOL = 1e-5
GLOBAL_DEFAULT_RTOL = 1e-5


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None

    try:
        value = float(raw)
    except ValueError as exc:
        msg = f"Invalid value for {name}: {raw!r}. Use a non-negative float."
        raise ValueError(msg) from exc

    if value < 0.0:
        msg = f"Invalid value for {name}: {raw!r}. Use a non-negative float."
        raise ValueError(msg)

    return value


def _tolerances() -> tuple[float, float]:
    atol = _env_float("ANEMOI_DISTRIBUTED_TEST_ATOL")
    rtol = _env_float("ANEMOI_DISTRIBUTED_TEST_RTOL")
    return (
        GLOBAL_DEFAULT_ATOL if atol is None else atol,
        GLOBAL_DEFAULT_RTOL if rtol is None else rtol,
    )


def _training_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "src" / "anemoi" / "training" / "config"


def _build_edge_data(src_size: int, dst_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.arange(src_size).repeat_interleave(dst_size)
    dst = torch.arange(dst_size).repeat(src_size)
    edge_index = torch.stack((src, dst), dim=0)
    edge_attr = ((src.float() + 1.0) / (src_size + 1.0) + (dst.float() + 1.0) / (dst_size + 1.0)).unsqueeze(-1)
    return edge_index, edge_attr


def _build_tiny_graph() -> HeteroData:
    num_data_nodes = 8
    num_hidden_nodes = 8
    graph = HeteroData()
    coords_data = torch.linspace(-1.0, 1.0, num_data_nodes)
    coords_hidden = torch.linspace(-0.8, 0.8, num_hidden_nodes)
    graph["data"].x = torch.stack((coords_data, coords_data.flip(0)), dim=1)
    graph["hidden"].x = torch.stack((coords_hidden, coords_hidden.flip(0)), dim=1)

    edge_index_dh, edge_attr_dh = _build_edge_data(num_data_nodes, num_hidden_nodes)
    edge_index_hh, edge_attr_hh = _build_edge_data(num_hidden_nodes, num_hidden_nodes)
    edge_index_hd, edge_attr_hd = _build_edge_data(num_hidden_nodes, num_data_nodes)

    graph[("data", "to", "hidden")].edge_index = edge_index_dh
    graph[("data", "to", "hidden")].edge_attr = edge_attr_dh
    graph[("hidden", "to", "hidden")].edge_index = edge_index_hh
    graph[("hidden", "to", "hidden")].edge_attr = edge_attr_hh
    graph[("hidden", "to", "data")].edge_index = edge_index_hd
    graph[("hidden", "to", "data")].edge_attr = edge_attr_hd
    return graph


def _build_tiny_model() -> AnemoiModelEncProcDec:
    num_heads = 8
    overrides = [
        "model=graphtransformer",
        "training.multistep_input=1",
        "training.multistep_output=1",
        "model.num_channels=16",
        "model.bounding=[]",
        "model.trainable_parameters.hidden=1",
        "model.attributes.edges=[edge_attr]",
        "model.processor.trainable_size=0",
        "model.processor.num_layers=1",
        "model.processor.num_chunks=1",
        f"model.processor.num_heads={num_heads}",
        "model.processor.mlp_hidden_ratio=2",
        "model.processor.qk_norm=False",
        "model.processor.graph_attention_backend=pyg",
        "model.processor.edge_pre_mlp=False",
        "model.encoder.trainable_size=0",
        "model.encoder.num_chunks=1",
        f"model.encoder.num_heads={num_heads}",
        "model.encoder.mlp_hidden_ratio=2",
        "model.encoder.qk_norm=False",
        "model.encoder.shard_strategy=heads",
        "model.encoder.graph_attention_backend=pyg",
        "model.encoder.edge_pre_mlp=False",
        "model.decoder.trainable_size=0",
        "model.decoder.num_chunks=1",
        f"model.decoder.num_heads={num_heads}",
        "model.decoder.mlp_hidden_ratio=2",
        "model.decoder.qk_norm=False",
        "model.decoder.shard_strategy=heads",
        "model.decoder.graph_attention_backend=pyg",
        "model.decoder.edge_pre_mlp=False",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_training_config_dir()),
        job_name="test_strategy_tiny_model",
    ):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved_cfg, dict)

    data_indices = {
        "data": IndexCollection(
            data_config=OmegaConf.create(
                {"forcing": [], "diagnostic": [], "target": []},
            ),
            name_to_index={"var0": 0, "var1": 1},
        ),
    }
    return AnemoiModelEncProcDec(
        model_config=resolved_cfg,
        data_indices=data_indices,
        statistics={"data": None},
        graph_data=_build_tiny_graph(),
    )


def _compute_full_model_grads(
    model: AnemoiModelEncProcDec,
) -> dict[str, torch.Tensor | None]:
    model.zero_grad(set_to_none=True)
    num_nodes = model._graph_data["data"].num_nodes
    x = torch.arange(num_nodes * 2, dtype=torch.float32).reshape(1, 1, 1, num_nodes, 2) / 16.0
    y = model({"data": x})["data"]
    loss = y.pow(2).mean()
    loss.backward()
    return {name: (None if p.grad is None else p.grad.detach().clone()) for name, p in model.named_parameters()}


def test_register_gradient_scaling_hooks_tiny_model_parameter_selection() -> None:
    base = _build_tiny_model()
    hooked = copy.deepcopy(base)
    scale = 4.0

    expected_skipped = {
        "node_attributes.trainable_tensors.data.trainable",
        "node_attributes.trainable_tensors.hidden.trainable",
    }
    actual_skipped = {name for name, _ in base.named_parameters() if "trainable" in name or "no_gradscaling" in name}
    assert actual_skipped == expected_skipped

    base_grads = _compute_full_model_grads(base)
    register_gradient_scaling_hooks(hooked, model_comm_group_size=scale)
    hooked_grads = _compute_full_model_grads(hooked)
    atol, rtol = _tolerances()

    for name, grad in base_grads.items():
        if grad is None:
            assert hooked_grads[name] is None
            continue
        if name in expected_skipped:
            torch.testing.assert_close(hooked_grads[name], grad, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(
                hooked_grads[name],
                grad * scale,
                atol=atol,
                rtol=rtol,
            )
