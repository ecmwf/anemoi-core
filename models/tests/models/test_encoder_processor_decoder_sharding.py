# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Test sharding orchestration through a minimal real GraphConv model.

Each rank supplies owned or replicated data-grid inputs. The normal model forward prepares
hidden-node shards, graph-provider edge shards, encoder/processor/decoder layouts,
and the prediction in the supplied external layout with its input residual.
Compare predictions, the global weighted loss, input gradients and all parameter
gradients with an unsharded model.

Trainable node and edge tables have row-varying values. Their reconstructed
gradients are compared directly; ordinary layer-gradient contributions are summed.
Run with --distributed and the selected backend/world size. Mapper/processor
checkpointing is disabled; graph providers retain their normal checkpoint behavior.
"""

import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from tests.distributed._distributed_runner import _run_distributed_test
from tests.distributed._graph_fixtures import build_model_graph


def _build_graphconv_config() -> DictConfig:
    """Minimal production configuration with compatible eight-channel latent features."""
    component_options = {
        "num_channels": 8,
        "num_chunks": 1,
        "mlp_extra_layers": 0,
        "gradient_checkpointing": False,
        "layer_kernels": {},
    }
    # The model reads provider settings from the same mapper/processor config entries.
    graph_provider_options = {
        "sub_graph_edge_attributes": ["edge_attr"],
        "trainable_size": 2,
    }
    return OmegaConf.create(
        {
            "model": {"hidden_nodes_name": "hidden", "latent_skip": False},
            "node_trainable_parameters": {"data": 2, "hidden": 2},
            "encoders": {
                "encoder": {
                    "source_datasets": ["data"],
                    "dataset_fusing_strategy": "not_supported",
                    "mapper": {
                        "_target_": "anemoi.models.layers.mapper.GNNForwardMapper",
                        **component_options,
                        **graph_provider_options,
                    },
                }
            },
            "processor": {
                "_target_": "anemoi.models.layers.processor.GNNProcessor",
                "num_layers": 1,
                **component_options,
                **graph_provider_options,
            },
            "latent_aggregator": {"_target_": "anemoi.models.layers.aggregator.SumAggregator"},
            "decoders": {
                "decoder": {
                    "target_datasets": ["data"],
                    "target_node_features": ["encoded_data"],
                    "mapper": {
                        "_target_": "anemoi.models.layers.mapper.GNNBackwardMapper",
                        **component_options,
                        **graph_provider_options,
                    },
                }
            },
            "residual": {"datasets": {"data": {"_target_": "anemoi.models.layers.residual.SkipConnection"}}},
            "bounding": {"datasets": {"data": []}},
        }
    )


def _build_graphconv_model(graph: HeteroData) -> AnemoiModelEncProcDec:
    """Construct the real model with two prognostic variables and one input/output step."""
    indices = IndexCollection(OmegaConf.create({"forcing": [], "diagnostic": []}), {"temperature": 0, "wind": 1})
    model = AnemoiModelEncProcDec(
        model_config=_build_graphconv_config(),
        data_indices={"data": indices},
        statistics={"data": None},
        n_step_input=1,
        n_step_output=1,
        graph_data=graph,
    )
    # Two variables + four coordinate features + two trainable data features = eight.
    assert (
        model.input_dim["data"] == model.target_dim["data"] == model.latent_aggregator.hidden_dim == 8
    ), "This fixture requires assembled data inputs, decoder target features and hidden latent features to have width eight"
    return model


def _test_graphconv_model_matches_unsharded_rank(
    *, rank: int, world_size: int, device: torch.device, group: dist.ProcessGroup, input_is_sharded: bool
) -> None:
    """Compare normal model orchestration with one globally normalized reference objective."""
    torch.set_num_threads(1)
    graph, data_splits = build_model_graph(world_size)
    # Ensure each relation exercises a remote read on this rank.
    hidden_splits = get_shard_sizes(graph["hidden"].x, 0, group)
    for src_name, dst_name in (("data", "hidden"), ("hidden", "hidden"), ("hidden", "data")):
        src_splits = data_splits if src_name == "data" else hidden_splits
        dst_splits = data_splits if dst_name == "data" else hidden_splits
        relation = (src_name, "to", dst_name)
        edge_index = graph[relation].edge_index
        assert (edge_index[1].diff() >= 0).all(), f"{relation} must be destination-sorted"
        src_start, src_stop = sum(src_splits[:rank]), sum(src_splits[: rank + 1])
        dst_start, dst_stop = sum(dst_splits[:rank]), sum(dst_splits[: rank + 1])
        sources = edge_index[0, (edge_index[1] >= dst_start) & (edge_index[1] < dst_stop)]
        assert (
            (sources < src_start) | (sources >= src_stop)
        ).any(), f"{relation} must have a remote source dependency for rank {rank}"
    torch.manual_seed(43)
    reference_model = _build_graphconv_model(graph).to(device)
    # Row-varying learned features expose ownership/order mistakes hidden by
    # default zero tables. Copy this initialized state to the distributed model.
    with torch.no_grad():
        for module in reference_model.modules():
            if isinstance(module, TrainableTensor) and module.trainable is not None:
                table = module.trainable
                values = torch.linspace(0.01, 0.1, table.numel(), dtype=table.dtype, device=table.device)
                table.copy_(values.reshape_as(table))
    distributed_model = _build_graphconv_model(graph).to(device)
    distributed_model.load_state_dict(reference_model.state_dict())
    reference_params = dict(reference_model.named_parameters())
    distributed_params = dict(distributed_model.named_parameters())
    # Tables are evaluated before production node/edge sharding. The surrounding
    # sharding operations gather their gradients in backward; TrainableTensor
    # itself does not communicate. Other parameters operate on owned work.
    table_modules = (
        distributed_model.node_attributes.trainable_tensors["data"],
        distributed_model.node_attributes.trainable_tensors["hidden"],
        distributed_model.encoder_graph_provider["data"].trainable,
        distributed_model.processor_graph_provider.trainable,
        distributed_model.decoder_graph_provider["data"].trainable,
    )
    assert all(
        isinstance(module, TrainableTensor) and module.trainable is not None for module in table_modules
    ), "Both node tables and all three relation tables must be enabled TrainableTensor modules"
    table_parameter_ids = {id(module.trainable) for module in table_modules}
    assert table_parameter_ids == {
        id(module.trainable) for module in distributed_model.modules() if isinstance(module, TrainableTensor)
    }, "The five listed tables must account for every TrainableTensor in the model"
    table_names = {name for name, param in distributed_params.items() if id(param) in table_parameter_ids}
    layer_names = set(distributed_params) - table_names
    assert len(table_names) == len(table_modules), "Each table must map to a named model parameter"
    assert layer_names, "The fixture must also exercise ordinary layer parameters"

    generator = torch.Generator().manual_seed(47)
    inputs = torch.randn(1, 1, 1, sum(data_splits), 2, generator=generator).to(device)
    target = torch.randn(inputs.shape, generator=generator).to(device)
    weights = (torch.arange(inputs.numel(), device=device).reshape_as(inputs) % 7 + 1).float() / 7

    # Reference execution uses all external inputs and no communication group.
    reference_input = inputs.clone().requires_grad_(True)
    reference_outputs = reference_model({"data": reference_input})
    assert set(reference_outputs) == {"data"}, "The reference must return the configured data dataset"
    reference_output = reference_outputs["data"]
    assert reference_output.shape == inputs.shape, "The reference prediction must preserve the data-grid layout"
    reference_loss = (weights * (reference_output - target).square()).sum() / weights.numel()
    reference_loss.backward()

    # Anemoi prepares hidden/edge shards and component layouts in both external modes.
    start = sum(data_splits[:rank])
    stop = start + data_splits[rank]
    distributed_input = (inputs[..., start:stop, :] if input_is_sharded else inputs).clone().requires_grad_(True)
    distributed_outputs = distributed_model(
        {"data": distributed_input},
        model_comm_group=group,
        grid_shard_sizes={"data": data_splits} if input_is_sharded else None,
    )
    assert set(distributed_outputs) == {"data"}, "Each rank must return the configured data dataset"
    distributed_output = distributed_outputs["data"]
    assert (
        distributed_output.shape == distributed_input.shape
    ), "Predictions must match the supplied external data-grid layout"
    if input_is_sharded:
        local_loss = (
            weights[..., start:stop, :] * (distributed_output - target[..., start:stop, :]).square()
        ).sum() / weights.numel()
    else:
        # Gather backward splits model sensitivities; the full residual loss also
        # gives the replicated input its complete direct-path gradient on every rank.
        local_loss = (weights * (distributed_output - target).square()).sum() / weights.numel()
    local_loss.backward()

    # Callable messages add context without replacing PyTorch's numerical diagnostics.
    output_tolerances = {"rtol": 1e-4, "atol": 1e-5}
    gradient_tolerances = {"rtol": 1e-4, "atol": 1e-7}
    torch.testing.assert_close(
        distributed_output,
        reference_output[..., start:stop, :] if input_is_sharded else reference_output,
        **output_tolerances,
        msg=lambda details: f"prediction on rank {rank}\n{details}",
    )
    global_loss = local_loss.detach().clone()
    if input_is_sharded:
        dist.all_reduce(global_loss, group=group)
    torch.testing.assert_close(
        global_loss,
        reference_loss.detach(),
        **output_tolerances,
        msg=lambda details: f"global loss on rank {rank}\n{details}",
    )
    assert reference_input.grad is not None, "Missing reference input gradient"
    assert distributed_input.grad is not None, "Missing distributed input gradient"
    torch.testing.assert_close(
        distributed_input.grad,
        reference_input.grad[..., start:stop, :] if input_is_sharded else reference_input.grad,
        **gradient_tolerances,
        msg=lambda details: f"input gradient on rank {rank}\n{details}",
    )
    for name, param in distributed_params.items():
        assert param.grad is not None, f"Missing distributed gradient of {name}"
        reference_grad = reference_params[name].grad
        assert reference_grad is not None, f"Missing reference gradient of {name}"
        if name in table_names:
            # A near-zero table gradient would not meaningfully check reconstruction.
            allowed_error = gradient_tolerances["atol"] + gradient_tolerances["rtol"] * reference_grad.abs()
            assert (reference_grad.abs() > 10 * allowed_error).any(), f"Insufficient gradient signal for {name}"
        grad = param.grad.detach().clone()
        if name in layer_names:
            dist.all_reduce(grad, group=group)
        torch.testing.assert_close(
            grad,
            reference_grad,
            **gradient_tolerances,
            msg=lambda details, name=name: f"gradient of {name} on rank {rank}\n{details}",
        )


@pytest.mark.distributed
@pytest.mark.parametrize("input_is_sharded", [True, False], ids=["sharded-input", "replicated-input"])
def test_graphconv_model_matches_unsharded(
    distributed_backend: str, distributed_world_size: int, input_is_sharded: bool
) -> None:
    """Model orchestration preserves predictions and gradients under the external input layout."""
    _run_distributed_test(
        _test_graphconv_model_matches_unsharded_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
        input_is_sharded=input_is_sharded,
    )
