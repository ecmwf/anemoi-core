# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Test sharding orchestration through a minimal real GraphConv model.

Each rank supplies its owned data-grid input. The normal model forward prepares
hidden-node shards, graph-provider edge shards, encoder/processor/decoder layouts,
and the local prediction with its input residual. Compare predictions, the global
weighted loss, input gradients and all parameter gradients with an unsharded model.

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
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from tests.distributed._distributed_runner import _run_distributed_test


def _build_model_graph(world_size: int) -> tuple[HeteroData, list[int]]:
    """Build data/hidden populations and destination-sorted edges for each model stage.

    Every destination uses an owned source and a source from the next rank.
    Different population sizes give uneven partitions and different endpoint offsets.
    """
    data_splits = [4] + [3] * (world_size - 1)
    hidden_splits = [3] + [2] * (world_size - 1)
    graph = HeteroData()
    generator = torch.Generator().manual_seed(31)
    for name, splits in (("data", data_splits), ("hidden", hidden_splits)):
        graph[name].x = torch.randn(sum(splits), 2, generator=generator)
        graph[name].num_nodes = sum(splits)

    relations = (
        ("data", "hidden", data_splits, hidden_splits),
        ("hidden", "hidden", hidden_splits, hidden_splits),
        ("hidden", "data", hidden_splits, data_splits),
    )
    for src_name, dst_name, src_splits, dst_splits in relations:
        src_starts = [sum(src_splits[:rank]) for rank in range(world_size)]
        dst_starts = [sum(dst_splits[:rank]) for rank in range(world_size)]
        edges = []
        for rank in range(world_size):
            peer = (rank + 1) % world_size
            for offset in range(dst_splits[rank]):
                dst = dst_starts[rank] + offset
                edges.extend(
                    [
                        (src_starts[rank] + offset % src_splits[rank], dst),
                        (src_starts[peer] + offset % src_splits[peer], dst),
                    ]
                )
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        relation = (src_name, "to", dst_name)
        graph[relation].edge_index = edge_index
        graph[relation].edge_attr = torch.randn(len(edges), 3, generator=generator)
        assert (edge_index[1].diff() >= 0).all(), f"{relation} must be destination-sorted"
        for rank in range(world_size):
            local_edges = (edge_index[1] >= dst_starts[rank]) & (edge_index[1] < dst_starts[rank] + dst_splits[rank])
            sources = edge_index[0, local_edges]
            assert (
                (sources < src_starts[rank]) | (sources >= src_starts[rank] + src_splits[rank])
            ).any(), f"{relation} must have a remote source dependency for rank {rank}"
    return graph, data_splits


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
    *, rank: int, world_size: int, device: torch.device, group: dist.ProcessGroup
) -> None:
    """Compare normal model orchestration with one globally normalized reference objective."""
    torch.set_num_threads(1)
    # Initialize independent models with identical state and distinguishable table rows.
    graph, data_splits = _build_model_graph(world_size)
    torch.manual_seed(43)
    reference_model = _build_graphconv_model(graph).to(device)
    # Distinguish learned-feature rows in forward, rather than testing only
    # TrainableTensor's default zero initialization. Populate before copying state.
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
    assert (
        reference_params.keys() == distributed_params.keys()
    ), "Reference and distributed models must have identical parameter names"
    # Callable messages prepend context without suppressing PyTorch's mismatch details.
    for name, param in reference_params.items():
        torch.testing.assert_close(
            distributed_params[name],
            param,
            rtol=0,
            atol=0,
            msg=lambda details, name=name: f"initial {name} on rank {rank}\n{details}",
        )
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

    # Distributed execution starts with externally owned input rows. The model
    # prepares hidden shards, edge shards, component layouts and residual output.
    start = sum(data_splits[:rank])
    stop = start + data_splits[rank]
    local_input = inputs[..., start:stop, :].clone().requires_grad_(True)
    distributed_outputs = distributed_model(
        {"data": local_input}, model_comm_group=group, grid_shard_sizes={"data": data_splits}
    )
    assert set(distributed_outputs) == {"data"}, "Each rank must return the configured data dataset"
    local_output = distributed_outputs["data"]
    assert local_output.shape == local_input.shape, "Each rank must predict only its owned data-grid rows"
    local_loss = (
        weights[..., start:stop, :] * (local_output - target[..., start:stop, :]).square()
    ).sum() / weights.numel()
    local_loss.backward()

    # Compare output/loss, owned input gradients, and complete/partial parameter gradients.
    output_tolerances = {"rtol": 1e-4, "atol": 1e-5}
    gradient_tolerances = {"rtol": 1e-4, "atol": 1e-7}
    torch.testing.assert_close(
        local_output,
        reference_output[..., start:stop, :],
        **output_tolerances,
        msg=lambda details: f"prediction on rank {rank}\n{details}",
    )
    global_loss = local_loss.detach().clone()
    dist.all_reduce(global_loss, group=group)
    torch.testing.assert_close(global_loss, reference_loss.detach(), **output_tolerances)
    assert reference_input.grad is not None, "Missing reference input gradient"
    assert local_input.grad is not None, "Missing distributed input gradient"
    torch.testing.assert_close(
        local_input.grad,
        reference_input.grad[..., start:stop, :],
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
def test_graphconv_model_matches_unsharded(distributed_backend: str, distributed_world_size: int) -> None:
    """A complete GraphConv model preserves predictions and gradients with sharded external inputs."""
    _run_distributed_test(
        _test_graphconv_model_matches_unsharded_rank,
        backend=distributed_backend,
        world_size=distributed_world_size,
    )
