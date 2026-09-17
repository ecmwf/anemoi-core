# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import timedelta
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.training.schemas.training import CorrectorSchema
from anemoi.training.train.methods.corrector import CorrectorMLP
from anemoi.training.train.methods.corrector import InstrumentCorrectors
from anemoi.training.train.methods.da_single import DASingleTraining


def test_corrector_mlp_zero_initialised() -> None:
    mlp = CorrectorMLP(n_target=3, n_corrector=2, hidden_dim=8)
    y = torch.randn(2, 4, 3)
    corr = torch.randn(2, 4, 2)
    # zero-init output head -> no correction at initialisation
    assert torch.allclose(mlp(y, corr), torch.zeros_like(y))


@pytest.fixture(
    params=[("GraphTransformerProcessor", "edges"), ("GraphTransformerProcessor", "heads"), ("GNNProcessor", None)],
    ids=["gt_edges", "gt_heads", "gnn"],
)
def processor_config(request: pytest.FixtureRequest) -> DictConfig:
    processor_name, shard_strategy = request.param
    config = {
        "_target_": f"anemoi.models.layers.processor.{processor_name}",
        "num_layers": 1,
        "num_chunks": 1,
        "mlp_hidden_ratio": 2,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": {},
        "trainable_size": 0,
        "sub_graph_edge_attributes": ["edge_attr"],
    }
    if processor_name == "GraphTransformerProcessor":
        config.update(num_heads=4, qk_norm=False, graph_attention_backend="pyg", shard_strategy=shard_strategy)
    else:
        config.update(mlp_extra_layers=0)
    return DictConfig(config)


def _make_graph(node_name: str = "data", *, bidirectional: bool = False) -> HeteroData:
    graph = HeteroData()
    graph[node_name].num_nodes = 5
    # Deliberately unsorted ring: every node receives its predecessor's message.
    edges = graph[node_name, "to", node_name]
    edges.edge_index = torch.tensor([[2, 0, 4, 1, 3], [3, 1, 0, 2, 4]])
    edges.edge_attr = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
    if bidirectional:
        edges.edge_index = torch.cat([edges.edge_index, edges.edge_index.flip(0)], dim=1)
        edges.edge_attr = edges.edge_attr.repeat(2, 1)
    # Self edges give attention a choice between local and neighbour features.
    edges.edge_index = torch.cat([edges.edge_index, torch.arange(5).repeat(2, 1)], dim=1)
    edges.edge_attr = torch.cat([edges.edge_attr, torch.zeros(5, 2)], dim=0)
    return graph


def _make_processor_corrector(processor_config: DictConfig) -> InstrumentCorrectors:
    # Exercise schema validation and serialisation before Hydra instantiation.
    config = CorrectorSchema(
        type="processor",
        hidden_dim=8,
        processor=processor_config,
        instrument_groups={
            "hirs": {"corrector_variables": ["geom"]},
            "mwt": {"corrector_variables": ["angle"]},
        },
    )
    provider = create_graph_provider(
        graph=_make_graph(bidirectional=processor_config.get("shard_strategy") == "edges")["data", "to", "data"],
        edge_attributes=config.processor.sub_graph_edge_attributes,
        src_size=5,
        dst_size=5,
        trainable_size=config.processor.trainable_size,
    )
    return InstrumentCorrectors(
        instrument_groups={name: group.model_dump() for name, group in config.instrument_groups.items()},
        all_corrector_names=["geom", "angle"],
        output_name_to_index={"z": 0, "hirs_1": 1, "mwt_1": 2},
        hidden_dim=config.hidden_dim,
        corrector_type=config.type,
        processor_config=DictConfig(config.processor.model_dump(by_alias=True)),
        graph_provider=provider,
    )


def _activate_heads(corrector: InstrumentCorrectors) -> None:
    with torch.no_grad():
        for network in corrector.correctors.values():
            network.out.weight.normal_(std=0.2)


def test_processor_corrector_initial_identity_and_head_gradient(processor_config: DictConfig) -> None:
    corrector = _make_processor_corrector(processor_config)
    pred = torch.randn(2, 1, 1, 5, 3, requires_grad=True)
    metadata = torch.randn(2, 1, 1, 5, 2)
    out = corrector(pred, metadata)
    torch.testing.assert_close(out, pred, atol=0, rtol=0)
    out[..., 1:].sum().backward()
    assert all(network.out.bias.grad.abs().sum() > 0 for network in corrector.correctors.values())
    torch.testing.assert_close(pred.grad[..., 1:], torch.ones_like(pred[..., 1:]))


def test_processor_corrector_targeting_and_neighbour_gradients(processor_config: DictConfig) -> None:
    torch.manual_seed(7)
    corrector = _make_processor_corrector(processor_config)
    _activate_heads(corrector)
    pred = torch.randn(1, 1, 1, 5, 3, requires_grad=True)
    metadata = torch.randn(1, 1, 1, 5, 2, requires_grad=True)
    out = corrector(pred, metadata)
    torch.testing.assert_close(out[..., 0], pred[..., 0], atol=0, rtol=0)
    assert not torch.allclose(out[..., 1:], pred[..., 1:])

    # HIRS node 1 reads node 0, but neither distant nodes nor MWT inputs.
    out[0, 0, 0, 1, 1].backward()
    assert pred.grad[0, 0, 0, 0, 1].abs() > 0
    assert metadata.grad[0, 0, 0, 0, 0].abs() > 0
    assert torch.count_nonzero(pred.grad[..., 3:, :]) == 0
    assert torch.count_nonzero(pred.grad[..., [0, 2]]) == 0
    assert torch.count_nonzero(metadata.grad[..., 1]) == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in corrector.correctors["mwt"].parameters())


def test_processor_corrector_instances_match_independent_graphs(processor_config: DictConfig) -> None:
    corrector = _make_processor_corrector(processor_config)
    _activate_heads(corrector)
    pred = torch.randn(2, 3, 2, 5, 3)
    metadata = torch.randn(2, 3, 2, 5, 2)
    batched = corrector(pred, metadata)
    for batch, time, ensemble in product(range(2), range(3), range(2)):
        instance = (slice(batch, batch + 1), slice(time, time + 1), slice(ensemble, ensemble + 1))
        expected = corrector(pred[instance], metadata[instance])
        torch.testing.assert_close(batched[instance], expected)


@pytest.mark.parametrize("processor_config", [("GraphTransformerProcessor", "edges")], indirect=True)
def test_edge_sharded_gt_corrector_requires_bidirectional_connectivity(processor_config: DictConfig) -> None:
    provider = create_graph_provider(
        graph=_make_graph()["data", "to", "data"],
        edge_attributes=["edge_attr"],
        src_size=5,
        dst_size=5,
        trainable_size=0,
    )
    corrector = InstrumentCorrectors(
        instrument_groups={"hirs": {"corrector_variables": ["geom"]}},
        all_corrector_names=["geom"],
        output_name_to_index={"hirs_1": 0},
        hidden_dim=8,
        corrector_type="processor",
        processor_config=processor_config,
        graph_provider=provider,
    )
    group = Mock()
    group.size.return_value = 2
    group.rank.return_value = 0
    with pytest.raises(ValueError, match="bidirectional graph connectivity"):
        corrector(
            torch.randn(1, 1, 1, 3, 1),
            torch.randn(1, 1, 1, 3, 1),
            model_comm_group=group,
            grid_shard_sizes=[3, 2],
        )


def test_processor_corrector_shared_provider_checkpoint_gradients(processor_config: DictConfig) -> None:
    processor_config.trainable_size = 2
    corrector = _make_processor_corrector(processor_config)
    _activate_heads(corrector)
    pred = torch.randn(1, 1, 1, 5, 3, requires_grad=True)
    metadata = torch.randn(1, 1, 1, 5, 2, requires_grad=True)
    eager = corrector(pred, metadata)
    eager.square().sum().backward()
    expected = {name: p.grad.clone() for name, p in corrector.named_parameters()}
    expected_pred_grad = pred.grad.clone()
    expected_metadata_grad = metadata.grad.clone()
    assert expected["graph_provider.trainable.trainable"].abs().sum() > 0

    corrector.zero_grad(set_to_none=True)
    pred.grad = None
    metadata.grad = None
    recomputed = checkpoint(corrector, pred, metadata, use_reentrant=False)
    recomputed.square().sum().backward()
    torch.testing.assert_close(recomputed, eager)
    torch.testing.assert_close(pred.grad, expected_pred_grad)
    torch.testing.assert_close(metadata.grad, expected_metadata_grad)
    for name, parameter in corrector.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name])


@pytest.mark.parametrize(
    ("batch_size", "local_nodes", "shard_sizes", "message"),
    [
        (2, 3, [3, 2], "batch_size=1"),
        (1, 3, None, "graph provider partitions"),
        (1, 2, [2, 3], "graph provider partitions"),
        (1, 2, [3, 2], "local grid nodes"),
    ],
)
def test_processor_corrector_validates_shards_before_communication(
    processor_config: DictConfig,
    batch_size: int,
    local_nodes: int,
    shard_sizes: list[int] | None,
    message: str,
) -> None:
    corrector = _make_processor_corrector(processor_config)
    group = Mock()
    group.size.return_value = 2
    group.rank.return_value = 0
    with pytest.raises(ValueError, match=message):
        corrector(
            torch.randn(batch_size, 1, 1, local_nodes, 3),
            torch.randn(batch_size, 1, 1, local_nodes, 2),
            model_comm_group=group,
            grid_shard_sizes=shard_sizes,
        )


@pytest.mark.parametrize("node_name", ["data", "observations"])
def test_da_initialises_processor_corrector(processor_config: DictConfig, node_name: str) -> None:
    config = DictConfig(
        {
            "training": {
                "corrector": {
                    "type": "processor",
                    "hidden_dim": 8,
                    "processor": processor_config,
                    "instrument_groups": {"hirs": {"corrector_variables": ["geom"]}},
                },
            },
        },
    )
    indices = SimpleNamespace(
        data=SimpleNamespace(input=SimpleNamespace(corrector=[1], name_to_index={"hirs_1": 0, "geom": 1})),
        model=SimpleNamespace(output=SimpleNamespace(name_to_index={"hirs_1": 0})),
    )
    module = SimpleNamespace(
        config=config,
        dataset_names=["observations"],
        target_dataset_names=["observations"],
        data_indices={"observations": indices},
        corrector=torch.nn.ModuleDict(),
    )
    DASingleTraining._init_correctors(module, _make_graph(node_name))
    pred = torch.randn(1, 1, 1, 5, 1)
    torch.testing.assert_close(module.corrector["observations"](pred, torch.randn_like(pred)), pred)


def _check_distributed_corrector(rank: int, config: dict, store_path: str, backend: str) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(19)
        reference = _make_processor_corrector(DictConfig(config)).to(device)
        _activate_heads(reference)
        sharded = _make_processor_corrector(DictConfig(config)).to(device)
        sharded.load_state_dict(reference.state_dict())
        pred = torch.randn(1, 1, 1, 5, 3, device=device, requires_grad=True)
        metadata = torch.randn(1, 1, 1, 5, 2, device=device, requires_grad=True)
        expected = reference(pred, metadata)
        expected.square().sum().backward()

        node_slice = slice(0, 3) if rank == 0 else slice(3, 5)
        local_pred = pred[..., node_slice, :].detach().clone().requires_grad_()
        local_metadata = metadata[..., node_slice, :].detach().clone().requires_grad_()
        actual = sharded(
            local_pred,
            local_metadata,
            model_comm_group=dist.group.WORLD,
            grid_shard_sizes=[3, 2],
        )
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected[..., node_slice, :], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(local_pred.grad, pred.grad[..., node_slice, :], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(local_metadata.grad, metadata.grad[..., node_slice, :], atol=1e-5, rtol=1e-5)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in sharded.named_parameters():
            dist.all_reduce(parameter.grad)
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, atol=1e-5, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def test_processor_corrector_distributed_matches_unsharded(processor_config: DictConfig, tmp_path: Path) -> None:
    backend = "gloo"
    if processor_config._target_.endswith("GraphTransformerProcessor"):
        if torch.cuda.device_count() < 2:
            pytest.skip("GT halo exchange requires two CUDA devices and NCCL")
        backend = "nccl"
    mp.spawn(
        _check_distributed_corrector,
        args=(OmegaConf.to_container(processor_config), str(tmp_path / "process_group"), backend),
        nprocs=2,
        join=True,
    )


def test_instrument_corrector_additive_and_targeted() -> None:
    # two output channels named hirs_1, hirs_2 plus an unrelated z; corrector var geom
    output_name_to_index = {"z": 0, "hirs_1": 1, "hirs_2": 2}
    corrector = InstrumentCorrectors(
        instrument_groups={"hirs": {"corrector_variables": ["geom"], "channels": None}},
        all_corrector_names=["geom"],
        output_name_to_index=output_name_to_index,
        hidden_dim=8,
        corrector_type="mlp",
    )
    y_pred = torch.randn(2, 5, 3)
    corrector_vars = torch.randn(2, 5, 1)
    out = corrector(y_pred, corrector_vars)
    # zero-init -> identity at start, but the non-targeted channel must always pass through
    assert torch.allclose(out[..., 0], y_pred[..., 0])
    assert torch.allclose(out, y_pred)  # zero-init correction

    # after perturbing the output head, only hirs channels change
    corrector.correctors["hirs"].out.weight.data.fill_(0.1)
    corrector.correctors["hirs"].out.bias.data.fill_(0.5)
    out2 = corrector(y_pred, corrector_vars)
    assert torch.allclose(out2[..., 0], y_pred[..., 0])  # z untouched
    assert not torch.allclose(out2[..., 1], y_pred[..., 1])  # hirs_1 corrected


def test_instrument_corrector_prefix_matching() -> None:
    output_name_to_index = {"mwt_1": 0, "mwt_2": 1, "other": 2}
    corrector = InstrumentCorrectors(
        instrument_groups={"mwt": {"corrector_variables": ["v"], "channels": None}},
        all_corrector_names=["v", "unused"],
        output_name_to_index=output_name_to_index,
        hidden_dim=4,
        corrector_type="mlp",
    )
    target_idx = corrector._target_idx_mwt.tolist()
    assert target_idx == [0, 1]  # mwt_1, mwt_2 matched by prefix, not "other"
