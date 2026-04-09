# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from dataclasses import field

import pytest
import torch

from anemoi.models.layers.mapper import PointWiseBackwardMapper
from anemoi.models.layers.mapper import PointWiseForwardMapper
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.utils.config import DotDict


@dataclass
class PointWiseMapperConfig:
    in_channels_src: int = 3
    in_channels_dst: int = 2
    hidden_dim: int = 8
    out_channels_dst: int = 5
    cpu_offload: bool = False
    gradient_checkpointing: bool = True
    layer_kernels: field(default_factory=DotDict) = None

    def __post_init__(self):
        self.layer_kernels = load_layer_kernels(instance=False)


@pytest.fixture
def mapper_init():
    return PointWiseMapperConfig()


@pytest.fixture
def pointwise_pair(mapper_init, device):
    num_nodes = 4
    return (
        torch.randn(num_nodes, mapper_init.in_channels_src, device=device),
        torch.randn(num_nodes, mapper_init.in_channels_dst, device=device),
    )


def test_pointwise_forward_mapper_only_embeds_source(mapper_init, pointwise_pair, device, monkeypatch):
    mapper = PointWiseForwardMapper(
        in_channels_src=mapper_init.in_channels_src,
        in_channels_dst=mapper_init.in_channels_dst,
        hidden_dim=mapper_init.hidden_dim,
        cpu_offload=mapper_init.cpu_offload,
        gradient_checkpointing=mapper_init.gradient_checkpointing,
        layer_kernels=mapper_init.layer_kernels,
    ).to(device)

    def fail_gather_tensor(*args, **kwargs):
        raise AssertionError("PointWiseForwardMapper should not gather its output.")

    import anemoi.models.layers.mapper as mapper_module

    monkeypatch.setattr(mapper_module, "gather_tensor", fail_gather_tensor)

    shard_shapes = ([list(pointwise_pair[0].shape)], [list(pointwise_pair[1].shape)])
    x_src, x_hidden = mapper.forward(
        pointwise_pair,
        batch_size=1,
        shard_shapes=shard_shapes,
        keep_x_dst_sharded=False,
    )

    assert torch.equal(x_src, pointwise_pair[0])
    assert x_hidden.shape == (pointwise_pair[0].shape[0], mapper_init.hidden_dim)
    assert torch.allclose(x_hidden, mapper.emb_nodes_src(pointwise_pair[0]))


def test_pointwise_forward_mapper_shards_unsharded_input_before_embedding(
    mapper_init, pointwise_pair, device, monkeypatch
):
    mapper = PointWiseForwardMapper(
        in_channels_src=mapper_init.in_channels_src,
        in_channels_dst=mapper_init.in_channels_dst,
        hidden_dim=mapper_init.hidden_dim,
        cpu_offload=mapper_init.cpu_offload,
        gradient_checkpointing=mapper_init.gradient_checkpointing,
        layer_kernels=mapper_init.layer_kernels,
    ).to(device)

    shard_shapes = (
        [[2, mapper_init.in_channels_src], [2, mapper_init.in_channels_src]],
        [[2, mapper_init.in_channels_dst], [2, mapper_init.in_channels_dst]],
    )
    fake_group = object()
    calls = {"shard": 0}

    def fake_shard_tensor(input_, dim, shapes, mgroup, gather_in_backward=True):
        calls["shard"] += 1
        assert dim == 0
        assert shapes == shard_shapes[0]
        assert mgroup is fake_group
        return input_[: shapes[0][0]]

    import anemoi.models.layers.mapper as mapper_module

    monkeypatch.setattr(mapper_module, "shard_tensor", fake_shard_tensor)

    _, x_hidden = mapper.forward(
        pointwise_pair,
        batch_size=1,
        shard_shapes=shard_shapes,
        model_comm_group=fake_group,
        x_src_is_sharded=False,
        keep_x_dst_sharded=True,
    )

    assert calls["shard"] == 1
    assert x_hidden.shape == (2, mapper_init.hidden_dim)
    assert torch.allclose(x_hidden, mapper.emb_nodes_src(pointwise_pair[0][:2]))


def test_pointwise_forward_mapper_accepts_sharded_source_with_unsharded_destination(mapper_init, device):
    mapper = PointWiseForwardMapper(
        in_channels_src=mapper_init.in_channels_src,
        in_channels_dst=mapper_init.in_channels_dst,
        hidden_dim=mapper_init.hidden_dim,
        cpu_offload=mapper_init.cpu_offload,
        gradient_checkpointing=mapper_init.gradient_checkpointing,
        layer_kernels=mapper_init.layer_kernels,
    ).to(device)

    x_src_local = torch.randn(2, mapper_init.in_channels_src, device=device)
    x_dst_full = torch.randn(4, mapper_init.in_channels_dst, device=device)
    shard_shapes = (
        [[2, mapper_init.in_channels_src], [2, mapper_init.in_channels_src]],
        [[2, mapper_init.in_channels_dst], [2, mapper_init.in_channels_dst]],
    )

    _, x_hidden = mapper.forward(
        (x_src_local, x_dst_full),
        batch_size=1,
        shard_shapes=shard_shapes,
        x_src_is_sharded=True,
        keep_x_dst_sharded=True,
    )

    assert x_hidden.shape == (2, mapper_init.hidden_dim)
    assert torch.allclose(x_hidden, mapper.emb_nodes_src(x_src_local))


def test_pointwise_backward_mapper_only_applies_extractor_and_gather(mapper_init, device, monkeypatch):
    mapper = PointWiseBackwardMapper(
        in_channels_src=mapper_init.hidden_dim,
        in_channels_dst=mapper_init.in_channels_dst,
        hidden_dim=mapper_init.hidden_dim,
        out_channels_dst=mapper_init.out_channels_dst,
        cpu_offload=mapper_init.cpu_offload,
        gradient_checkpointing=mapper_init.gradient_checkpointing,
        layer_kernels=mapper_init.layer_kernels,
    ).to(device)

    x_hidden = torch.randn(4, mapper_init.hidden_dim, device=device)
    x_dst = torch.randn(4, mapper_init.in_channels_dst, device=device)
    shard_shapes = (
        [[2, 1], [2, 1]],
        [[2, mapper_init.in_channels_dst], [2, mapper_init.in_channels_dst]],
    )
    fake_group = object()
    calls = {"shard": 0, "gather": 0}

    def fake_shard_tensor(input_, dim, shapes, mgroup, gather_in_backward=True):
        calls["shard"] += 1
        assert shapes == [[2, mapper_init.hidden_dim], [2, mapper_init.hidden_dim]]
        assert mgroup is fake_group
        return input_[:2]

    def fake_gather_tensor(input_, dim, shapes, mgroup):
        calls["gather"] += 1
        assert dim == 0
        assert shapes == [[2, mapper_init.out_channels_dst], [2, mapper_init.out_channels_dst]]
        assert mgroup is fake_group
        return torch.cat([input_, input_], dim=0)

    import anemoi.models.layers.mapper as mapper_module

    monkeypatch.setattr(mapper_module, "shard_tensor", fake_shard_tensor)
    monkeypatch.setattr(mapper_module, "gather_tensor", fake_gather_tensor)

    x_out = mapper.forward(
        (x_hidden, x_dst),
        batch_size=1,
        shard_shapes=shard_shapes,
        model_comm_group=fake_group,
        x_src_is_sharded=False,
        keep_x_dst_sharded=False,
    )

    assert calls == {"shard": 1, "gather": 1}
    assert x_out.shape == (4, mapper_init.out_channels_dst)
    expected_local = mapper.node_data_extractor(x_hidden[:2])
    assert torch.allclose(x_out, torch.cat([expected_local, expected_local], dim=0))


def test_pointwise_backward_mapper_keeps_output_sharded_when_requested(mapper_init, device):
    mapper = PointWiseBackwardMapper(
        in_channels_src=mapper_init.hidden_dim,
        in_channels_dst=mapper_init.in_channels_dst,
        hidden_dim=mapper_init.hidden_dim,
        out_channels_dst=mapper_init.out_channels_dst,
        cpu_offload=mapper_init.cpu_offload,
        gradient_checkpointing=mapper_init.gradient_checkpointing,
        layer_kernels=mapper_init.layer_kernels,
    ).to(device)

    x_hidden_local = torch.randn(2, mapper_init.hidden_dim, device=device)
    x_dst_full = torch.randn(4, mapper_init.in_channels_dst, device=device)
    shard_shapes = (
        [[2, 1], [2, 1]],
        [[2, mapper_init.in_channels_dst], [2, mapper_init.in_channels_dst]],
    )

    x_out = mapper.forward(
        (x_hidden_local, x_dst_full),
        batch_size=1,
        shard_shapes=shard_shapes,
        x_src_is_sharded=True,
        keep_x_dst_sharded=True,
    )

    assert x_out.shape == (2, mapper_init.out_channels_dst)
    assert torch.allclose(x_out, mapper.node_data_extractor(x_hidden_local))


def test_pointwise_mappers_honor_gradient_checkpointing_config(mapper_init, device):
    forward_mapper = PointWiseForwardMapper(
        in_channels_src=mapper_init.in_channels_src,
        in_channels_dst=mapper_init.in_channels_dst,
        hidden_dim=mapper_init.hidden_dim,
        cpu_offload=mapper_init.cpu_offload,
        gradient_checkpointing=False,
        layer_kernels=mapper_init.layer_kernels,
    ).to(device)
    backward_mapper = PointWiseBackwardMapper(
        in_channels_src=mapper_init.hidden_dim,
        in_channels_dst=mapper_init.in_channels_dst,
        hidden_dim=mapper_init.hidden_dim,
        out_channels_dst=mapper_init.out_channels_dst,
        cpu_offload=mapper_init.cpu_offload,
        gradient_checkpointing=False,
        layer_kernels=mapper_init.layer_kernels,
    ).to(device)

    assert forward_mapper.gradient_checkpointing is False
    assert backward_mapper.gradient_checkpointing is False
