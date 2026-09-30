# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The Triton rotary kernel against the PyTorch version (apply_rotary), forward and backward."""

import pytest
import torch

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.layers.attention import MultiHeadCrossAttention
from anemoi.models.layers.attention import MultiHeadSelfAttention
from anemoi.models.layers.neighbourhood_attention import GridNeighbourhood
from anemoi.models.layers.reduced_grid import ReducedGrid
from anemoi.models.layers.spherical_rotary import SphericalRotaryEmbedding
from anemoi.models.layers.spherical_rotary import apply_rotary
from anemoi.models.layers.spherical_rotary import rotary_angles
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.triton.spherical_rotary import spherical_rotary

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

# The kernel and PyTorch both turn in float32 (float64 for float64) and round once at the end; they may
# still differ in the last bit where one fuses a multiply and an add and the other does not.
TOLERANCE = {
    torch.float64: dict(rtol=1e-12, atol=1e-12),
    torch.float32: dict(rtol=1e-6, atol=1e-6),
    torch.float16: dict(rtol=2**-10, atol=1e-3),
    torch.bfloat16: dict(rtol=2**-7, atol=1e-2),
}


def tables(num_points: int, head_dim: int, max_frequency: float = 100.0, dtype=torch.float32):
    """Cosines and sines of the rotary angles of points spread over the sphere, on the GPU."""
    generator = torch.Generator().manual_seed(num_points)
    lat = torch.rand(num_points, generator=generator) * torch.pi - torch.pi / 2
    lon = torch.rand(num_points, generator=generator) * 2 * torch.pi
    angles = rotary_angles(torch.stack([lat, lon], dim=1), head_dim, max_frequency)
    return torch.cos(angles).to("cuda", dtype), torch.sin(angles).to("cuda", dtype)


def make_input(layout: str, batch: int, heads: int, num_points: int, head_dim: int, dtype, seed: int = 0):
    """A leaf tensor and a (batch, heads, points, channels) view of it in the given memory layout."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "contiguous":
        leaf = torch.randn(batch, heads, num_points, head_dim, generator=generator, device="cuda", dtype=dtype)
        return leaf, lambda t: t
    if layout == "heads_moved":
        # As in the attention layers: (batch * points, heads * channels) split into heads.
        leaf = torch.randn(batch * num_points, heads * head_dim, generator=generator, device="cuda", dtype=dtype)
        return leaf, lambda t: t.view(batch, num_points, heads, head_dim).permute(0, 2, 1, 3)
    if layout == "channels_strided":
        # The channels are not next to each other; the kernel makes them so first.
        leaf = torch.randn(batch, heads, num_points, 2 * head_dim, generator=generator, device="cuda", dtype=dtype)
        return leaf, lambda t: t[..., ::2]
    raise ValueError(layout)


def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return apply_rotary(x, cos, sin).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [6, 16, 64, 96, 128, 129])
@pytest.mark.parametrize("num_points", [1, 37, 1000])
@pytest.mark.parametrize("layout", ["contiguous", "heads_moved", "channels_strided"])
def test_triton_matches_pytorch_forward_and_backward(dtype, head_dim, num_points, layout):
    cos, sin = tables(num_points, head_dim)
    leaf, view = make_input(layout, 2, 3, num_points, head_dim, dtype)
    grad_out = torch.randn(
        2, 3, num_points, head_dim, device="cuda", dtype=dtype, generator=torch.Generator("cuda").manual_seed(1)
    )

    leaf_t = leaf.clone().requires_grad_()
    out_t = spherical_rotary(view(leaf_t), cos, sin)
    out_t.backward(grad_out)
    leaf_r = leaf.clone().requires_grad_()
    out_r = reference(view(leaf_r), cos, sin)
    out_r.backward(grad_out)

    assert out_t.dtype == dtype and out_t.shape == out_r.shape
    torch.testing.assert_close(out_t, out_r, **TOLERANCE[dtype])
    torch.testing.assert_close(leaf_t.grad, leaf_r.grad, **TOLERANCE[dtype])


def test_triton_passes_the_unturned_channels_through_exactly():
    cos, sin = tables(100, 128)  # 63 pairs turned; channels 63 and 127 are not
    x = torch.randn(2, 4, 100, 129, device="cuda", dtype=torch.bfloat16)  # and channel 128 has no partner
    out = spherical_rotary(x, cos, sin)
    for channel in (63, 127, 128):
        assert torch.equal(out[..., channel], x[..., channel])


def test_triton_keeps_the_layout_of_its_input():
    leaf, view = make_input("heads_moved", 2, 8, 50, 64, torch.bfloat16)
    x = view(leaf)
    cos, sin = tables(50, 64)
    out = spherical_rotary(x, cos, sin)
    assert out.stride() == x.stride()


def test_triton_accepts_lower_dimensional_inputs():
    cos, sin = tables(20, 32)
    x = torch.randn(20, 32, device="cuda")
    torch.testing.assert_close(spherical_rotary(x, cos, sin), reference(x, cos, sin), **TOLERANCE[torch.float32])
    x3 = torch.randn(4, 20, 32, device="cuda")
    torch.testing.assert_close(spherical_rotary(x3, cos, sin), reference(x3, cos, sin), **TOLERANCE[torch.float32])


def test_triton_gradient_passes_gradcheck():
    cos, sin = tables(7, 12, dtype=torch.float64)
    x = torch.randn(1, 2, 7, 12, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: spherical_rotary(t, cos, sin), (x,), eps=1e-6, atol=1e-8)


def test_triton_rejects_tables_that_do_not_fit():
    cos, sin = tables(10, 32)
    with pytest.raises(ValueError, match="do not fit"):
        spherical_rotary(torch.randn(1, 1, 11, 32, device="cuda"), cos, sin)
    with pytest.raises(ValueError, match="do not fit"):
        spherical_rotary(torch.randn(1, 1, 10, 16, device="cuda"), cos, sin)


def test_module_backends_agree_between_two_grids():
    data, hidden = ReducedGrid.octahedral(96), ReducedGrid.octahedral(48)
    modules = {
        backend: SphericalRotaryEmbedding(
            data.coords.float(), hidden.coords.float(), 128, 100.0, backend=backend
        ).cuda()
        for backend in ("triton", "torch")
    }
    generator = torch.Generator("cuda").manual_seed(0)
    q = torch.randn(2, 8, data.num_points, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(2, 8, hidden.num_points, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    results = {}
    for backend, module in modules.items():
        qq, kk = q.clone().requires_grad_(), k.clone().requires_grad_()
        tq, tk = module(qq, kk)
        (tq.float().square().sum() + tk.float().sum()).backward()
        results[backend] = (tq, tk, qq.grad, kk.grad)
    for got, expected in zip(results["triton"], results["torch"]):
        torch.testing.assert_close(got, expected, **TOLERANCE[torch.bfloat16])


def _layers(settings: dict, make_rotary):
    torch.manual_seed(0)
    layers = {}
    for backend in ("triton", "torch"):
        layers[backend] = settings["cls"](**settings["kwargs"], rotary=make_rotary(backend)).cuda()
    layers["torch"].load_state_dict(layers["triton"].state_dict())
    return layers


def _run(layer, inputs, shard, batch_size):
    layer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(inputs, shard, batch_size)
    out.float().square().mean().backward()
    return out.float(), {name: p.grad.float() for name, p in layer.named_parameters()}


def _assert_layers_agree(layers, inputs, shard, batch_size):
    out_t, grads_t = _run(layers["triton"], inputs, shard, batch_size)
    out_r, grads_r = _run(layers["torch"], inputs, shard, batch_size)
    assert (out_t - out_r).norm() / out_r.norm() < 1e-2
    for name in grads_r:
        assert (grads_t[name] - grads_r[name]).norm() / grads_r[name].norm() < 2e-2, name


def test_neighbourhood_cross_attention_with_either_backend():
    data, hidden = ReducedGrid.octahedral(16), ReducedGrid.octahedral(8)
    neighbourhood = GridNeighbourhood.from_config(
        {"grid": "octahedral", "kernel_size": [3, 5], "backend": "triton"}, data.coords.float(), hidden.coords.float()
    )
    settings = {
        "cls": MultiHeadCrossAttention,
        "kwargs": dict(
            num_heads=4,
            embed_dim=128,
            layer_kernels=load_layer_kernels(),
            attention_implementation="neighbourhood",
            neighbourhood=neighbourhood,
            qk_norm=True,
        ),
    }
    layers = _layers(
        settings,
        lambda backend: SphericalRotaryEmbedding(
            hidden.coords.float(), data.coords.float(), 32, 100.0, backend=backend
        ),
    )
    inputs = (
        torch.randn(2 * data.num_points, 128, device="cuda"),
        torch.randn(2 * hidden.num_points, 128, device="cuda"),
    )
    shard = BipartiteGraphShardInfo(src_nodes=[2 * data.num_points], dst_nodes=[2 * hidden.num_points])
    _assert_layers_agree(layers, inputs, shard, 2)


def test_sliding_window_flash_attention_with_either_backend():
    pytest.importorskip("flash_attn")
    grid = ReducedGrid.octahedral(16)
    settings = {
        "cls": MultiHeadSelfAttention,
        "kwargs": dict(
            num_heads=4,
            embed_dim=128,
            layer_kernels=load_layer_kernels(),
            attention_implementation="flash_attention",
            window_size=64,
            softcap=0.0,
        ),
    }
    layers = _layers(
        settings, lambda backend: SphericalRotaryEmbedding(grid.coords.float(), None, 32, 100.0, backend=backend)
    )
    inputs = torch.randn(2 * grid.num_points, 128, device="cuda")
    _assert_layers_agree(layers, inputs, GraphShardInfo(nodes=[2 * grid.num_points]), 2)
