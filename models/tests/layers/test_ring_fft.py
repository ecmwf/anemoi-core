# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare native and padded ring FFTs and their explicit adjoints with per-ring torch.fft calls and their autograd."""

import pytest
import torch
from torch.nn import functional as F

from anemoi.models.layers import padded_fft as padded_fft_module
from anemoi.models.layers.padded_fft import PaddedFFT
from anemoi.models.layers.ring_fft import GraphedRingFFT
from anemoi.models.layers.ring_fft import NativeRingFFT
from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform


@pytest.fixture(params=["cpu", "cuda"])
def fft_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


@pytest.fixture(params=[NativeRingFFT, PaddedFFT], ids=["ring", "padded"])
def fft_class(request):
    return request.param


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(17)


def reference_rfft(x, lengths):
    """Compute each ring's FFT separately with torch.fft and autograd."""
    modes = max(lengths) // 2 + 1
    return torch.stack(
        [
            F.pad(torch.fft.rfft(ring, norm="forward"), (0, modes - n // 2 - 1))
            for ring, n in zip(x.split(lengths, dim=-1), lengths)
        ],
        dim=-2,
    )


def reference_irfft(x, lengths):
    return torch.cat([torch.fft.irfft(x[..., ring, :], n=n, norm="forward") for ring, n in enumerate(lengths)], dim=-1)


def tolerances(dtype):
    # Small FFTs: float64 checks are close to machine precision.
    return dict(rtol=2e-6, atol=2e-6) if dtype == torch.float32 else dict(rtol=2e-13, atol=2e-13)


def compare_value_and_vjp(actual, expected, x, dtype):
    torch.testing.assert_close(actual, expected, **tolerances(dtype))
    # Arbitrary, noncontiguous cotangents exercise real AND imaginary gradients.
    grad = torch.randn(*actual.shape, 2, device=x.device, dtype=actual.dtype)[..., 0]
    actual_grad = torch.autograd.grad(actual, x, grad)[0]
    expected_grad = torch.autograd.grad(expected, x, grad)[0]
    torch.testing.assert_close(actual_grad, expected_grad, **tolerances(dtype))
    return actual_grad


@pytest.mark.parametrize("lengths", [[1, 2, 5, 6, 5, 2, 1], [9, 4, 7, 4, 9], [8, 8, 8]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_rfft_matches_torch(fft_class, fft_device, lengths, dtype, batch_shape):
    fft = fft_class(lengths)
    x = torch.randn(*batch_shape, 2 * sum(lengths), device=fft_device, dtype=dtype)[..., ::2].requires_grad_()
    actual = fft.rfft(x)
    compare_value_and_vjp(actual, reference_rfft(x, lengths), x, dtype)
    for ring, n in enumerate(lengths):
        assert torch.count_nonzero(actual[..., ring, n // 2 + 1 :]) == 0


@pytest.mark.parametrize("lengths", [[1, 2, 5, 6, 5, 2, 1], [9, 4, 7, 4, 9], [8, 8, 8]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("width", ["dc", "short", "full", "extra"])
def test_irfft_matches_torch(fft_class, fft_device, lengths, dtype, batch_shape, width):
    fft = fft_class(lengths)
    modes = {"dc": 1, "short": fft.modes - 1, "full": fft.modes, "extra": fft.modes + 3}[width]
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    x = torch.randn(*batch_shape, len(lengths), 2 * modes, device=fft_device, dtype=cdtype)[..., ::2].requires_grad_()
    grad = compare_value_and_vjp(fft.irfft(x), reference_irfft(x, lengths), x, dtype)
    assert torch.count_nonzero(grad[..., 0].imag) == 0
    for ring, n in enumerate(lengths):
        assert torch.count_nonzero(grad[..., ring, n // 2 + 1 :]) == 0
        if n % 2 == 0 and modes > n // 2:
            assert torch.count_nonzero(grad[..., ring, n // 2].imag) == 0


@pytest.mark.parametrize("component", [1.0, 1.0j])
def test_rfft_backward_each_mode(fft_class, fft_device, component):
    """Check each Jacobian row, including padded modes and imaginary endpoints."""
    lengths = [1, 2, 5, 6, 5]
    fft = fft_class(lengths)
    x = torch.randn(sum(lengths), dtype=torch.float64, device=fft_device, requires_grad=True)
    actual, expected = fft.rfft(x), reference_rfft(x, lengths)
    for ring in range(len(lengths)):
        for mode in range(fft.modes):
            grad = torch.zeros_like(actual)
            grad[ring, mode] = component
            got = torch.autograd.grad(actual, x, grad, retain_graph=True)[0]
            want = torch.autograd.grad(expected, x, grad, retain_graph=True)[0]
            torch.testing.assert_close(got, want, rtol=2e-13, atol=2e-13)
            if mode > lengths[ring] // 2 or (
                component == 1.0j and (mode == 0 or (lengths[ring] % 2 == 0 and mode == lengths[ring] // 2))
            ):
                assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("direction", ["rfft", "irfft"])
def test_gradcheck(fft_class, fft_device, direction):
    fft = fft_class([3, 4, 3])
    shape = (fft.points,) if direction == "rfft" else (fft.nlat, fft.modes + 1)
    dtype = torch.float64 if direction == "rfft" else torch.complex128
    x = torch.randn(shape, device=fft_device, dtype=dtype, requires_grad=True)
    fn = getattr(fft, direction)
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_roundtrip_without_saved_activations(fft_class, fft_device):
    fft = fft_class([5, 6, 5])
    x = torch.randn(2, fft.points, device=fft_device, dtype=torch.float64, requires_grad=True)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = fft.irfft(fft.rfft(x))
    torch.testing.assert_close(out, x, rtol=2e-13, atol=2e-13)
    grad = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(out, x, grad)[0], grad, rtol=2e-13, atol=2e-13)
    assert not saved


def test_sht_keeps_no_grid_state_and_follows_the_input(fft_device):
    """The ring tables are built for the device and precision of the input, and nothing is checkpointed."""
    lengths = [5, 8, 8, 5]
    direct = SphericalHarmonicTransform(lengths, truncation=2)
    inverse = InverseSphericalHarmonicTransform(lengths, truncation=2)
    for module in (direct, inverse):
        module.float().to(fft_device).double()
        assert not module.state_dict()
    x = torch.randn(sum(lengths), device=fft_device, dtype=torch.float64, requires_grad=True)
    compare_value_and_vjp(direct.rfft_rings(x), reference_rfft(x, lengths)[..., :3], x, torch.float64)
    z = torch.randn(len(lengths), 5, device=fft_device, dtype=torch.complex128, requires_grad=True)
    compare_value_and_vjp(inverse.irfft_rings(z), reference_irfft(z, lengths), z, torch.float64)


@pytest.mark.parametrize("modes", [1, 3, 7])
def test_rfft_gives_the_orders_asked_for(fft_class, fft_device, modes):
    lengths = [1, 2, 5, 6, 5, 2, 1]
    x = torch.randn(2, sum(lengths), device=fft_device, dtype=torch.float64, requires_grad=True)
    expected = F.pad(reference_rfft(x, lengths), (0, max(0, modes - 4)))[..., :modes]
    compare_value_and_vjp(fft_class(lengths).rfft(x, modes), expected, x, torch.float64)


def test_fields_in_several_groups(fft_device, monkeypatch):
    """With a small buffer limit the fields go through in groups, with the same result."""
    monkeypatch.setattr(padded_fft_module, "_MAX_BUFFER_VALUES", 64)
    lengths = [9, 4, 7, 4, 9]
    x = torch.randn(5, sum(lengths), device=fft_device, dtype=torch.float64, requires_grad=True)
    compare_value_and_vjp(PaddedFFT(lengths).rfft(x), reference_rfft(x, lengths), x, torch.float64)
    z = torch.randn(5, len(lengths), 5, device=fft_device, dtype=torch.complex128, requires_grad=True)
    compare_value_and_vjp(PaddedFFT(lengths).irfft(z), reference_irfft(z, lengths), z, torch.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the compiled packing kernels")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_packing_matches_torch(dtype, monkeypatch):
    """Fused packing handles strided inputs, changing truncations, partial chunks and both adjoints."""
    monkeypatch.setattr(padded_fft_module, "_MAX_BUFFER_VALUES", 256)
    lengths = [9, 4, 7, 4, 9]
    fft = PaddedFFT(lengths)
    fft.compile_for_training()
    for modes in (1, 5, 8):
        x = torch.randn(5, 2 * fft.points, device="cuda", dtype=dtype)[:, ::2]
        x[1] *= 1e-7
        x[3] = 0
        x.requires_grad_()
        expected = F.pad(reference_rfft(x, lengths), (0, max(0, modes - fft.modes)))[..., :modes]
        actual = fft.rfft(x, modes)
        compare_value_and_vjp(actual, expected, x, dtype)
        assert (actual[1] - expected[1]).norm() < tolerances(dtype)["rtol"] * expected[1].norm()
        assert torch.count_nonzero(actual[3]) == 0

        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        a = torch.randn(5, fft.nlat, 2 * modes, device="cuda", dtype=cdtype)[..., ::2]
        a[1] *= 1e-7
        a[3] = 0
        a.requires_grad_()
        actual = fft.irfft(a)
        expected = reference_irfft(a, lengths)
        compare_value_and_vjp(actual, expected, a, dtype)
        assert (actual[1] - expected[1]).norm() < tolerances(dtype)["rtol"] * expected[1].norm()
        assert torch.count_nonzero(actual[3]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the compiled packing kernels")
def test_compiled_packing_keeps_second_derivatives():
    fft = PaddedFFT([3, 4, 3])
    fft.compile_for_training()
    x = torch.randn(fft.points, device="cuda", dtype=torch.float64, requires_grad=True)
    a = torch.randn(fft.nlat, fft.modes, device="cuda", dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradgradcheck(fft.rfft, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(fft.irfft, (a,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_chirp_data_is_built_once_and_reused(fft_device):
    lengths = [9, 4, 7, 4, 9]
    fft = PaddedFFT(lengths)
    x = torch.randn(3, sum(lengths), device=fft_device, dtype=torch.float64, requires_grad=True)

    def run():
        spectrum = fft.rfft(x, 4)
        values = fft.irfft(spectrum)
        (grad,) = torch.autograd.grad(values.square().sum(), x)
        return spectrum.detach(), values.detach(), grad

    first = run()
    groups = [group for plan in fft._plans.values() for group in plan[0]]
    kept = [dict(group.cache) for group in groups]
    assert all(kept)
    second = run()
    for before, group in zip(kept, groups):
        assert group.cache.keys() == before.keys()
        assert all(group.cache[key] is value for key, value in before.items())
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_long_rings_keep_float32_accuracy(fft_class):
    """O640 rings of up to 2576 points keep float32 accuracy, which for padded FFTs needs exact chirp phases."""
    north = [20 + 4 * ring for ring in range(640)]
    lengths = north + north[::-1]
    fft = fft_class(lengths)
    x = torch.randn(2, sum(lengths), device="cuda")
    expected = reference_rfft(x.double(), lengths)[..., :640]
    torch.testing.assert_close(fft.rfft(x, 640).to(torch.complex128), expected, rtol=0, atol=1e-6)
    z = torch.randn(2, len(lengths), 640, device="cuda", dtype=torch.complex64)
    expected = reference_irfft(z.to(torch.complex128), lengths)
    torch.testing.assert_close(fft.irfft(z).double(), expected, rtol=0, atol=2e-6 * expected.abs().max().item())


def fields_of_different_size(fft_class, ratio, device):
    """Return a smooth large field and a small one whose size also changes by a factor 1000 from ring to ring."""
    north = [20 + 4 * ring for ring in range(32)]
    lengths = north + north[::-1]
    fft = fft_class(lengths)
    point_ring = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), torch.tensor(lengths, device=device)
    )
    latitude = point_ring.to(torch.float64) / (fft.nlat - 1)
    large = 20.0 + torch.cos(2 * torch.pi * latitude)
    small = torch.randn(fft.points, device=device, dtype=torch.float64) * ratio * 10.0 ** (-3.0 * latitude)
    return fft, lengths, torch.stack([large, small]), point_ring


def ring_errors(actual, expected):
    """Return the error of each ring relative to the size of that ring."""
    return (actual - expected).abs().pow(2).sum(-1).sqrt() / expected.abs().pow(2).sum(-1).sqrt()


@pytest.mark.parametrize("ratio", [1.0, 1e-2, 1e-4, 1e-6])
def test_small_field_keeps_its_accuracy_next_to_a_large_one(fft_class, fft_device, ratio):
    fft, lengths, x, _ = fields_of_different_size(fft_class, ratio, fft_device)
    expected = reference_rfft(x, lengths)
    errors = ring_errors(fft.rfft(x.float()).to(torch.complex128), expected)
    assert errors.max() < 2e-6


@pytest.mark.parametrize("ratio", [1.0, 1e-2, 1e-4, 1e-6])
def test_inverse_keeps_a_small_field_accurate_next_to_a_large_one(fft_class, fft_device, ratio):
    fft, lengths, x, point_ring = fields_of_different_size(fft_class, ratio, fft_device)
    coefficients = reference_rfft(x, lengths)
    expected = reference_irfft(coefficients, lengths)
    actual = fft.irfft(coefficients.to(torch.complex64)).double()
    squares = torch.zeros(2, fft.nlat, device=fft_device, dtype=torch.float64)
    errors = squares.index_add(1, point_ring, (actual - expected) ** 2).sqrt()
    sizes = squares.index_add(1, point_ring, expected**2).sqrt()
    assert (errors / sizes).max() < 2e-6


@pytest.mark.parametrize("ratio", [1e-2, 1e-4, 1e-6])
@pytest.mark.parametrize("direction", ["rfft", "irfft"])
def test_gradients_of_a_small_field_keep_their_accuracy(fft_class, fft_device, ratio, direction):
    """Both adjoints preserve the accuracy of small cotangents in a batch."""
    fft, lengths, x, _ = fields_of_different_size(fft_class, 1.0, fft_device)
    x = x.float() if direction == "rfft" else reference_rfft(x.float(), lengths)
    x.requires_grad_()
    out = getattr(fft, direction)(x)
    grad = torch.randn_like(out)
    grad[1] *= ratio
    actual = torch.autograd.grad(out, x, grad)[0][1]
    x64 = x.detach().to(torch.float64 if direction == "rfft" else torch.complex128).requires_grad_()
    reference = reference_rfft if direction == "rfft" else reference_irfft
    out64 = reference(x64, lengths)
    expected = torch.autograd.grad(out64, x64, grad.to(out64.dtype))[0][1]
    assert (actual.to(expected.dtype) - expected).norm() < 2e-6 * expected.norm()


@pytest.mark.parametrize("direction", ["rfft", "irfft"])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("large_or_bad", [1e12, float("nan"), float("inf")])
def test_fields_are_independent(fft_class, fft_device, direction, backward, large_or_bad):
    """Large or non-finite fields cannot contaminate other fields, including exact zeros, in either adjoint."""
    lengths = [20, 24, 28, 28, 24, 20]
    fft = fft_class(lengths)
    shape = (fft.points,) if direction == "rfft" else (fft.nlat, fft.modes)
    dtype = torch.float32 if direction == "rfft" else torch.complex64
    x = torch.randn(4, *shape, device=fft_device, dtype=dtype)
    reference = reference_rfft if direction == "rfft" else reference_irfft

    if backward:
        x.requires_grad_()
        out = getattr(fft, direction)(x)
        cotangent = torch.randn_like(out)
        cotangent[0] *= large_or_bad
        cotangent[2] *= 1e12
        cotangent[3] = 0
        actual = torch.autograd.grad(out, x, cotangent)[0]
        expected = torch.autograd.grad(reference(x, lengths), x, cotangent)[0]
    else:
        x[0] *= large_or_bad
        x[2] *= 1e12
        x[3] = 0
        actual = getattr(fft, direction)(x)
        expected = reference(x, lengths)

    torch.testing.assert_close(actual[1], expected[1], **tolerances(torch.float32))
    assert torch.count_nonzero(actual[3]) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_orders_a_ring_cannot_hold_stay_zero(fft_class, fft_device, bad):
    """Orders above half the length of a ring are zero in the forward transform and ignored by the inverse, also
    when the values involved are not finite.

    Rings of 9 and 10 points share one padded FFT, with orders 0 to 4 on the 9-point ring and 0 to 5 on the other.
    """
    fft = fft_class([9, 10])
    x = torch.randn(2, fft.points, device=fft_device, requires_grad=True)
    field = x.detach().clone()
    field[0, 0] = bad
    assert torch.count_nonzero(fft.rfft(field)[0, 0, 5:]) == 0

    a = torch.randn(2, fft.nlat, fft.modes, device=fft_device, dtype=torch.complex64)
    clean = a.clone()
    a[:, 0, 5] = bad
    torch.testing.assert_close(fft.irfft(a), fft.irfft(clean), rtol=0, atol=0)

    cotangent = torch.randn(2, fft.nlat, fft.modes, device=fft_device, dtype=torch.complex64)
    clean_cotangent = cotangent.clone()
    cotangent[:, 0, 5] = bad
    out = fft.rfft(x)
    actual = torch.autograd.grad(out, x, cotangent, retain_graph=True)[0]
    expected = torch.autograd.grad(out, x, clean_cotangent)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chirp_data_is_float64_data_rounded(fft_device):
    lengths = [20, 24, 28, 28, 24, 20]
    fft = PaddedFFT(lengths)
    device = torch.device(fft_device)
    assert torch.equal(fft._chirp(torch.float32, device), fft._chirp(torch.float64, device).to(torch.complex64))
    groups, _, _ = fft._plan(8, device)
    for group in groups:
        for real in (torch.float32, torch.float64):
            kernel = fft._chirp_kernel(group, group.modes, group.n - 1, 1.0, real)
            post = fft._order_chirp(group, group.taken, 1.0 / group.n.to(torch.float64), -1.0, real)
            if real == torch.float32:
                kernel32, post32 = kernel, post
        assert torch.equal(kernel32, kernel.to(torch.complex64))
        assert torch.equal(post32, post.to(torch.complex64))


@pytest.mark.parametrize("any_fft_class", [NativeRingFFT, PaddedFFT, GraphedRingFFT], ids=["ring", "padded", "graphed"])
@pytest.mark.parametrize("lengths", [[], [0, 4], [-1, 4], [2.5, 4]])
def test_invalid_lengths(any_fft_class, lengths):
    with pytest.raises(ValueError, match="positive integers"):
        any_fft_class(lengths)


@pytest.mark.parametrize(
    "direction,shape,dtype,error",
    [
        ("rfft", (), torch.float32, ValueError),
        ("rfft", (7,), torch.float32, ValueError),
        ("rfft", (8,), torch.complex64, TypeError),
        ("irfft", (3,), torch.complex64, ValueError),
        ("irfft", (3, 3), torch.complex64, ValueError),
        ("irfft", (2, 0), torch.complex64, ValueError),
        ("irfft", (2, 3), torch.float32, TypeError),
    ],
)
@pytest.mark.parametrize("any_fft_class", [NativeRingFFT, PaddedFFT, GraphedRingFFT], ids=["ring", "padded", "graphed"])
def test_invalid_input(any_fft_class, direction, shape, dtype, error):
    fft = any_fft_class([3, 5])
    with pytest.raises(error):
        getattr(fft, direction)(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", ["direct", "inverse", "roundtrip"])
@pytest.mark.parametrize("lengths", [[5, 7, 9, 11, 7, 5], [20, 24, 28, 28, 24, 20]])
@pytest.mark.parametrize("backend", ["ring", "padded"])
def test_sht_matches_torch_fft_pipeline(fft_device, dtype, direction, lengths, backend):
    """Compare the complete Legendre/FFT pipeline and gradients with native FFT autograd."""
    direct = SphericalHarmonicTransform(lengths, truncation=2, fft_backend=backend).to(fft_device)
    inverse = InverseSphericalHarmonicTransform(lengths, truncation=2, fft_backend=backend).to(fft_device)
    ref_direct = SphericalHarmonicTransform(lengths, truncation=2, fft_backend=backend).to(fft_device)
    ref_inverse = InverseSphericalHarmonicTransform(lengths, truncation=2, fft_backend=backend).to(fft_device)
    ref_direct.rfft_rings = lambda x: reference_rfft(x, lengths)
    ref_inverse.irfft_rings = lambda x: reference_irfft(x, lengths)
    if direction == "inverse":
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        x = torch.randn(2, 3, 3, 3, device=fft_device, dtype=cdtype, requires_grad=True)
        actual, expected = inverse(x), ref_inverse(x)
    else:
        x = torch.randn(2, 3, sum(lengths), device=fft_device, dtype=dtype, requires_grad=True)
        actual, expected = direct(x), ref_direct(x)
        if direction == "roundtrip":
            actual, expected = inverse(actual), ref_inverse(expected)
    compare_value_and_vjp(actual, expected, x, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", ["rfft", "irfft"])
def test_autocast_matches_torch(fft_class, fft_device, dtype, direction):
    lengths = [5, 8, 8, 5]
    fft = fft_class(lengths)
    if direction == "rfft":
        x = torch.randn(2, fft.points, device=fft_device, dtype=dtype, requires_grad=True)
        reference = reference_rfft
    else:
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        x = torch.randn(2, fft.nlat, 3, device=fft_device, dtype=cdtype, requires_grad=True)
        reference = reference_irfft
    with torch.autocast(fft_device):
        actual, expected = getattr(fft, direction)(x), reference(x, lengths)
    compare_value_and_vjp(actual, expected, x, dtype)
