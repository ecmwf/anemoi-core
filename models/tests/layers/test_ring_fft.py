# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare grouped FFTs and their explicit adjoints with PyTorch's own autograd."""

import pytest
import torch
from torch.nn import functional as F

from anemoi.models.layers.ring_fft import RingFFT
from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform


@pytest.fixture(params=["cpu", "cuda"])
def fft_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(17)


def reference_rfft(x, lengths):
    """Use independent per-ring torch.fft calls with native normalization/autograd."""
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
    # Small FFTs: float64 checks are close to machine precision; float32 allows
    # rounding from applying normalization after the batched FFT.
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
def test_rfft_matches_torch(fft_device, lengths, dtype, batch_shape):
    fft = RingFFT(lengths).to(fft_device)
    x = torch.randn(*batch_shape, 2 * sum(lengths), device=fft_device, dtype=dtype)[..., ::2].requires_grad_()
    actual = fft.rfft(x)
    compare_value_and_vjp(actual, reference_rfft(x, lengths), x, dtype)
    for ring, n in enumerate(lengths):
        assert torch.count_nonzero(actual[..., ring, n // 2 + 1 :]) == 0


@pytest.mark.parametrize("lengths", [[1, 2, 5, 6, 5, 2, 1], [9, 4, 7, 4, 9], [8, 8, 8]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
@pytest.mark.parametrize("width", ["dc", "short", "full", "extra"])
def test_irfft_matches_torch(fft_device, lengths, dtype, batch_shape, width):
    fft = RingFFT(lengths).to(fft_device)
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
def test_rfft_backward_each_mode(fft_device, component):
    """Check each Jacobian row, including padded modes and imaginary endpoints."""
    lengths = [1, 2, 5, 6, 5]
    fft = RingFFT(lengths).to(fft_device)
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
def test_gradcheck(fft_device, direction):
    fft = RingFFT([3, 4, 3]).to(fft_device)
    shape = (fft.points,) if direction == "rfft" else (fft.nlat, fft.modes + 1)
    dtype = torch.float64 if direction == "rfft" else torch.complex128
    x = torch.randn(shape, device=fft_device, dtype=dtype, requires_grad=True)
    fn = getattr(fft, direction)
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_roundtrip_without_saved_activations(fft_device):
    fft = RingFFT([5, 6, 5]).to(fft_device)
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


def test_metadata_follows_module_conversion(fft_device):
    """Moving an initialized SHT must move FFT metadata without rounding its scales."""
    lengths = [5, 8, 8, 5]
    direct = SphericalHarmonicTransform(lengths, truncation=2).cpu()
    inverse = InverseSphericalHarmonicTransform(lengths, truncation=2).cpu()
    for module in (direct, inverse):
        module.float().to(fft_device).double()
        assert not module.state_dict()  # Grid metadata is recomputed, not checkpointed.
    x = torch.randn(sum(lengths), device=fft_device, dtype=torch.float64, requires_grad=True)
    compare_value_and_vjp(direct.rfft_rings(x), reference_rfft(x, lengths), x, torch.float64)
    z = torch.randn(len(lengths), 5, device=fft_device, dtype=torch.complex128, requires_grad=True)
    compare_value_and_vjp(inverse.irfft_rings(z), reference_irfft(z, lengths), z, torch.float64)


@pytest.mark.parametrize("lengths", [[], [0, 4], [-1, 4], [2.5, 4]])
def test_invalid_lengths(lengths):
    with pytest.raises(ValueError, match="positive integers"):
        RingFFT(lengths)


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
def test_invalid_input(direction, shape, dtype, error):
    fft = RingFFT([3, 5])
    with pytest.raises(error):
        getattr(fft, direction)(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("direction", ["rfft", "irfft"])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("lengths", [[3, 8], [3, 5, 8, 8, 5, 3]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("autocast", [False, True])
def test_graphed_fft_matches_torch(direction, requires_grad, lengths, dtype, autocast):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if direction == "rfft":
        transform = SphericalHarmonicTransform(lengths, truncation=1, use_graphed_rfft=True).cuda()
        fn, reference = transform.rfft_rings, reference_rfft
        shape, input_dtype = (2, sum(lengths)), dtype
    else:
        transform = InverseSphericalHarmonicTransform(lengths, truncation=1, use_graphed_irfft=True).cuda()
        fn, reference = transform.irfft_rings, reference_irfft
        shape = (2, len(lengths), max(lengths) // 2 + 3)
        input_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    # Replays use fresh values, so stale outputs/gradients cannot pass.
    for _ in range(3):
        x = torch.randn(shape, device="cuda", dtype=input_dtype, requires_grad=requires_grad)
        with torch.autocast("cuda", enabled=autocast):
            actual, expected = fn(x), reference(x, lengths)
        if requires_grad:
            compare_value_and_vjp(actual, expected, x, dtype)
        else:
            torch.testing.assert_close(actual, expected, **tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", ["direct", "inverse", "roundtrip"])
@pytest.mark.parametrize("lengths", [[5, 7, 9, 11, 7, 5], [20, 24, 28, 28, 24, 20]])
def test_sht_matches_torch_fft_pipeline(fft_device, dtype, direction, lengths):
    """Compare the complete Legendre/FFT pipeline and gradients with native FFT autograd."""
    direct = SphericalHarmonicTransform(lengths, truncation=2).to(fft_device)
    inverse = InverseSphericalHarmonicTransform(lengths, truncation=2).to(fft_device)
    ref_direct = SphericalHarmonicTransform(lengths, truncation=2).to(fft_device)
    ref_inverse = InverseSphericalHarmonicTransform(lengths, truncation=2).to(fft_device)
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
def test_autocast_matches_torch(fft_device, dtype, direction):
    lengths = [5, 8, 8, 5]
    fft = RingFFT(lengths).to(fft_device)
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
