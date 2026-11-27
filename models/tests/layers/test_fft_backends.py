# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Backend configuration, lazy allocation, precision and native CUDA graph replay."""

import pytest
import torch

from anemoi.models.layers import ring_fft
from anemoi.models.layers.padded_fft import PaddedFFT
from anemoi.models.layers.ring_fft import GraphedRingFFT
from anemoi.models.layers.ring_fft import NativeRingFFT
from anemoi.models.layers.ring_fft import shared_ring_fft
from anemoi.models.layers.spectral_helpers import InverseSphericalHarmonicTransform
from anemoi.models.layers.spectral_helpers import SphericalHarmonicTransform
from anemoi.models.layers.spectral_transforms import InverseOctahedralSHT
from anemoi.models.layers.spectral_transforms import InverseReducedSHT
from anemoi.models.layers.spectral_transforms import OctahedralSHT
from anemoi.models.layers.spectral_transforms import ReducedSHT

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
]


@pytest.fixture(autouse=True)
def clear_fft_cache():
    shared_ring_fft.cache_clear()
    yield
    shared_ring_fft.cache_clear()


@pytest.mark.parametrize("backend", ["ring", "padded"])
def test_only_selected_backend_is_constructed(backend, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("The unselected backend must not be constructed")

    monkeypatch.setattr(ring_fft, "PaddedFFT" if backend == "ring" else "NativeRingFFT", unexpected)
    lengths = (5, 8, 8, 5)
    fft = shared_ring_fft(lengths, backend, False)
    assert fft is shared_ring_fft(lengths, backend, False)
    x = torch.randn(2, sum(lengths), dtype=torch.float64)
    torch.testing.assert_close(fft.irfft(fft.rfft(x)), x, rtol=2e-13, atol=2e-13)


def test_default_is_native_and_backend_caches_are_distinct():
    lengths = (5, 8, 8, 5)
    assert isinstance(shared_ring_fft(lengths), NativeRingFFT)
    assert isinstance(shared_ring_fft(lengths, "padded", False), PaddedFFT)


@pytest.mark.parametrize("backend", ["ring", "padded"])
def test_every_call_form_gives_the_same_instance(backend):
    lengths = (5, 8, 8, 5)
    fft = shared_ring_fft(lengths, backend)
    assert shared_ring_fft(list(lengths), backend) is fft
    assert shared_ring_fft(lengths, backend, False) is fft
    assert shared_ring_fft(lengths, backend=backend, graphed=False) is fft
    assert shared_ring_fft.cache_info().currsize == 1
    if backend == "ring":
        assert shared_ring_fft(lengths) is fft


def test_graphed_ffts_take_a_list_of_lengths():
    fft = GraphedRingFFT([5, 8, 8, 5])
    assert fft.lengths == (5, 8, 8, 5)
    assert [band.lengths for band in fft.ffts] == [(5,), (8,), (8, 5)]


@pytest.mark.parametrize("backend,graphed", [("unknown", False), ("padded", True)])
def test_invalid_backend_options_fail_before_construction(backend, graphed, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid configuration must fail before constructing either backend")

    monkeypatch.setattr(ring_fft, "NativeRingFFT", unexpected)
    monkeypatch.setattr(ring_fft, "PaddedFFT", unexpected)
    with pytest.raises(ValueError, match="fft_backend"):
        shared_ring_fft((5, 8, 8, 5), backend, graphed)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("backend,graphed", [("ring", False), ("ring", True), ("padded", False)])
def test_sht_wrappers_forward_backend_options(inverse, backend, graphed):
    transform = InverseOctahedralSHT if inverse else OctahedralSHT
    flag = "use_graphed_irfft" if inverse else "use_graphed_rfft"
    wrapper = transform(nlat=8, truncation=2, fft_backend=backend, **{flag: graphed})
    inner = wrapper._isht if inverse else wrapper._sht
    assert inner.fft_backend == backend
    assert getattr(inner, flag) == graphed
    reduced = InverseReducedSHT if inverse else ReducedSHT
    wrapper = reduced(grid="n320", truncation=2, fft_backend=backend, **{flag: graphed})
    inner = wrapper._isht if inverse else wrapper._sht
    assert inner.fft_backend == backend
    assert getattr(inner, flag) == graphed
    assert shared_ring_fft.cache_info().currsize == 0


@pytest.mark.parametrize("inverse", [False, True])
def test_padded_graph_combination_is_rejected_by_transforms(inverse):
    transform = InverseSphericalHarmonicTransform if inverse else SphericalHarmonicTransform
    flag = "use_graphed_irfft" if inverse else "use_graphed_rfft"
    with pytest.raises(ValueError, match="CUDA graphs require"):
        transform([5, 8, 8, 5], 2, fft_backend="padded", **{flag: True})


@pytest.mark.parametrize("inverse", [False, True])
def test_graphs_require_cuda(inverse):
    fft = shared_ring_fft((5, 8, 8, 5), "ring", True)
    assert isinstance(fft, GraphedRingFFT)
    with pytest.raises(RuntimeError, match="CUDA input"):
        fft.irfft(torch.zeros(4, 3, dtype=torch.complex64)) if inverse else fft.rfft(torch.zeros(26), 3)
    assert not fft._captures


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_graph_replay_matches_values_and_adjoints(inverse, dtype):
    torch.manual_seed(42)
    lengths = (5, 8, 8, 5)
    native = NativeRingFFT(lengths)
    graphed = shared_ring_fft(lengths, "ring", True)
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    tolerance = dict(rtol=3e-6, atol=3e-6) if dtype == torch.float32 else dict(rtol=2e-13, atol=2e-13)
    for fields, modes in ((2, 3), (1, 7), (2, native.modes)):
        shape = (fields, len(lengths), modes) if inverse else (fields, sum(lengths))
        transform = (lambda fft, x: fft.irfft(x)) if inverse else (lambda fft, x: fft.rfft(x, modes))
        for _ in range(2):
            x = torch.randn(shape, device="cuda", dtype=cdtype if inverse else dtype, requires_grad=True)
            actual, expected = transform(graphed, x), transform(native, x)
            # A second replay must not overwrite an earlier result or its adjoint's state.
            first = actual.detach().clone()
            transform(graphed, x * 2)
            torch.testing.assert_close(actual, first, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, **tolerance)
            cotangent = torch.randn_like(actual)
            got = torch.autograd.grad(actual, x, cotangent)[0]
            want = torch.autograd.grad(expected, x, cotangent)[0]
            torch.testing.assert_close(got, want, **tolerance)
        with torch.no_grad():
            torch.testing.assert_close(transform(graphed, x), transform(native, x), **tolerance)
        assert len(graphed._captures) > 0


@pytest.mark.parametrize("fft_class", [NativeRingFFT, PaddedFFT, GraphedRingFFT])
@pytest.mark.parametrize("modes", [0, -1, 1.5])
def test_requested_order_count_must_be_positive_integer(fft_class, modes):
    with pytest.raises(ValueError, match="positive integer"):
        fft_class((5, 8, 8, 5)).rfft(torch.zeros(26), modes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
@pytest.mark.parametrize("inverse", [False, True])
def test_graph_captured_in_inference_mode_replays_in_training(inverse):
    """A capture made during validation takes new inputs and gives gradients in the training step that follows."""
    torch.manual_seed(3)
    lengths = (5, 8, 8, 5)
    native = NativeRingFFT(lengths)
    graphed = shared_ring_fft(lengths, "ring", True)
    shape = (2, len(lengths), 3) if inverse else (2, sum(lengths))
    dtype = torch.complex64 if inverse else torch.float32
    transform = (lambda fft, x: fft.irfft(x)) if inverse else (lambda fft, x: fft.rfft(x, 3))
    with torch.inference_mode():
        x = torch.randn(shape, device="cuda", dtype=dtype)
        torch.testing.assert_close(transform(graphed, x), transform(native, x), rtol=3e-6, atol=3e-6)
    # The target in training goes through the capture made during validation.
    target = torch.randn(shape, device="cuda", dtype=dtype)
    torch.testing.assert_close(transform(graphed, target), transform(native, target), rtol=3e-6, atol=3e-6)
    pred = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    actual, expected = transform(graphed, pred), transform(native, pred)
    cotangent = torch.randn_like(actual)
    got = torch.autograd.grad(actual, pred, cotangent)[0]
    want = torch.autograd.grad(expected, pred, cotangent)[0]
    torch.testing.assert_close(got, want, rtol=3e-6, atol=3e-6)


TRUNCATION = 11
LONS = OctahedralSHT(nlat=24, truncation=TRUNCATION).lons_per_lat
COMPLEX = {torch.bfloat16: torch.complex64, torch.float16: torch.complex64, torch.float32: torch.complex64}
COMPLEX[torch.float64] = torch.complex128


def _relative_error(actual: torch.Tensor, exact: torch.Tensor) -> float:
    return ((actual.to(exact.dtype) - exact).abs().max() / exact.abs().max()).item()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["ring", "padded"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_transforms_run_in_at_least_float32_under_autocast(device, backend, dtype):
    """Half-precision fields are promoted to float32, other fields keep their precision, and autocast has no effect."""
    torch.manual_seed(0)
    sht = SphericalHarmonicTransform(LONS, TRUNCATION, fft_backend=backend).to(device)
    isht = InverseSphericalHarmonicTransform(LONS, TRUNCATION, fft_backend=backend).to(device)
    x = torch.randn(3, sum(LONS), dtype=torch.float64, device=device).to(dtype)
    exact = sht(x.double())
    with torch.autocast(device, dtype=torch.bfloat16):
        coeffs = sht(x)
        fields = isht(coeffs)
    assert coeffs.dtype == COMPLEX[dtype]
    assert fields.dtype == COMPLEX[dtype].to_real()
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    assert _relative_error(coeffs, exact) < tolerance
    assert _relative_error(fields, isht(exact)) < tolerance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["ring", "padded"])
def test_inverse_raises_half_precision_coefficients(device, backend):
    sht = SphericalHarmonicTransform(LONS, TRUNCATION, fft_backend=backend).to(device)
    isht = InverseSphericalHarmonicTransform(LONS, TRUNCATION, fft_backend=backend).to(device)
    coeffs = sht(torch.randn(2, sum(LONS), dtype=torch.float64, device=device)).to(torch.complex32)
    fields = isht(coeffs)
    assert fields.dtype == torch.float32
    assert _relative_error(fields, isht(coeffs.to(torch.complex128))) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["ring", "padded"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_under_autocast_match_float64(device, backend, dtype):
    torch.manual_seed(1)
    sht = SphericalHarmonicTransform(LONS, TRUNCATION, fft_backend=backend).to(device)
    isht = InverseSphericalHarmonicTransform(LONS, TRUNCATION, fft_backend=backend).to(device)
    x = torch.randn(3, sum(LONS), dtype=torch.float64, device=device)
    weight = torch.randn(sum(LONS), dtype=torch.float64, device=device)

    def gradient(x: torch.Tensor, autocast: bool) -> torch.Tensor:
        x = x.clone().requires_grad_()
        # As in training, the forward pass runs under autocast and the backward pass after it.
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            value = (isht(sht(x).square()) * weight.to(x.dtype)).sum()
        value.backward()
        return x.grad

    exact = gradient(x, autocast=False)
    actual = gradient(x.to(dtype), autocast=True)
    assert actual.dtype == dtype
    assert _relative_error(actual, exact) < (1e-12 if dtype == torch.float64 else 1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fft_class", [NativeRingFFT, PaddedFFT], ids=["ring", "padded"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ring_ffts_raise_half_precision_to_float32(device, fft_class, dtype):
    fft = fft_class((5, 8, 8, 5))
    x = torch.randn(2, fft.points, device=device).to(dtype).requires_grad_()
    coefficients = fft.rfft(x)
    assert coefficients.dtype == torch.complex64
    torch.testing.assert_close(coefficients, fft.rfft(x.detach().float()), rtol=0, atol=0)
    assert torch.autograd.grad(coefficients.abs().sum(), x)[0].dtype == dtype

    half = coefficients.detach().to(torch.complex32)
    fields = fft.irfft(half)
    assert fields.dtype == torch.float32
    torch.testing.assert_close(fields, fft.irfft(half.to(torch.complex64)), rtol=0, atol=0)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize(
    ("device", "backend", "graphed"),
    [
        ("cpu", "ring", False),
        ("cpu", "padded", False),
        pytest.param("cuda", "ring", False, marks=CUDA),
        pytest.param("cuda", "padded", False, marks=CUDA),
        pytest.param("cuda", "ring", True, marks=CUDA),
    ],
)
def test_ring_ffts_of_no_fields_are_empty_and_pass_gradients(device, backend, graphed):
    """Without fields, both directions give empty results, and the backward pass still reaches the input."""
    fft = shared_ring_fft(tuple(LONS), backend, graphed)
    x = torch.zeros(2, 0, fft.points, device=device, requires_grad=True)
    a = torch.zeros(2, 0, fft.nlat, TRUNCATION + 1, dtype=torch.complex64, device=device, requires_grad=True)
    coefficients = fft.rfft(x, TRUNCATION + 1)
    fields = fft.irfft(a)
    assert coefficients.shape == (2, 0, fft.nlat, TRUNCATION + 1)
    assert coefficients.dtype == torch.complex64
    assert fields.shape == (2, 0, fft.points)
    assert fields.dtype == torch.float32
    (coefficients.abs().sum() + fields.sum()).backward()
    assert x.grad.shape == x.shape
    assert a.grad.shape == a.shape


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lons", [LONS, [2 * len(LONS)] * len(LONS)], ids=["octahedral", "regular"])
def test_transforms_of_no_fields_are_empty_and_pass_gradients(device, lons):
    """The transforms of a batch without fields are empty, and the backward pass still reaches the input."""
    sht = SphericalHarmonicTransform(lons, TRUNCATION).to(device)
    isht = InverseSphericalHarmonicTransform(lons, TRUNCATION).to(device)
    x = torch.zeros(2, 0, sum(lons), device=device, requires_grad=True)
    coefficients = sht(x)
    fields = isht(coefficients)
    assert coefficients.shape == (2, 0, TRUNCATION + 1, TRUNCATION + 1)
    assert fields.shape == x.shape
    fields.sum().backward()
    assert x.grad.shape == x.shape
