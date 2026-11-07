# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import builtins
from types import SimpleNamespace

import pytest
import torch

from anemoi.models.layers.wavelet_transforms import DTCWT2D
from anemoi.models.layers.wavelet_transforms import WaveletPyramid


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(8, 10), (9, 11)])
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_reference_coefficients_and_gradients(shape, levels):
    wavelets = pytest.importorskip("pytorch_wavelets")
    x = torch.randn(2, 2, *shape, generator=torch.Generator().manual_seed(7), requires_grad=True)
    native = wavelets.DTCWTForward(J=levels, biort="near_sym_b", qshift="qshift_b")
    expected_low, expected_high = native(x)
    actual = DTCWT2D(J=levels)(x)
    assert isinstance(actual, WaveletPyramid)
    assert len(actual.highpasses) == levels
    torch.testing.assert_close(actual.lowpass, expected_low)
    for level, expected in zip(actual.highpasses, expected_high, strict=True):
        torch.testing.assert_close(level, expected)
        assert level.shape[:3] == (2, 2, 6)
        assert level.shape[-1] == 2
        assert level.is_contiguous()
    actual_energy = actual.lowpass.square().sum() + sum(h.square().sum() for h in actual.highpasses)
    expected_energy = expected_low.square().sum() + sum(h.square().sum() for h in expected_high)
    torch.testing.assert_close(
        torch.autograd.grad(actual_energy, x)[0],
        torch.autograd.grad(expected_energy, x)[0],
    )


@pytest.mark.parametrize("shape", [(10, 12), (9, 11)])
def test_inverse_reconstructs_real_fields(shape):
    wavelets = pytest.importorskip("pytorch_wavelets")
    x = torch.randn(1, 1, *shape, generator=torch.Generator().manual_seed(9))
    pyramid = DTCWT2D(J=2)(x)
    inverse = wavelets.DTCWTInverse(biort="near_sym_b", qshift="qshift_b")
    reconstructed = inverse((pyramid.lowpass, list(pyramid.highpasses)))
    torch.testing.assert_close(reconstructed[..., : shape[0], : shape[1]], x, rtol=2e-5, atol=2e-6)


def test_reference_transform_passes_double_precision_gradcheck():
    pytest.importorskip("pytorch_wavelets")
    transform = DTCWT2D(J=2)
    x = torch.randn(1, 1, 4, 6, dtype=torch.float64, requires_grad=True)

    def energy(data):
        pyramid = transform(data)
        return pyramid.lowpass.square().sum() + sum(h.square().sum() for h in pyramid.highpasses)

    assert torch.autograd.gradcheck(energy, (x,), eps=1e-6, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("levels", [0, -1, True, 1.5])
def test_invalid_level_count(levels):
    with pytest.raises(ValueError, match="positive integer"):
        DTCWT2D(J=levels)


def test_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        DTCWT2D(J=1, backend="invalid")


def test_missing_optional_dependency_is_actionable(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "pytorch_wavelets":
            raise ImportError("dependency not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ImportError, match="Install"):
        DTCWT2D(J=1)


@pytest.mark.parametrize("bad_input", [torch.zeros(1, 4, 4), torch.zeros(0, 1, 4, 4)])
def test_invalid_input_shape(bad_input):
    pytest.importorskip("pytorch_wavelets")
    with pytest.raises(ValueError, match="nonempty"):
        DTCWT2D(J=1)(bad_input)


def test_invalid_input_dtype():
    pytest.importorskip("pytorch_wavelets")
    with pytest.raises(TypeError, match="float32 or float64"):
        DTCWT2D(J=1)(torch.zeros(1, 1, 4, 4, dtype=torch.int64))


class _ObjectBackend(torch.nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.invalid = invalid

    def forward(self, data):
        orientations = 5 if self.invalid else 6
        high = data.new_zeros(*data.shape[:2], orientations, 2, 2, 2)
        return SimpleNamespace(yl=data, yh=[high])


def test_object_backend_contract_and_validation():
    pytest.importorskip("pytorch_wavelets")
    transform = DTCWT2D(J=1)
    transform.impl = _ObjectBackend()
    result = transform(torch.ones(1, 2, 4, 4))
    assert result.highpasses[0].shape == (1, 2, 6, 2, 2, 2)
    transform.impl = _ObjectBackend(invalid=True)
    with pytest.raises(ValueError, match="high-pass"):
        transform(torch.ones(1, 2, 4, 4))


def test_prepare_moves_reference_filters():
    pytest.importorskip("pytorch_wavelets")
    transform = DTCWT2D(J=2)
    transform.prepare(torch.device("cpu"), (7, 9), dtype=torch.float64)
    assert all(buffer.dtype == torch.float64 for buffer in transform.buffers())
    assert transform(torch.ones(1, 1, 7, 9, dtype=torch.float64)).lowpass.dtype == torch.float64
