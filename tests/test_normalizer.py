import numpy as np
import pytest

from anemoi.graphs.normalizer import NormalizerMixin


@pytest.mark.parametrize("norm", ["l1", "l2", "unit-max", "unit-range", "unit-std"])
def test_normalizer(norm: str):
    """Test NormalizerMixin normalize method."""

    class Normalizer(NormalizerMixin):
        def __init__(self, norm):
            self.norm = norm

        def __call__(self, data):
            return self.normalize(data)

    normalizer = Normalizer(norm=norm)
    data = np.random.rand(10, 5)
    normalized_data = normalizer(data)
    assert isinstance(normalized_data, np.ndarray)
    assert normalized_data.shape == data.shape


@pytest.mark.parametrize("norm", ["l3", "invalid"])
def test_normalizer_wrong_norm(norm: str):
    """Test NormalizerMixin normalize method."""

    class Normalizer(NormalizerMixin):
        def __init__(self, norm: str):
            self.norm = norm

        def __call__(self, data):
            return self.normalize(data)

    with pytest.raises(ValueError):
        normalizer = Normalizer(norm=norm)
        data = np.random.rand(10, 5)
        normalizer(data)


def test_normalizer_wrong_inheritance():
    """Test NormalizerMixin normalize method."""

    class Normalizer(NormalizerMixin):
        def __init__(self, attr):
            self.attr = attr

        def __call__(self, data):
            return self.normalize(data)

    with pytest.raises(AttributeError):
        normalizer = Normalizer(attr="attr_name")
        data = np.random.rand(10, 5)
        normalizer(data)
