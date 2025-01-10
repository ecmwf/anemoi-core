import pytest
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf

from anemoi.models.layers.utils import load_layer_kernels


@pytest.fixture
def default_layer_kernels():
    # Default layer kernels
    kernels_config = OmegaConf.create(
        {
            "LayerNorm": {
                "_target_": "torch.nn.LayerNorm",
                "_partial_": True,
            },
        }
    )
    return instantiate(load_layer_kernels(kernels_config))


@pytest.fixture
def custom_layer_kernels():
    # Custom layer kernels
    kernels_config = OmegaConf.create(
        {
            "LayerNorm": {
                "_target_": "torch.nn.LayerNorm",
                "_partial_": True,
                "eps": 1e-3,
                "elementwise_affine": False,
            },
            "Linear": {"_target_": "torch.nn.Linear", "_partial_": True, "bias": False},
        }
    )
    return instantiate(load_layer_kernels(kernels_config))


def test_kernels_init(default_layer_kernels):
    """Test that the layer kernels are instantiated."""
    channels = 10
    linear_layer = default_layer_kernels["Linear"](in_features=channels, out_features=channels)
    layer_norm = default_layer_kernels["LayerNorm"](normalized_shape=channels)
    assert isinstance(linear_layer, torch.nn.Linear)
    assert isinstance(layer_norm, torch.nn.LayerNorm)
    assert linear_layer.bias.shape == torch.Size([channels])
    assert layer_norm.bias.shape == torch.Size([channels])


def test_custom_kernels(custom_layer_kernels):
    """Test that the custom layer kernels are instantiated."""
    linear_layer = custom_layer_kernels["Linear"](in_features=10, out_features=10)
    layer_norm = custom_layer_kernels["LayerNorm"](normalized_shape=10)

    assert linear_layer.bias is None
    assert layer_norm.bias is None


def test_unavailable_kernel():
    """Config with an unavailable kernel that should raise an error."""
    kernels_config = OmegaConf.create(
        {
            "LayerNorm": {"_target_": "nonexistent_package.LayerNorm", "_partial_": True},
            "Linear": {"_target_": "torch.nn.Linear", "_partial_": True},
        }
    )
    # Catch InstantiationException
    with pytest.raises(InstantiationException):
        load_layer_kernels(kernels_config)
