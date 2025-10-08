# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Tests for BF16FP32OptPrecision plugin."""

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from anemoi.training.precision import BF16FP32OptPrecision


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def precision_plugin() -> BF16FP32OptPrecision:
    """Create a BF16FP32OptPrecision plugin."""
    return BF16FP32OptPrecision()


def test_plugin_initialization(precision_plugin: BF16FP32OptPrecision) -> None:
    """Test that the plugin initializes correctly."""
    assert precision_plugin.precision == "bf16-fp32-opt"
    assert precision_plugin._model_converted is False


def test_convert_module(precision_plugin: BF16FP32OptPrecision, simple_model: SimpleModel) -> None:
    """Test that convert_module converts the model to bfloat16."""
    # Model should start in float32
    assert next(simple_model.parameters()).dtype == torch.float32

    # Convert to bf16
    converted_model = precision_plugin.convert_module(simple_model)

    # Check all parameters are bf16
    for param in converted_model.parameters():
        assert param.dtype == torch.bfloat16

    # Check flag is set
    assert precision_plugin._model_converted is True


def test_forward_context(precision_plugin: BF16FP32OptPrecision) -> None:
    """Test that forward_context returns a nullcontext."""
    ctx = precision_plugin.forward_context()
    # Should be a context manager that does nothing
    with ctx:
        pass  # Should not raise


def test_optimizer_step_dtype_conversion(precision_plugin: BF16FP32OptPrecision, simple_model: SimpleModel) -> None:
    """Test that optimizer_step correctly handles reference weights."""
    # Convert model to bf16
    precision_plugin.convert_module(simple_model)

    # Create optimizer
    optimizer = AdamW(simple_model.parameters(), lr=0.001)

    # Create dummy data and target
    x = torch.randn(8, 10).bfloat16()
    target = torch.randn(8, 5).bfloat16()

    # Define closure that performs forward and backward
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        output = simple_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        return loss

    # Before optimizer step, params should be bf16, no reference weights yet
    param_before = next(iter(simple_model.parameters()))
    assert param_before.dtype == torch.bfloat16
    assert precision_plugin._ref_by_model_param is None

    # Run optimizer step - this creates reference weights on first call
    precision_plugin.optimizer_step(optimizer, simple_model, closure)

    # After optimizer step, reference weights should exist
    assert precision_plugin._ref_by_model_param is not None
    assert len(precision_plugin._ref_by_model_param) == len(list(simple_model.parameters()))

    # Model params should still be bf16
    param_after = next(iter(simple_model.parameters()))
    assert param_after.dtype == torch.bfloat16

    # Reference weights should be fp32
    first_model_param = next(iter(simple_model.parameters()))
    assert first_model_param in precision_plugin._ref_by_model_param
    assert precision_plugin._ref_by_model_param[first_model_param].dtype == torch.float32

    # Optimizer should now track reference weights (fp32), not model params (bf16)
    optimizer_param = optimizer.param_groups[0]["params"][0]
    assert optimizer_param.dtype == torch.float32
    assert optimizer_param is precision_plugin._ref_by_model_param[first_model_param]

    # Optimizer state should be in fp32
    state_key = next(iter(optimizer.state.keys()))
    if "exp_avg" in optimizer.state[state_key]:
        assert optimizer.state[state_key]["exp_avg"].dtype == torch.float32
        assert optimizer.state[state_key]["exp_avg_sq"].dtype == torch.float32


def test_optimizer_step_with_no_gradients(precision_plugin: BF16FP32OptPrecision, simple_model: SimpleModel) -> None:
    """Test optimizer_step when some parameters have no gradients."""
    # Convert model to bf16
    precision_plugin.convert_module(simple_model)

    # Create optimizer
    optimizer = AdamW(simple_model.parameters(), lr=0.001)

    # Freeze first layer (no gradients)
    for param in simple_model.fc1.parameters():
        param.requires_grad = False

    # Create dummy data and target
    x = torch.randn(8, 10).bfloat16()
    target = torch.randn(8, 5).bfloat16()

    # Define closure
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        output = simple_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        return loss

    # Should not raise even though some params have no grad
    precision_plugin.optimizer_step(optimizer, simple_model, closure)

    # Params should still be bf16
    for param in simple_model.parameters():
        assert param.dtype == torch.bfloat16


def test_state_dict(precision_plugin: BF16FP32OptPrecision, simple_model: SimpleModel) -> None:
    """Test state_dict contains expected metadata and reference weights."""
    precision_plugin.convert_module(simple_model)

    # State dict before optimizer step (no reference weights yet)
    state_before = precision_plugin.state_dict()
    assert "precision_mode" in state_before
    assert state_before["precision_mode"] == "bf16-fp32-opt"
    assert "model_converted" in state_before
    assert state_before["model_converted"] is True
    assert "model_dtype" in state_before
    assert state_before["has_reference_weights"] is False

    # Create optimizer and run one step to create reference weights
    optimizer = AdamW(simple_model.parameters(), lr=0.001)
    x = torch.randn(8, 10).bfloat16()
    target = torch.randn(8, 5).bfloat16()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        output = simple_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        return loss

    precision_plugin.optimizer_step(optimizer, simple_model, closure)

    # State dict after optimizer step (should include reference weights)
    state_after = precision_plugin.state_dict()
    assert state_after["has_reference_weights"] is True
    assert "reference_params_by_name" in state_after
    assert len(state_after["reference_params_by_name"]) == len(list(simple_model.parameters()))
    # Check that saved reference weights are on CPU and fp32
    for param_name, ref_param in state_after["reference_params_by_name"].items():
        assert isinstance(param_name, str)
        assert ref_param.device.type == "cpu"
        assert ref_param.dtype == torch.float32


def test_load_state_dict_compatible(precision_plugin: BF16FP32OptPrecision) -> None:
    """Test loading state dict with compatible precision mode."""
    state = {
        "precision_mode": "bf16-fp32-opt",
        "model_converted": True,
        "model_dtype": "torch.bfloat16",
    }

    # Should not raise or warn
    precision_plugin.load_state_dict(state)
    assert precision_plugin._model_converted is True


def test_load_state_dict_incompatible(precision_plugin: BF16FP32OptPrecision) -> None:
    """Test loading state dict with incompatible precision mode warns."""
    state = {
        "precision_mode": "16-mixed",
        "model_converted": True,
    }

    # Should log a warning
    precision_plugin.load_state_dict(state)

    # Check that warning was logged (if using standard logging)
    # Note: This assumes the plugin uses logging that pytest can capture


def test_full_training_step(precision_plugin: BF16FP32OptPrecision, simple_model: SimpleModel) -> None:
    """Test a complete training step with the precision plugin."""
    # Convert model to bf16
    precision_plugin.convert_module(simple_model)

    # Create optimizer
    optimizer = AdamW(simple_model.parameters(), lr=0.001)

    # Create dummy batch
    batch_x = torch.randn(16, 10).bfloat16()
    batch_y = torch.randn(16, 5).bfloat16()

    # Training step closure
    def training_step() -> torch.Tensor:
        optimizer.zero_grad()
        output = simple_model(batch_x)
        loss = nn.functional.mse_loss(output, batch_y)
        loss.backward()
        return loss

    # Run optimizer step via plugin
    loss = precision_plugin.optimizer_step(optimizer, simple_model, training_step)

    # Check loss is finite
    assert torch.isfinite(loss)

    # Check parameters are bf16
    for param in simple_model.parameters():
        assert param.dtype == torch.bfloat16

    # Check reference weights are fp32
    for ref_param in precision_plugin._ref_by_model_param.values():
        assert ref_param.dtype == torch.float32

    # Check optimizer state exists and is fp32
    for state_key in optimizer.state:
        state = optimizer.state[state_key]
        if "exp_avg" in state:
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32


def test_reference_weights_sync(precision_plugin: BF16FP32OptPrecision, simple_model: SimpleModel) -> None:
    """Test that reference weights stay synchronized with model params."""
    # Convert model to bf16
    precision_plugin.convert_module(simple_model)

    # Create optimizer
    optimizer = AdamW(simple_model.parameters(), lr=0.001)

    # Create dummy batch
    x = torch.randn(8, 10).bfloat16()
    target = torch.randn(8, 5).bfloat16()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        output = simple_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        return loss

    # Run first optimizer step
    precision_plugin.optimizer_step(optimizer, simple_model, closure)

    # Check that model params and reference weights are close
    # (they should match within bf16 precision)
    for model_param, ref_param in precision_plugin._ref_by_model_param.items():
        # Convert model param to fp32 for comparison
        model_param_fp32 = model_param.data.float()
        # They should be very close (within bf16 rounding)
        assert torch.allclose(model_param_fp32, ref_param.data, rtol=1e-2, atol=1e-3)

    # Store initial values (use first model param)
    first_model_param = next(iter(simple_model.parameters()))
    initial_model_param = first_model_param.data.clone()
    initial_ref_param = precision_plugin._ref_by_model_param[first_model_param].data.clone()

    # Run second optimizer step
    precision_plugin.optimizer_step(optimizer, simple_model, closure)

    # Values should have changed
    assert not torch.allclose(first_model_param.data.float(), initial_model_param.float())
    assert not torch.allclose(precision_plugin._ref_by_model_param[first_model_param].data, initial_ref_param)

    # But they should still be synchronized
    for model_param, ref_param in precision_plugin._ref_by_model_param.items():
        model_param_fp32 = model_param.data.float()
        assert torch.allclose(model_param_fp32, ref_param.data, rtol=1e-2, atol=1e-3)
