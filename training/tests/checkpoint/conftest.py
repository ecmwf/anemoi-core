# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Common test fixtures for checkpoint tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

from anemoi.training.checkpoint import CheckpointContext
from anemoi.training.checkpoint import PipelineStage
from anemoi.training.checkpoint.exceptions import CheckpointError

if TYPE_CHECKING:
    from pathlib import Path


class MockStage(PipelineStage):
    """Mock pipeline stage for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.process_called = False
        self.context_received = None

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Process the context."""
        self.process_called = True
        self.context_received = context

        if self.should_fail:
            error_msg = f"Stage {self.name} failed"
            raise CheckpointError(error_msg)

        # Add marker to metadata
        context.update_metadata(**{f"stage_{self.name}": "processed"})
        return context


class SimpleModel(nn.Module):
    """Simple model for testing checkpoint operations."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


class ComplexModel(nn.Module):
    """More complex model for testing advanced scenarios."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
        )
        self.processor = nn.ModuleList(
            [
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ],
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.encoder(x)
        for layer in self.processor:
            x = layer(x)
        return self.decoder(x)


@pytest.fixture
def simple_model() -> SimpleModel:
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def complex_model() -> ComplexModel:
    """Create a complex model for testing advanced scenarios."""
    return ComplexModel()


@pytest.fixture
def sample_state_dict(simple_model: SimpleModel) -> dict:
    """Create a sample state dictionary from simple model."""
    return simple_model.state_dict()


@pytest.fixture
def complex_state_dict(complex_model: ComplexModel) -> dict:
    """Create a complex state dictionary for advanced testing."""
    return complex_model.state_dict()


@pytest.fixture
def sample_optimizer(simple_model: SimpleModel) -> torch.optim.Adam:
    """Create a sample optimizer for testing."""
    return torch.optim.Adam(simple_model.parameters(), lr=1e-3)


@pytest.fixture
def sample_scheduler(sample_optimizer: torch.optim.Adam) -> torch.optim.lr_scheduler.StepLR:
    """Create a sample scheduler for testing."""
    return torch.optim.lr_scheduler.StepLR(sample_optimizer, step_size=10)


@pytest.fixture
def lightning_checkpoint(sample_state_dict: dict) -> dict:
    """Create a mock Lightning checkpoint."""
    return {
        "state_dict": sample_state_dict,
        "pytorch-lightning_version": "2.0.0",
        "epoch": 10,
        "global_step": 1000,
        "optimizer_states": [{"state": {}, "param_groups": [{"lr": 1e-3}]}],
        "lr_schedulers": [{"scheduler": {"step_size": 10}, "_step_count": 1}],
        "callbacks": {"ModelCheckpoint": {"best_model_score": 0.95}},
        "loops": {"fit_loop": {"epoch_loop": {"batch_loop": {}}}},
        "hyper_parameters": {"batch_size": 32, "learning_rate": 1e-3},
    }


@pytest.fixture
def pytorch_checkpoint(sample_state_dict: dict) -> dict:
    """Create a mock PyTorch checkpoint."""
    return {
        "model_state_dict": sample_state_dict,
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-3}]},
        "scheduler_state_dict": {"step_size": 10, "_step_count": 1},
        "epoch": 5,
        "global_step": 500,
        "loss": 0.123,
        "best_accuracy": 0.89,
        "training_time": 3600,
    }


@pytest.fixture
def minimal_checkpoint(sample_state_dict: dict) -> dict:
    """Create a minimal checkpoint with just state dict."""
    return sample_state_dict


@pytest.fixture
def corrupted_checkpoint_data() -> dict:
    """Create intentionally corrupted checkpoint data for error testing."""
    return {
        "state_dict": {"layer.weight": "not_a_tensor"},  # String instead of tensor
        "optimizer_state_dict": None,  # None instead of dict
        "epoch": "ten",  # String instead of int
    }


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for checkpoint files."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def checkpoint_files(
    temp_checkpoint_dir: Path,
    lightning_checkpoint: dict,
    pytorch_checkpoint: dict,
    sample_state_dict: dict,
) -> dict[str, Path]:
    """Create various checkpoint files for testing."""
    files = {}

    # Lightning checkpoint
    lightning_path = temp_checkpoint_dir / "lightning_model.ckpt"
    torch.save(lightning_checkpoint, lightning_path)
    files["lightning"] = lightning_path

    # PyTorch checkpoint
    pytorch_path = temp_checkpoint_dir / "pytorch_model.pt"
    torch.save(pytorch_checkpoint, pytorch_path)
    files["pytorch"] = pytorch_path

    # Raw state dict
    state_dict_path = temp_checkpoint_dir / "state_dict.pth"
    torch.save(sample_state_dict, state_dict_path)
    files["state_dict"] = state_dict_path

    return files


@pytest.fixture
def large_checkpoint_data() -> dict:
    """Create large checkpoint data for performance testing."""
    # Create large tensors to simulate real-world checkpoints
    large_state_dict = {f"layer_{i}.weight": torch.randn(512, 512) for i in range(10)}
    large_state_dict.update({f"layer_{i}.bias": torch.randn(512) for i in range(10)})

    return {
        "state_dict": large_state_dict,
        "optimizer_state_dict": {
            "state": {i: {"momentum_buffer": torch.randn(512, 512)} for i in range(100)},
            "param_groups": [{"lr": 1e-3}],
        },
        "epoch": 50,
        "global_step": 10000,
        "loss_history": [0.5 - 0.001 * i for i in range(1000)],  # Large metadata
    }


@pytest.fixture
def mock_config(tmp_path: Path) -> DictConfig:
    """Create a mock configuration for testing."""
    return DictConfig(
        {
            "checkpoint": {
                "cache_dir": str(tmp_path / "checkpoints"),
                "max_retries": 3,
                "timeout": 300,
            },
            "pipeline": {
                "async_execution": True,
                "continue_on_error": False,
            },
        },
    )


@pytest.fixture
def network_urls() -> dict[str, str]:
    """Provide test URLs for network operations."""
    return {
        "valid": "https://httpbin.org/bytes/1024",  # Returns 1KB of data
        "timeout": "https://httpbin.org/delay/10",  # Delays 10 seconds
        "not_found": "https://httpbin.org/status/404",  # Returns 404
        "server_error": "https://httpbin.org/status/500",  # Returns 500
        "large_file": "https://httpbin.org/bytes/10485760",  # 10MB file
    }


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path: Path) -> None:
    """Automatically cleanup temporary files after each test."""
    # Cleanup happens automatically with tmp_path, but we can add custom cleanup here if needed


# Parameterized fixtures for testing multiple formats
@pytest.fixture(params=["lightning", "pytorch", "state_dict"])
def checkpoint_format(request: pytest.FixtureRequest) -> str:
    """Parametrize tests across different checkpoint formats."""
    return request.param


@pytest.fixture(params=["cpu", "meta"])
def map_location(request: pytest.FixtureRequest) -> str:
    """Parametrize tests across different map locations."""
    return request.param


# Custom markers for test categorization
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "network: marks tests as requiring network access")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
