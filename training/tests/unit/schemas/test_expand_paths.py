from pathlib import Path
from typing import Union

import pytest
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.schemas.base_schema import expand_paths


@pytest.fixture
def base_config() -> DictConfig:
    """Create a minimal config structure for testing."""
    return OmegaConf.create(
        {
            "output": {
                "root": "runs",
                "plots": "plots",
                "profiler": "profiler",
                "logs": {
                    "root": "logs",
                    "wandb": None,
                    "mlflow": None,
                    "tensorboard": None,
                },
                "checkpoints": {"root": "checkpoints"},
            },
        },
    )


def test_expand_paths_returns_same_object(base_config: DictConfig) -> None:
    """It should return the same object (identity preserved)."""
    returned = expand_paths(base_config)
    assert returned is base_config, "expand_paths should return the same instance"


def test_expand_paths_basic(base_config: DictConfig) -> None:
    """Test that expand_paths correctly expands all subpaths."""
    cfg = OmegaConf.create(base_config)

    result = expand_paths(cfg)
    output = result.output
    root = Path("runs")

    # --- top-level paths ---
    assert output.plots == root / "plots"
    assert output.profiler == root / "profiler"

    # --- logs ---
    assert output.logs.root == root / "logs"
    assert output.logs.wandb == root / "logs" / "wandb"
    assert output.logs.mlflow == root / "logs" / "mlflow"
    assert output.logs.tensorboard == root / "logs" / "tensorboard"

    # --- checkpoints ---
    assert output.checkpoints.root == root / "checkpoints"


def test_expand_paths_respects_existing_values(base_config: DictConfig) -> None:
    """If wandb/mlflow/tensorboard are already set, keep and prefix them."""
    base_config.output.logs.wandb = "custom_wandb"
    base_config.output.logs.mlflow = "custom_mlflow"
    base_config.output.logs.tensorboard = "custom_tb"

    result = expand_paths(base_config)

    root = Path("runs") / "logs"
    assert result.output.logs.wandb == root / "custom_wandb"
    assert result.output.logs.mlflow == root / "custom_mlflow"
    assert result.output.logs.tensorboard == root / "custom_tb"


@pytest.mark.parametrize("root_value", [None, ""])
def test_expand_paths_with_no_root(root_value: Union[None, str]) -> None:
    """When root is None or empty, relative paths should not be prefixed."""
    config = OmegaConf.create(
        {
            "output": {
                "root": root_value,
                "plots": "plots",
                "profiler": "profiler",
                "logs": {
                    "root": "logs",
                    "wandb": None,
                    "mlflow": None,
                    "tensorboard": None,
                },
                "checkpoints": {"root": "checkpoints"},
            },
        },
    )

    result = expand_paths(config)
    output = result.output

    # root is None -> no prefix
    assert output.plots == Path("plots")
    assert output.profiler == Path("profiler")
    assert output.logs.root == Path("logs")
    assert output.logs.wandb == Path("logs") / "wandb"
    assert output.checkpoints.root == Path("checkpoints")


@pytest.mark.parametrize("root_value", [None, ""])
def test_expand_paths_with_no_root_but_full_paths(root_value: Union[None, str]) -> None:
    """When root is None or empty, relative paths should not be prefixed."""
    config = OmegaConf.create(
        {
            "output": {
                "root": root_value,
                "plots": "/test/plots",
                "profiler": "/home/profiler",
                "logs": {
                    "root": "/scratch/logs",
                    "wandb": None,
                    "mlflow": None,
                    "tensorboard": None,
                },
                "checkpoints": {"root": "/scratch/checkpoints"},
            },
        },
    )

    result = expand_paths(config)
    output = result.output

    # root is None -> no prefix
    assert output.plots == Path("/test") / Path("plots")
    assert output.profiler == Path("/home") / Path("profiler")
    assert output.logs.root == Path("/scratch") / Path("logs")
    assert output.logs.wandb == Path("/scratch") / Path("logs") / "wandb"
    assert output.checkpoints.root == Path("/scratch") / Path("checkpoints")
