from typing import Union

import pytest
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger

from anemoi.training.schemas.diagnostics import WandbSchema


@pytest.fixture
def config(tmp_path: str) -> Union[DictConfig, ListConfig]:
    """Create a config with offline mode and temporary log directory."""
    return OmegaConf.create(
        {
            "diagnostics": {
                "log": {
                    "wandb": {
                        "project": "pytest_project",
                        "entity": "localtest",
                        "offline": True,
                        "log_model": False,
                        "gradients": True,
                        "parameters": False,
                        "interval": 5,
                    },
                },
            },
            "training": {"run_id": None},
            "hardware": {"paths": {"logs": {"wandb": str(tmp_path / "wandb_logs")}}},
        },
    )


def test_wandb_logger_offline(tmp_path: str, config: DictConfig) -> None:
    """Test W&B logger initialization and offline logging.

    This will create local wandb logs inside a temporary pytest directory.
    """
    # Initialize the logger
    logger = WandbLogger(
        project=config.diagnostics.log.wandb.project,
        entity=config.diagnostics.log.wandb.entity,
        save_dir=config.hardware.paths.logs.wandb,
        offline=config.diagnostics.log.wandb.offline,
        log_model=config.diagnostics.log.wandb.log_model,
        resume=False,
    )

    # Log hyperparameters
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    # --- Assertions ---
    log_dir = tmp_path / "wandb_logs"
    files = list(log_dir.glob("**/*"))
    assert log_dir.exists(), "W&B offline directory was not created"
    assert any(f.name.startswith("wandb") for f in files), "No W&B log files were generated"


def test_weights_and_biases_schema() -> None:
    config = {
        "enabled": False,
        "offline": False,
        "log_model": False,
        "project": "Anemoi",
        "entity": "Anemoi",
        "gradients": False,
        "parameters": False,
    }
    schema = WandbSchema(**config)

    assert not schema.enabled
