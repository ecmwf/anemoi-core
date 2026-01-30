from typing import Union

import pytest
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from anemoi.training.schemas.diagnostics import WandbSchema


@pytest.fixture
def config(tmp_path: str) -> Union[DictConfig, ListConfig]:
    """Create a config with offline mode and temporary log directory."""
    return OmegaConf.create(
        {
            "diagnostics": {
                "log": {
                    "wandb": {
                        "_target_": "anemoi.training.diagnostics.wandb.logger.WandbLogger",
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
            "system": {"output": {"logs": {"wandb": str(tmp_path / "wandb_logs")}}},
        },
    )


def test_wandb_logger_offline(tmp_path: str, config: DictConfig) -> None:
    """Test W&B logger initialization and offline logging.

    This will create local wandb logs inside a temporary pytest directory.
    """
    logger_config = config.diagnostics.log.wandb
    save_dir = config.system.output.logs.wandb
    run_id = config.training.run_id

    # Initialize the logger
    logger = instantiate(
        logger_config,
        run_id=run_id,
        save_dir=save_dir,
        resume=run_id is not None,
    )
    # Log hyperparameters
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    # --- Assertions ---
    log_dir = tmp_path / "wandb_logs"
    files = list(log_dir.glob("**/*"))
    assert log_dir.exists(), "W&B offline directory was not created"
    assert any(f.name.startswith("wandb") for f in files), "No W&B log files were generated"


def test_weights_and_biases_schema_backward_compatibility() -> None:
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
