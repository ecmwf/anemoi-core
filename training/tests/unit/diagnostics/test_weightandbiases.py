from typing import Union

import pytest
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf

from anemoi.training.diagnostics.logger import get_wandb_logger
from anemoi.training.schemas.diagnostics import WandbSchema


@pytest.fixture
def config(tmp_path: str) -> Union[DictConfig, ListConfig]:
    """Create a config with offline mode and temporary log directory."""
    return OmegaConf.create(
        {
            "diagnostics": {
                "log": {
                    "wandb": {
                        "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
                        "enabled": True,
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
    logger_config = config.diagnostics.log
    run_id = config.training.run_id
    import torch

    model = torch.nn.Linear(10, 1)
    logger = get_wandb_logger(run_id, config.system.output, model, logger_config)
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
