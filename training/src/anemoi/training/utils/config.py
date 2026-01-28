# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.training.config_types import Settings
from anemoi.training.config_types import get_path


def as_settings(config: Any) -> Settings:
    """Normalize config into a typed Settings instance."""
    if isinstance(config, Settings):
        return config
    if hasattr(config, "model_dump"):
        return Settings.model_validate(config.model_dump(by_alias=True))
    if isinstance(config, DictConfig):
        OmegaConf.resolve(config)
        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, Mapping):
            msg = "Config must be a mapping."
            raise TypeError(msg)
        return Settings.model_validate(container)
    if isinstance(config, Mapping):
        return Settings.model_validate(config)
    msg = f"Config must be a mapping or Pydantic model, got {type(config).__name__}."
    raise TypeError(msg)


def expand_paths(config_system: Settings | Any) -> Settings | Any:
    """Expand output/logging paths relative to output root."""
    output_config = config_system.output
    root_output_path = Path(output_config.root) if output_config.root else Path()

    if output_config.plots:
        config_system.output.plots = root_output_path / output_config.plots
    if output_config.profiler:
        config_system.output.profiler = root_output_path / output_config.profiler

    if output_config.logs is not None:
        output_config.logs.root = (
            root_output_path / output_config.logs.root if output_config.logs.root else root_output_path
        )
        base = output_config.logs.root

        output_config.logs.wandb = (
            base / "wandb" if output_config.logs.wandb is None else base / output_config.logs.wandb
        )
        output_config.logs.mlflow = (
            base / "mlflow" if output_config.logs.mlflow is None else base / output_config.logs.mlflow
        )
        output_config.logs.tensorboard = (
            base / "tensorboard" if output_config.logs.tensorboard is None else base / output_config.logs.tensorboard
        )

    if output_config.checkpoints is not None:
        output_config.checkpoints.root = (
            root_output_path / output_config.checkpoints.root if output_config.checkpoints.root else root_output_path
        )

    return config_system


def postprocess_config(config: Settings, *, validate: bool = True) -> Settings:
    """Apply post-processing and lightweight validation to Settings."""
    if hasattr(config, "system"):
        expand_paths(config.system)

    log_cfg = config.diagnostics.log
    mlflow_cfg = log_cfg.mlflow if log_cfg is not None else None
    mlflow_enabled = bool(mlflow_cfg and mlflow_cfg.enabled)
    mlflow_save_dir = mlflow_cfg.save_dir if mlflow_cfg is not None else None
    output_logs = config.system.output.logs
    mlflow_root = output_logs.mlflow if output_logs is not None else None
    if mlflow_enabled and mlflow_root is not None and mlflow_save_dir != mlflow_root:
        mlflow_cfg.save_dir = str(mlflow_root)

    if not validate:
        return config

    if hasattr(config, "dataloader") and not config.dataloader.read_group_size:
        config.dataloader.read_group_size = config.system.hardware.num_gpus_per_model

    decoder_init_zero = get_path(config, "model.decoder.initialise_data_extractor_zero")
    if decoder_init_zero and get_path(config, "model.bounding"):
        msg = (
            "Boundings cannot be used with zero initialized weights in decoder. "
            "Set initialise_data_extractor_zero to False."
        )
        raise ValueError(msg)

    return config


__all__ = ["as_settings", "expand_paths", "postprocess_config"]
