# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
import sys
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from pydantic._internal import _model_construction
from pydantic_core import PydanticCustomError
from pydantic_core import ValidationError
from typing_extensions import Self

from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.models.schemas.decoder import GraphTransformerDecoderSchema
from anemoi.models.schemas.models import ModelSchema
from anemoi.utils.schemas import BaseModel
from anemoi.utils.schemas.errors import CUSTOM_MESSAGES
from anemoi.utils.schemas.errors import convert_errors

# to make these available at runtime for pydantic, bug should be resolved in
# future versions (see https://github.com/astral-sh/ruff/issues/7866)
from .data import DataSchema
from .dataloader import DataLoaderSchema
from .datamodule import DataModuleSchema
from .diagnostics import DiagnosticsSchema
from .hardware import HardwareSchema
from .training import TrainingSchema

_object_setattr = _model_construction.object_setattr

LOGGER = logging.getLogger(__name__)


class BaseSchema(BaseModel):
    """Top-level schema for the training configuration."""

    data: DataSchema
    """Data configuration."""
    dataloader: DataLoaderSchema
    """Dataloader configuration."""
    datamodule: DataModuleSchema
    """Datamodule configuration."""
    diagnostics: DiagnosticsSchema
    """Diagnostics configuration such as logging, plots and metrics."""
    hardware: HardwareSchema
    """Hardware configuration."""
    graph: BaseGraphSchema
    """Graph configuration."""
    model: ModelSchema
    """Model configuration."""
    training: TrainingSchema
    """Training configuration."""
    config_validation: bool = True
    """Flag to disable validation of the configuration"""

    @model_validator(mode="after")
    def set_read_group_size_if_not_provided(self) -> Self:
        if not self.dataloader.read_group_size:
            self.dataloader.read_group_size = self.hardware.num_gpus_per_model
        return self

    @model_validator(mode="after")
    def check_log_paths_available_for_loggers(self) -> Self:
        logger = []
        if self.diagnostics.log.wandb.enabled and (not self.hardware.paths.logs or not self.hardware.paths.logs.wandb):
            logger.append("wandb")
        if self.diagnostics.log.mlflow.enabled and (
            not self.hardware.paths.logs or not self.hardware.paths.logs.mlflow
        ):
            logger.append("mlflow")
        if self.diagnostics.log.tensorboard.enabled and (
            not self.hardware.paths.logs or not self.hardware.paths.logs.tensorboard
        ):
            logger.append("tensorboard")

        if logger:
            msg = ", ".join(logger) + " logging path(s) not provided."
            raise PydanticCustomError("logger_path_missing", msg)  # noqa: EM101
        return self

    @model_validator(mode="after")
    def check_bounding_not_used_with_data_extractor_zero(self) -> Self:
        """Check that bounding is not used with zero data extractor."""
        if (
            isinstance(self.model.decoder, GraphTransformerDecoderSchema)
            and self.model.decoder.initialise_data_extractor_zero
            and self.model.bounding
        ):
            error = "bounding_conflict_with_data_extractor_zero"
            msg = (
                "Boundings cannot be used with zero initialized weights in decoder. "
                "Set initalise_data_extractor_zero to False."
            )
            raise PydanticCustomError(
                error,
                msg,
            )
        return self

    @model_validator(mode="after")
    def check_mass_conserving_accumulations(self) -> Self:
        """Validate mass-conserving accumulation configuration when enabled."""
        mca = getattr(self.model, "mass_conserving_accumulations", None)
        if not mca:
            return self

        # Ensure mapping is dict-like
        try:
            pairs = list(mca.items())
        except Exception as e:  # noqa: BLE001
            error_title = "mass_conserving_accumulations_invalid"
            error_msg = f"model.mass_conserving_accumulations must be a mapping of target->constraint. Error: {e}"
            raise PydanticCustomError(
                error_title,
                error_msg,
            ) from None

        targets = [t for t, _ in pairs]
        constraints = [c for _, c in pairs]

        # 1) Targets must be in data.diagnostic
        diagnostic = self.data.diagnostic or []
        missing_diag = [t for t in targets if t not in diagnostic]
        if missing_diag:
            error_title = "mass_conserving_accumulations_target_not_diagnostic"
            error_msg = f"The following targets must be listed in data.diagnostic: {missing_diag}"
            raise PydanticCustomError(
                error_title,
                error_msg,
            )

        # 2) Constraints must be in data.forcing
        forcing = self.data.forcing or []
        missing_forcing = [c for c in constraints if c not in forcing]
        if missing_forcing:
            error_title = "mass_conserving_accumulations_constraint_not_forcing"
            error_msg = f"The following constraint variables must be listed in data.forcing: {missing_forcing}"
            raise PydanticCustomError(
                error_title,
                error_msg,
            )

        # 3) zero_overwriter processor must exist and include all constraint vars in at least one group's vars
        processors = getattr(self.data, "processors", {}) or {}
        zero_overwriter = processors.get("zero_overwriter")
        if zero_overwriter is None:
            error_title = "zero_overwriter_missing"
            error_msg = (
                "data.processors.zero_overwriter must be defined to zero the left boundary of accumulated inputs."
            )
            raise PydanticCustomError(
                error_title,
                error_msg,
            )

        zow_cfg = getattr(zero_overwriter, "config", None) or {}
        groups = zow_cfg.get("groups") or []
        zow_vars: set[str] = set()
        for grp in groups:
            # if isinstance(grp, dict):
            v = grp.get("vars")
            # if isinstance(v, list):
            zow_vars.update(v)

        missing_zow_vars = [c for c in constraints if c not in zow_vars]
        if missing_zow_vars:
            error_title = "zero_overwriter_missing_vars"
            error_msg = f"""The following constraint variables must appear in at least one
            data.processors.zero_overwriter.config.groups[*].vars list: {missing_zow_vars}"""
            raise PydanticCustomError(error_title, error_msg)

        # 4) normalizer remap must include each target -> constraint mapping
        normalizer_proc = processors.get("normalizer", None) or {}
        norm_cfg = getattr(normalizer_proc, "config", None) or {}
        remap = norm_cfg.get("remap", {})

        remap_mismatch = [t for (t, c) in pairs if remap.get(t) != c]
        if remap_mismatch:
            error_title = "normalizer_remap_mismatch"
            error_msg = f"""data.processors.normalizer.config.remap must map each target to its corresponding constraint
             variable. {remap_mismatch}. By default every pairing in config.model.mass_conserving_accumulations must be
             included in config.data.processors.normalizer.config.remap"""

            raise PydanticCustomError(error_title, error_msg)

        return self

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)


class UnvalidatedBaseSchema(PydanticBaseModel):
    data: Any
    """Data configuration."""
    dataloader: Any
    """Dataloader configuration."""
    datamodule: Any
    """Datamodule configuration."""
    diagnostics: Any
    """Diagnostics configuration such as logging, plots and metrics."""
    hardware: Any
    """Hardware configuration."""
    graph: Any
    """Graph configuration."""
    model: Any
    """Model configuration."""
    training: Any
    """Training configuration."""
    config_validation: bool = False
    """Flag to disable validation of the configuration"""

    def model_dump(self, by_alias: bool = False) -> dict:
        dumped_model = super().model_dump(by_alias=by_alias)
        return DictConfig(dumped_model)


def convert_to_omegaconf(config: BaseSchema) -> dict:
    config = config.model_dump(by_alias=True)
    return OmegaConf.create(config)


def validate_schema(config: DictConfig) -> BaseSchema:
    try:
        config = BaseSchema(**config)
    except ValidationError as e:
        errors = convert_errors(e, CUSTOM_MESSAGES)
        LOGGER.error(errors)  # noqa: TRY400
        sys.exit(0)
    else:
        return config
