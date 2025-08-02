# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import re
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf
from pahtlib import Path

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.utils.schemas import BaseModel

from . import Command

LOGGER = logging.getLogger(__name__)


class Validate(Command):
    """Validate a graph."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument("config", help="Path to the configuration file (a YAML file).")
        command_parser.add_argument(
            "--mask_env_vars",
            "-m",
            help="Mask environment variables from config. Default False",
            action="store_true",
        )

    def _mask_slurm_env_variables(self, cfg: DictConfig) -> None:
        """Mask environment variables are set."""
        # Convert OmegaConf dict to YAML format (raw string)
        raw_cfg = OmegaConf.to_yaml(cfg)
        # To extract and replace environment variables, loop through the matches
        updated_cfg = raw_cfg
        primitive_type_hints = extract_primitive_type_hints(BaseGraphSchema)

        patterns = [
            r"(\w+):\s*\$\{oc\.env:([A-Z0-9_]+)\}(?!\})",
            r"(\w+):\s*\$\{oc\.decode:\$\{oc\.env:([A-Z0-9_]+)\}\}",
        ]
        replaces = ["${{oc.env:{match}}}", "${{oc.decode:${{oc.env:{match}}}}}"]
        # Find all matches in the raw_cfg string
        for pattern, replace in zip(patterns, replaces, strict=False):
            matches = re.findall(pattern, raw_cfg)
            # Find the corresponding type hints for each extracted key
            for extracted_key, match in matches:
                corresponding_keys = next(iter([key for key in primitive_type_hints if extracted_key in key]))
                # Check if the environment variable exists
                env_value = os.getenv(match)

                # If environment variable doesn't exist, replace with default string
                if env_value is None:
                    def_str = "default"
                    def_int = 0
                    def_bool = True
                    if primitive_type_hints[corresponding_keys] is str:
                        env_value = def_str
                    elif primitive_type_hints[corresponding_keys] in [int, float]:
                        env_value = def_int
                    elif primitive_type_hints[corresponding_keys] is bool:
                        env_value = def_bool
                    elif primitive_type_hints[corresponding_keys] is Path:
                        env_value = Path(def_str)
                    else:
                        msg = "Type not supported for masking environment variables"
                        raise TypeError(msg)
                    LOGGER.warning("Environment variable %s not found, masking with %s", match, env_value)
                    # Replace the pattern with the actual value or the default string
                    updated_cfg = updated_cfg.replace(replace.format(match=match), str(env_value))

        return OmegaConf.create(updated_cfg)

    def run(self, args):
        LOGGER.info("Validating configs.")
        LOGGER.warning(
            "Note that this command is not taking into account if your config has set \
                the config_validation flag to false."
            "So this command will validate the config regardless of the flag.",
        )
        config = OmegaConf.load(args.config, resolve=True)

        # Mask environment variables if requested
        if args.mask_env_vars:
            LOGGER.info("Masking environment variables in the config.")
            config = self._mask_slurm_env_variables(config)

        # Validate the config
        GraphCreator(config=config, config_validation=True)
        LOGGER.info("Config files validated.")


def extract_primitive_type_hints(model: type[BaseModel], prefix: str = "") -> dict[str, Any]:
    field_types = {}

    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name

        # Check if the field type has 'model_fields' (indicating a nested Pydantic model)
        if hasattr(field_type, "model_fields"):
            field_types.update(extract_primitive_type_hints(field_type, full_field_name))
        else:
            try:
                field_types[full_field_name] = field_type.__args__[0]
            except AttributeError:
                field_types[full_field_name] = field_type

    return field_types


command = Validate
