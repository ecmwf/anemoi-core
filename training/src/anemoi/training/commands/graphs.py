# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import contextlib
import logging
import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

from hydra import compose
from hydra import initialize
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel

from anemoi.training.commands import Command
from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.graphs.create import GraphCreator
from anemoi.graphs.describe import GraphDescriptor

LOGGER = logging.getLogger(__name__)


class GraphCreation(Command):
    """Commands to interact with training configs."""

    @staticmethod
    def add_arguments(command_parser: argparse.ArgumentParser) -> None:
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        help_msg = "Create the graph."
        generate = subparsers.add_parser(
            "create",
            help=help_msg,
            description=help_msg,
        )
        generate.add_argument("--config-name", help="Name of the primary config file")
        generate.add_argument(
            "--description",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Show the description of the graph.",
        )
        generate.add_argument("--overwrite", "-f", action="store_true")
        generate.add_argument(
            "--mask_env_vars",
            "-m",
            help="Mask environment variables from config. Default False",
            action="store_true",
        )

        help_msg = "Validate the anemoi graph config."
        validate = subparsers.add_parser("validate", help=help_msg, description=help_msg)

        validate.add_argument("--config-name", help="Name of the primary config file")
        validate.add_argument(
            "--mask_env_vars",
            "-m",
            help="Mask environment variables from config. Default False",
            action="store_true",
        )

        help_msg = "Dump Anemoi configs to a YAML file."
        dump = subparsers.add_parser(
            "dump",
            help=help_msg,
            description=help_msg,
        )
        dump.add_argument("--config-name", "-n", default="dev", help="Name of the configuration")
        dump.add_argument("--output", "-o", default="./graph_config.yaml", type=Path, help="Output config path")
        dump.add_argument(
            "--mask_env_vars",
            "-m",
            help="Mask environment variables from config. Default False",
            action="store_true",
        )

    def run_create(self, graph_config, graph_filename, args: argparse.Namespace) -> None:
        if graph_filename.exists() and not args.overwrite:
            LOGGER.warning(
                "Graph already exists at %s. Skipping creation. Use --overwrite to regenerate.",
                graph_filename,
            )
            if args.description:
                GraphDescriptor(graph_filename).describe()
            return

        graph_creator = GraphCreator(config=graph_config)
        graph_creator.create(save_path=graph_filename, overwrite=args.overwrite)
        if args.description:
            if graph_filename.exists():
                GraphDescriptor(graph_filename).describe()
            else:
                print("Graph description is not shown if the graph is not saved.")

    def run(self, args: argparse.Namespace) -> None:

        self.overwrite = args.overwrite

        graph_config = self.get_graph_config(args.config_name, args.mask_env_vars)
        graph_filename = self.get_graph_filename(args.config_name, args.mask_env_vars)

        if args.subcommand == "create":
            LOGGER.info("Generating graph, please wait.")
            self.run_create(graph_config, graph_filename, args)
            return

        if args.subcommand == "validate":
            LOGGER.info("Validating configs.")
            LOGGER.warning(
                "Note that this command is not taking into account if your config has set \
                    the config_validation flag to false."
                "So this command will validate the config regardless of the flag.",
            )
            self.validate_config(graph_config)
            LOGGER.info("Config files validated.")
            return

        if args.subcommand == "dump":
            LOGGER.info("Dumping config to %s", args.output)
            self.dump_config(graph_config, args.output)
            return

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

    def get_graph_config(self, config_name: Path | str, mask_env_vars: bool) -> DictConfig:
        """Loads the graph configuration file."""
        with initialize(version_base=None, config_path=""):
            cfg = compose(config_name=config_name)
            if mask_env_vars:
                cfg = self._mask_slurm_env_variables(cfg)
            OmegaConf.resolve(cfg)
            return cfg.graph

    def get_graph_filename(self, config_name: Path | str, mask_env_vars: bool) -> Path:
        """Gets the graph filename from the configuration."""
        with initialize(version_base=None, config_path=""):
            cfg = compose(config_name=config_name)
            if mask_env_vars:
                cfg = self._mask_slurm_env_variables(cfg)
            OmegaConf.resolve(cfg)
            return Path(cfg.hardware.paths.graph) / cfg.hardware.files.graph

    def validate_config(self, graph_config: DictConfig) -> None:
        """Validates the configuration files in the given directory."""
        with initialize(version_base=None, config_path=""):
            OmegaConf.resolve(graph_config)
            BaseGraphSchema(**graph_config)

    def dump_config(self, graph_config: DictConfig, output: Path) -> None:
        """Dump config files in one YAML file."""
        # Dump configuration in output file
        LOGGER.info("Dumping file in %s.", output)
        with output.open("w") as f:
            f.write(OmegaConf.to_yaml(graph_config))


@contextlib.contextmanager
def change_directory(destination: Path) -> Generator[None, None, None]:
    """A context manager to temporarily change the current working directory."""
    original_directory = Path.cwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(original_directory)


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


command = GraphCreation
