# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import contextlib
import importlib.resources as pkg_resources
import logging
import os
import re
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

from hydra import compose
from hydra import initialize
from hydra import initialize_config_dir
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel

from anemoi.training.commands import Command
from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class ConfigGenerator(Command):
    """Commands to interact with training configs."""

    @staticmethod
    def add_arguments(command_parser: argparse.ArgumentParser) -> None:
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        help_msg = "Generate the Anemoi training configs."
        generate = subparsers.add_parser(
            "generate",
            help=help_msg,
            description=help_msg,
        )
        generate.add_argument("--output", "-o", default=Path.cwd(), help="Output directory")
        generate.add_argument("--overwrite", "-f", action="store_true")

        help_msg = "Validate the Anemoi training configs."
        validate = subparsers.add_parser("validate", help=help_msg, description=help_msg)

        validate.add_argument("--config-name", help="Name of the primary config file")
        validate.add_argument("--config-path", "-i", type=Path, default=None, help="Configuration directory")
        validate.add_argument("--overwrite", "-f", action="store_true")
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
        dump.add_argument("--config-path", "-i", default=None, type=Path, help="Configuration directory")
        dump.add_argument("--config-name", "-n", default="dev", help="Name of the configuration")
        dump.add_argument("--output", "-o", default="./config.yaml", type=Path, help="Output file path")
        dump.add_argument("--overwrite", "-f", action="store_true")
        dump.add_argument("--sort", "-s", action="store_true", help="Sort keys alphabetically in the output YAML")

    def run(self, args: argparse.Namespace) -> None:

        self.overwrite = args.overwrite

        if args.subcommand == "generate":
            LOGGER.info(
                "Generating configs, please wait.",
            )
            self.traverse_config(args.output)
            return

        if args.subcommand == "validate":
            LOGGER.info("Validating configs.")
            LOGGER.warning(
                "Note that this command is not taking into account if your config has set \
                    the config_validation flag to false."
                "So this command will validate the config regardless of the flag.",
            )
            self.validate_config(args.config_name, args.mask_env_vars, args.config_path)
            LOGGER.info("Config files validated.")
            return

        if args.subcommand == "dump":
            LOGGER.info("Dumping config to %s", args.output)
            self.dump_config(args.config_path, args.config_name, args.output, sort=args.sort)
            return

    def traverse_config(self, destination_dir: Path | str) -> None:
        """Writes the given configuration data to the specified file path."""
        config_package = "anemoi.training.config"

        # Ensure the destination directory exists
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Traverse through the package's config directory
        with pkg_resources.as_file(pkg_resources.files(config_package)) as config_path:
            self.copy_files(config_path, destination_dir)

    @staticmethod
    def copy_file(item: Path, file_path: Path) -> None:
        """Copies the file to the destination directory."""
        try:
            shutil.copy2(item, file_path)
            LOGGER.debug("Copied %s to %s", item.name, file_path)
        except Exception:
            LOGGER.exception("Failed to copy %s", item.name)

    def copy_files(self, source_directory: Path, target_directory: Path) -> None:
        """Copies directory files to a target directory."""
        for data in source_directory.rglob("*yaml"):  # Recursively walk through all files and directories
            item = Path(data)
            if item.is_file():
                file_path = Path(target_directory, item.relative_to(source_directory))

                file_path.parent.mkdir(parents=True, exist_ok=True)

                if not file_path.exists() or self.overwrite:
                    self.copy_file(item, file_path)
                else:
                    LOGGER.info("File %s already exists, skipping", file_path)

    def _mask_slurm_env_variables(self, cfg: DictConfig) -> None:
        """Mask environment variables are set."""
        # Convert OmegaConf dict to YAML format (raw string)
        raw_cfg = OmegaConf.to_yaml(cfg)
        # To extract and replace environment variables, loop through the matches
        updated_cfg = raw_cfg
        primitive_type_hints = extract_primitive_type_hints(BaseSchema)

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

    def validate_config(self, config_name: Path | str, mask_env_vars: bool, config_path: Path | None = None) -> None:
        """Validate a configuration, composed from ``config_path`` when given (see ``_initialize_hydra``)."""
        with _initialize_hydra(config_path):
            cfg = compose(config_name=config_name)
            if mask_env_vars:
                cfg = self._mask_slurm_env_variables(cfg)
            OmegaConf.resolve(cfg)
            BaseSchema(**cfg)

    def dump_config(self, config_path: Path | None, name: str, output: Path, sort: bool = False) -> None:
        """Dump config files in one YAML file, composed from ``config_path`` when given."""
        with _initialize_hydra(config_path):
            cfg = compose(config_name=name)

        LOGGER.info("Dumping file in %s.", output)
        with output.open("w") as f:
            f.write(OmegaConf.to_yaml(cfg, sort_keys=sort))


@contextlib.contextmanager
def _initialize_hydra(config_path: Path | None) -> Generator[None, None, None]:
    """Initialise Hydra for config composition, shared by ``validate`` and ``dump``.

    With ``config_path`` the configs are composed from that directory, installed as Hydra's
    primary (``main``) search-path entry so it takes precedence over the current working
    directory and the packaged defaults — the same model ``@hydra.main`` gives ``train``'s
    ``--config-path``. Without it, discovery is left to the ``AnemoiSearchPathPlugin`` (the
    current working directory and the packaged defaults). ``initialize_config_dir`` requires
    an absolute path.
    """
    if config_path is not None:
        with initialize_config_dir(version_base=None, config_dir=str(Path(config_path).absolute())):
            yield
    else:
        with initialize(version_base=None):
            yield


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


command = ConfigGenerator
