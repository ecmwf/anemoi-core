# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose
from hydra import initialize

from anemoi.training.commands import Command
from anemoi.training.schemas.base_schema import BaseSchema

if TYPE_CHECKING:
    import argparse

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

        help_msg = "Generate the Anemoi training configs in home."
        anemoi_training_home = subparsers.add_parser(
            "training-home",
            help=help_msg,
            description=help_msg,
        )
        anemoi_training_home.add_argument("--overwrite", "-f", action="store_true")

        help_msg = "Validate the Anemoi training configs."
        validate = subparsers.add_parser("validate", help=help_msg, description=help_msg)

        validate.add_argument("--name", help="Name of the primary config file")
        validate.add_argument("--overwrite", "-f", action="store_true")

    def run(self, args: argparse.Namespace) -> None:

        self.overwrite = args.overwrite
        if args.subcommand == "generate":
            LOGGER.info(
                "Generating configs, please wait.",
            )
            self.traverse_config(args.output)

            LOGGER.info("Inference checkpoint saved to %s", args.output)
            return

        if args.subcommand == "training-home":
            anemoi_home = Path.home() / ".config" / "anemoi" / "training" / "config"
            LOGGER.info(
                "Generating configs, please wait.",
            )
            self.traverse_config(anemoi_home)
            LOGGER.info("Inference checkpoint saved to %s", anemoi_home)
            return

        if args.subcommand == "validate":
            LOGGER.info("Validating configs.")
            LOGGER.warning(
                "Note that this command is not taking into account if your config has a no_validation flag."
                "So this command will validate the config regardless of the flag.",
            )
            self.validate_config(args.name)
            LOGGER.info("Config files validated.")
            return

    def traverse_config(self, destination_dir: Path | str) -> None:
        """Writes the given configuration data to the specified file path."""
        config_package = "anemoi.training.config"

        # Ensure the destination directory exists
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Traverse through the package's config directory
        with pkg_resources.as_file(pkg_resources.files(config_package)) as config_path:
            for data in config_path.rglob("*"):  # Recursively walk through all files and directories
                item = Path(data)
                if item.is_file() and item.suffix == ".yaml":
                    file_path = Path(destination_dir, item.relative_to(config_path))

                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    if not file_path.exists() or self.overwrite:
                        self.copy_file(item, file_path)
                    else:
                        LOGGER.info("File %s already exists, skipping", file_path)

    @staticmethod
    def copy_file(item: Path, file_path: Path) -> None:
        """Copies the file to the destination directory."""
        try:
            shutil.copy2(item, file_path)
            LOGGER.info("Copied %s to %s", item.name, file_path)
        except Exception:
            LOGGER.exception("Failed to copy %s", item.name)

    def validate_config(self, name: Path | str) -> None:
        """Validates the configuration files in the given directory."""
        with initialize(version_base=None, config_path=""):
            cfg = compose(config_name=name)
            BaseSchema(**cfg)


command = ConfigGenerator
