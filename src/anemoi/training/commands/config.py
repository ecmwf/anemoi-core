# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import importlib.resources as pkg_resources
import logging
import shutil
from pathlib import Path

from . import Command

LOGGER = logging.getLogger(__name__)


class ConfigGenerator(Command):
    """Commands to interact with training configs."""

    @staticmethod
    def add_arguments(command_parser: argparse.ArgumentParser) -> None:
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        help_msg = "Generate the Anemoi training configs."
        generate = subparsers.add_parser(
            "generate",
            help=help,
            description=help_msg,
        )
        generate.add_argument("--output", "-o", default=Path.cwd(), help="Output directory")
        generate.add_argument("--overwrite", "-f", action="store_true")

        help_msg = "Generate the Anemoi training configs in home."
        anemoi_training_home = subparsers.add_parser(
            "training_home",
            help=help,
            description=help_msg,
        )
        anemoi_training_home.add_argument("--overwrite", "-f", action="store_true")

    def run(self, args: argparse.Namespace) -> None:
        LOGGER.info(
            "Generating configs, please wait.",
        )
        if args.subcommand == "generate":

            self.write_config(args.output, args.overwrite)

            LOGGER.info("Inference checkpoint saved to %s", args.output)
            return

        if args.subcommand == "training_home":
            anemoi_home = Path.home() / ".config" / "anemoi" / "training" / "config"
            self.write_config(anemoi_home, args.overwrite)
            LOGGER.info("Inference checkpoint saved to %s", anemoi_home)
            return

    @staticmethod
    def write_config(destination_dir: Path | str, overwrite: bool = False) -> None:
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

                    # Copy the file
                    if not file_path.exists() or overwrite:
                        try:
                            shutil.copy2(item, file_path)
                            LOGGER.info("Copied %s to %s", item.name, file_path)
                        except Exception:
                            LOGGER.exception("Failed to copy %s", item.name)
                    else:
                        LOGGER.info("File %s already exists, skipping", file_path)


command = ConfigGenerator
