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
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose
from hydra import initialize
from omegaconf import OmegaConf

from anemoi.training.commands import Command

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

        help_msg = "Dump Anemoi configs to a YAML file."
        dump = subparsers.add_parser(
            "dump",
            help=help_msg,
            description=help_msg,
        )
        dump.add_argument("--config-path", "-i", default=Path.cwd(), type=Path, help="Configuration directory")
        dump.add_argument("--name", "-n", default="config", help="Name of the configuration")
        dump.add_argument("--output", "-o", default="./config.yaml", type=Path, help="Output file path")
        dump.add_argument("--overwrite", "-f", action="store_true")

    def run(self, args: argparse.Namespace) -> None:
        self.overwrite = args.overwrite

        if args.subcommand == "generate":
            LOGGER.info("Generating configs, please wait.",)
            self.traverse_config(args.output)
            return

        if args.subcommand == "training-home":
            anemoi_home = Path.home() / ".config" / "anemoi" / "training" / "config"
            
            LOGGER.info("Inference checkpoint saved to %s", anemoi_home)
            self.traverse_config(anemoi_home)
            return

        if args.subcommand == "dump":
            LOGGER.info("Dumping config to %s", args.output)
            self.dump_config(args.config_path, args.name, args.output)
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
            LOGGER.info("Copied %s to %s", item.name, file_path)
        except Exception:
            LOGGER.exception("Failed to copy %s", item.name)

    def copy_files(self, source_directory: Path, target_directory: Path) -> None:
        """Copies directory files to a target directory."""
        for data in source_directory.rglob("*"):  # Recursively walk through all files and directories
            item = Path(data)
            if item.is_file() and item.suffix == ".yaml":
                file_path = Path(target_directory, item.relative_to(source_directory))

                file_path.parent.mkdir(parents=True, exist_ok=True)

                if not file_path.exists() or self.overwrite:
                    self.copy_file(item, file_path)
                else:
                    LOGGER.info("File %s already exists, skipping", file_path)        

    def dump_config(self, config_path: Path, name: str, output: Path) -> None:
        """Dump config files in one YAML file."""
        # Copy config files in tmp, use absolute path to avoid issues with hydra and shutil
        tmp_dir = Path(f"./.tmp_{time.time()}").absolute()
        output = output.absolute()    
        self.copy_files(config_path, tmp_dir)
        if not tmp_dir.exists():
            raise FileNotFoundError(f"No config files found in {config_path.absolute()}.")

        # Move to config directory to be able to handle hydra
        os.chdir(tmp_dir)
        with initialize(version_base=None, config_path="./"):
            cfg = compose(config_name=name)     

        # Dump configuration in output file
        LOGGER.info("Dumping file in %s.", output)  
        with output.open("w") as f:
            f.write(OmegaConf.to_yaml(cfg))              
        
        # Remove tmp dir
        os.chdir(tmp_dir.absolute().parent)
        for fp in tmp_dir.rglob("*"):
            if fp.is_file():
                os.remove(fp)
        LOGGER.info("Remove temporary directory %s.", output)  
        shutil.rmtree(tmp_dir)


command = ConfigGenerator
