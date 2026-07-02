# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to load metadata from a file into a checkpoint."""

import json
import logging
import os
from argparse import ArgumentParser
from argparse import Namespace

import yaml

from anemoi.utils.cli import Command

LOG = logging.getLogger(__name__)


class LoadCommand(Command):
    """Load metadata from a file into a checkpoint."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")
        command_parser.add_argument(
            "--input",
            metavar="FILE",
            required=True,
            help="JSON or YAML file containing the new metadata.",
        )
        command_parser.add_argument(
            "--yaml",
            action="store_true",
            help="Treat the input file as YAML (requires PyYAML).",
        )
        command_parser.add_argument(
            "--json",
            action="store_true",
            help="Treat the input file as JSON (default).",
        )

    def run(self, args: Namespace) -> None:
        """Execute the load command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.

        Raises
        ------
        ValueError
            If the file extension is unknown and neither --json nor --yaml
            is specified.
        """
        from ..checkpoint import has_metadata
        from ..checkpoint import replace_metadata
        from ..checkpoint import save_metadata

        _, ext = os.path.splitext(args.input)

        if ext == ".json" or args.json:
            with open(args.input) as f:
                new_metadata = json.load(f)
        elif ext in (".yaml", ".yml") or args.yaml:
            with open(args.input) as f:
                new_metadata = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unknown file extension '{ext}'." " Use --json or --yaml to specify the format explicitly."
            )

        if has_metadata(args.path):
            replace_metadata(args.path, new_metadata)
        else:
            LOG.warning(
                "Checkpoint %s has no existing metadata; adding new metadata entry.",
                args.path,
            )
            save_metadata(args.path, new_metadata)

        LOG.info("Metadata written to %s", args.path)


command = LoadCommand
