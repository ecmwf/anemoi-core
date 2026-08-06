# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to dump raw metadata from a checkpoint as JSON or YAML."""

import contextlib
import json
from argparse import ArgumentParser
from argparse import Namespace

import yaml

from anemoi.utils.cli import Command


class DumpCommand(Command):
    """Dump raw metadata as JSON or YAML."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")
        command_parser.add_argument(
            "--output",
            metavar="FILE",
            help="Write output to FILE instead of stdout.",
        )
        command_parser.add_argument(
            "--yaml",
            action="store_true",
            help="Output in YAML format (requires PyYAML).",
        )
        command_parser.add_argument(
            "--json",
            action="store_true",
            help="Output in JSON format (default).",
        )

    def run(self, args: Namespace) -> None:
        """Execute the dump command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from ..checkpoint import extract_metadata_dict

        metadata = extract_metadata_dict(args.path)

        with contextlib.ExitStack() as stack:
            out_file = stack.enter_context(open(args.output, "w")) if args.output else None
            if args.yaml:
                print(yaml.dump(metadata, indent=2, sort_keys=True), file=out_file)
            else:
                # Default to JSON (also when --json is explicit).
                print(json.dumps(metadata, indent=4, sort_keys=True), file=out_file)


command = DumpCommand
