# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to view checkpoint metadata in a pager (read-only)."""

import json
import os
import shlex
import subprocess
from argparse import ArgumentParser
from argparse import Namespace
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from anemoi.utils.cli import Command


class ViewCommand(Command):
    """Browse checkpoint metadata in a pager."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")
        command_parser.add_argument(
            "--pager",
            default=os.environ.get("PAGER", "less"),
            help=("Pager command to use. Defaults to $PAGER if set, otherwise 'less'."),
        )
        command_parser.add_argument(
            "--yaml",
            action="store_true",
            help="View in YAML format (requires PyYAML).",
        )
        command_parser.add_argument(
            "--json",
            action="store_true",
            help="View in JSON format (default).",
        )

    def run(self, args: Namespace) -> None:
        """Execute the view command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from ..checkpoint import extract_metadata_dict

        ext = "yaml" if args.yaml else "json"

        def _serialise(data: dict[str, Any], fh: Any) -> None:
            if args.yaml:
                yaml.dump(data, fh, default_flow_style=False)
            else:
                json.dump(data, fh, indent=4, sort_keys=True)

        metadata = extract_metadata_dict(args.path)

        with TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, f"checkpoint.{ext}")

            with open(tmp_file, "w") as fh:
                _serialise(metadata, fh)

            subprocess.check_call([*shlex.split(args.pager), tmp_file])


command = ViewCommand
