# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to edit checkpoint metadata in an external editor."""

import json
import logging
import os
import shlex
import subprocess
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from anemoi.utils.cli import Command

LOG = logging.getLogger(__name__)

# Some editors require extra flags to block until the file is closed.
_EDITOR_WAIT_FLAGS: dict[str, list[str]] = {"code": ["--wait"]}


class EditCommand(Command):
    """Edit checkpoint metadata in an editor."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")
        command_parser.add_argument(
            "--editor",
            default=os.environ.get("EDITOR", "vi"),
            help=("Editor command to use. Defaults to $EDITOR if set, otherwise 'vi'."),
        )
        command_parser.add_argument(
            "--yaml",
            action="store_true",
            help="Edit in YAML format (requires PyYAML).",
        )
        command_parser.add_argument(
            "--json",
            action="store_true",
            help="Edit in JSON format (default).",
        )

    def run(self, args: Namespace) -> None:
        """Execute the edit command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from ..checkpoint import extract_metadata_dict
        from ..checkpoint import replace_metadata

        use_yaml = args.yaml
        ext = "yaml" if use_yaml else "json"

        def _serialise(data: dict[str, Any], fh: Any) -> None:
            if use_yaml:
                yaml.dump(data, fh, default_flow_style=False)
            else:
                json.dump(data, fh, indent=4, sort_keys=True)

        def _deserialise(fh: Any) -> dict[str, Any]:
            if use_yaml:
                return yaml.safe_load(fh)
            return json.load(fh)

        metadata = extract_metadata_dict(args.path)

        with TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, f"checkpoint.{ext}")

            with open(tmp_file, "w") as fh:
                _serialise(metadata, fh)

            cmd_parts = shlex.split(args.editor)
            extra_flags = _EDITOR_WAIT_FLAGS.get(os.path.basename(cmd_parts[0]), [])
            subprocess.check_call([*cmd_parts, *extra_flags, tmp_file])

            with open(tmp_file) as fh:
                edited = _deserialise(fh)

            if edited != metadata:
                replace_metadata(Path(args.path), edited)
                LOG.info("Metadata updated in %s", args.path)
            else:
                LOG.info("No changes detected; checkpoint left unmodified.")


command = EditCommand
