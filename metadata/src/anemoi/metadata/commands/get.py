# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to navigate metadata via a dot-separated key path."""

import json
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any

import yaml

from anemoi.utils.cli import Command


class GetCommand(Command):
    """Get a value from metadata by dot-path key."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")
        command_parser.add_argument(
            "key",
            help=(
                "Dot-separated key path into the metadata dict "
                "(e.g. 'config.model.type'). Use '.' to list top-level keys."
            ),
        )
        command_parser.add_argument(
            "--yaml",
            action="store_true",
            help="Print dict/list values in YAML format (requires PyYAML).",
        )
        command_parser.add_argument(
            "--json",
            action="store_true",
            help="Print dict/list values in JSON format (default).",
        )

    def run(self, args: Namespace) -> None:
        """Execute the get command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.

        Raises
        ------
        KeyError
            If a key segment is not found in the metadata.
        """
        from ..checkpoint import extract_metadata_dict

        metadata: Any = extract_metadata_dict(args.path)
        key_path: str = args.key

        # Special case: bare '.' lists top-level keys.
        if key_path == ".":
            print("Metadata keys (root):", list(metadata.keys()))
            return

        # Walk the path, stopping early if a trailing '.' requests a key list.
        keys = key_path.split(".")
        for i, key in enumerate(keys):
            if key == "":
                # Trailing dot — list keys at current level.
                traversed = ".".join(keys[:i])
                print(f"Metadata keys at '{traversed}':", list(metadata.keys()))
                return
            try:
                metadata = metadata[key]
            except (KeyError, TypeError) as exc:
                traversed = ".".join(keys[: i + 1])
                raise KeyError(f"Key '{traversed}' not found in metadata.") from exc

        # Print the final value.
        print(
            f"Metadata[{key_path!r}]:",
            end="\n" if isinstance(metadata, (dict, list)) else " ",
        )
        if isinstance(metadata, (dict, list)):
            if args.yaml:
                print(yaml.dump(metadata, indent=2, sort_keys=True))
            else:
                print(json.dumps(metadata, indent=2, sort_keys=True))
        else:
            print(metadata)


command = GetCommand
