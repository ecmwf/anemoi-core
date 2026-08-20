# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to list supporting arrays stored in a checkpoint."""

from argparse import ArgumentParser
from argparse import Namespace
from typing import Any
from typing import cast

from anemoi.utils.cli import Command


class ArraysCommand(Command):
    """List supporting arrays in a checkpoint."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")

    def run(self, args: Namespace) -> None:
        """Execute the arrays command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from ..checkpoint import load_metadata

        result = load_metadata(args.path, supporting_arrays=True)
        _, arrays = cast(tuple[Any, dict[str, Any]], result)

        if not arrays:
            print("No supporting arrays found in checkpoint.")
            return

        for name, array in sorted(arrays.items()):
            print(f"{name}: shape={array.shape} dtype={array.dtype}")


command = ArraysCommand
