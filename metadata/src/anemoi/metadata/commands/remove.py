# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to remove metadata from a checkpoint."""

import logging
import shutil
from argparse import ArgumentParser
from argparse import Namespace

from anemoi.utils.cli import Command

LOG = logging.getLogger(__name__)


class RemoveCommand(Command):
    """Remove metadata from a checkpoint."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")

        target_group = command_parser.add_mutually_exclusive_group(required=True)
        target_group.add_argument(
            "--inplace",
            action="store_true",
            help="Remove metadata from the source checkpoint in-place.",
        )
        target_group.add_argument(
            "--output",
            metavar="FILE",
            help="Write the cleaned checkpoint to FILE instead of modifying in-place.",
        )

    def run(self, args: Namespace) -> None:
        """Execute the remove command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from ..checkpoint import remove_metadata

        if args.inplace:
            target = args.path
        else:
            shutil.copy2(args.path, args.output)
            target = args.output

        LOG.info("Removing metadata from %s", target)
        remove_metadata(target)
        LOG.info("Done.")


command = RemoveCommand
