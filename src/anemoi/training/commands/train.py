# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from anemoi.training.commands import Command

if TYPE_CHECKING:
    import argparse

LOGGER = logging.getLogger(__name__)


class Train(Command):
    """Commands to train Anemoi models."""

    accept_unknown_args = True

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def run(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None:

        # Merge the known subcommands with a non-whitespace character for hydra
        new_sysargv = self._merge_sysargv(args)

        # Add the unknown arguments (belonging to hydra) to sys.argv
        if unknown_args is not None:
            sys.argv = [new_sysargv, *unknown_args]
        else:
            sys.argv = [new_sysargv]

        # Import and run the training command
        LOGGER.info("Running anemoi training command with overrides: %s", sys.argv[1:])
        from anemoi.training.train.train import main as anemoi_train

        anemoi_train()

    def _merge_sysargv(self, args: argparse.Namespace) -> str:
        """Merge the sys.argv with the known subcommands to pass to hydra.

        Parameters
        ----------
        args : argparse.Namespace
            args from the command line

        Returns
        -------
        str
            Modified sys.argv as string
        """
        modified_sysargv = f"{sys.argv[0]} {args.command}"
        if hasattr(args, "subcommand"):
            modified_sysargv += f" {args.subcommand}"
        return modified_sysargv


command = Train
