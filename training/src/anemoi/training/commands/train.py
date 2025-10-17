# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import logging
import os
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path

from anemoi.training.commands import Command

LOGGER = logging.getLogger(__name__)


class TrainBase(Command, ABC):
    accept_unknown_args = True

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def _merge_sysargv(self, args: argparse.Namespace) -> str:
        """Merge the sys.argv with the known subcommands to pass to hydra.

        This is done for interactive DDP, which will spawn the rank > 0 processes from sys.argv[0]
        and for hydra, which ingests sys.argv[1:]

        Parameters
        ----------
        args : argparse.Namespace
            args from the command line

        Returns
        -------
        str
            Modified sys.argv as string
        """
        argv = Path(sys.argv[0])

        # this will turn "/env/bin/anemoi-training train" into "/env/bin/.anemoi-training-train"
        # the dot at the beginning is intentional to not interfere with autocomplete
        modified_sysargv = argv.with_name(f".{argv.name}-{args.command}")

        if hasattr(args, "subcommand"):
            modified_sysargv += f"-{args.subcommand}"
        return str(modified_sysargv)

    def prepare_sysargv(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None:
        os.environ["ANEMOI_TRAINING_CMD"] = f"{sys.argv[0]} {args.command}"
        # Merge the known subcommands with a non-whitespace character for hydra
        new_sysargv = self._merge_sysargv(args)

        # Add the unknown arguments (belonging to hydra) to sys.argv
        if unknown_args is not None:
            sys.argv = [new_sysargv, *unknown_args]
        else:
            sys.argv = [new_sysargv]

    @abstractmethod
    def run(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None: ...


class Train(TrainBase):
    """Commands to train Anemoi models."""

    def run(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None:
        # This will be picked up by the logger
        self.prepare_sysargv(args, unknown_args)

        LOGGER.info("Running anemoi training command with overrides: %s", sys.argv[1:])
        main()


def main() -> None:
    # Use the environment variable to check if main is being called from the subcommand, not from the ddp entrypoint
    if not os.environ.get("ANEMOI_TRAINING_CMD"):
        error = "This entrypoint should not be called directly. Use `anemoi-training train` instead."
        raise RuntimeError(error)

    from anemoi.training.train.train import main as anemoi_train

    anemoi_train()


command = Train
