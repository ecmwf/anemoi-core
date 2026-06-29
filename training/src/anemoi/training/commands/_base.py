# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import os
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path

from anemoi.training.commands import Command


class HydraCommand(Command, ABC):
    """Base class for commands that delegate to a Hydra entrypoint.

    Subclasses must set `env_var` to the environment variable used to guard the entrypoint.
    """

    accept_unknown_args = True
    env_var: str  # e.g. "ANEMOI_TRAINING_CMD"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def _merge_sysargv(self, args: argparse.Namespace) -> str:
        """Merge the sys.argv with the known subcommands to pass to hydra.

        This is done for interactive DDP, which will spawn the rank > 0 processes from sys.argv[0]
        and for hydra, which ingests sys.argv[1:]
        """
        argv = Path(sys.argv[0])

        # turns "/env/bin/anemoi-training <cmd>" into "/env/bin/.anemoi-training-<cmd>"
        # the dot at the beginning is intentional to not interfere with autocomplete
        modified_sysargv = str(argv.with_name(f".{argv.name}-{args.command}"))

        if hasattr(args, "subcommand"):
            modified_sysargv += f"-{args.subcommand}"
        return modified_sysargv

    def prepare_sysargv(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None:
        os.environ[self.env_var] = f"{sys.argv[0]} {args.command}"
        new_sysargv = self._merge_sysargv(args)

        if unknown_args is not None:
            sys.argv = [new_sysargv, *unknown_args]
        else:
            sys.argv = [new_sysargv]

    @abstractmethod
    def run(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None: ...
