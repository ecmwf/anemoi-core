# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import torch

from ..migrations import Migrator
from ..migrations import registered_migrations
from . import Command

LOGGER = logging.getLogger(__name__)


class Migrate(Command):
    """Migrate a checkpoint"""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("ckpt", help="Path to the checkpoint to migrate")
        command_parser.add_argument(
            "--steps",
            default=None,
            type=int,
            help=(
                "Relative number of steps to execute. Positive migrates, negative rollbacks. "
                "Defaults to execute all migrations."
                "Mutually exclusive with --n_migrations and --target"
            ),
        )
        command_parser.add_argument(
            "--n_migrations",
            default=None,
            type=int,
            help=(
                "Absolute number of migrations to be executed. "
                "Will migrate or rollback to have exactly this number of migrations executed. "
                "Cannot be negative."
                "Defaults to all migrations. "
                "Mutually exclusive with --steps and --target"
            ),
        )
        command_parser.add_argument(
            "--target",
            default=None,
            type=str,
            help=(
                "Target version of anemoi-models. Will migrate or rollback accordingly. "
                "Defaults to latest version. "
                "Mutually exclusive with --steps and --n_migrations."
            ),
        )

    def run(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        new_ckpt, done_migrations, done_rollbacks = Migrator().sync(
            ckpt, steps=args.steps, n_migrations=args.n_migrations, target=args.target
        )
        if len(done_migrations):
            version = len(registered_migrations(ckpt))
            ckpt_path = Path(args.ckpt)
            new_path = ckpt_path.with_stem(f"{ckpt_path.stem}-v{version}")
            torch.save(ckpt, new_path)
            LOGGER.info("Saved previous checkpoint here: %s", str(new_path.resolve()))
            torch.save(new_ckpt, ckpt_path)
        if len(done_migrations):
            LOGGER.info("Executed %s migrations: %s", len(done_migrations), done_migrations)
        if len(done_rollbacks):
            LOGGER.info("Executed %s migration rollbacks: %s", len(done_rollbacks), done_rollbacks)


command = Migrate
