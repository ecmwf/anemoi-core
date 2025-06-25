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

import torch

from ..migrations import load_migrations
from ..migrations import migrate_ckpt
from . import Command

LOGGER = logging.getLogger(__name__)


class RunMigration(Command):
    """Migrate a checkpoint"""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("ckpt", help="Path to the checkpoint to migrate")
        command_parser.add_argument("export_path", help="Where to export the new checkpoint")

    def run(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        migrations, failed_loading = load_migrations()
        if len(failed_loading):
            LOGGER.warning("Some migrations could not be loaded: %s", ", ".join(failed_loading))
        new_ckpt, done_migrations, done_rollbacks = migrate_ckpt(
            torch.load(args.ckpt, map_location="cpu", weights_only=False), migrations
        )
        torch.save(new_ckpt, args.export_path)
        if len(done_migrations):
            LOGGER.info("Executed %s migrations: %s", len(done_migrations), done_migrations)
        if len(done_rollbacks):
            LOGGER.info("Executed %s migration rollbacks: %s", len(done_rollbacks), done_rollbacks)


command = RunMigration
