# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from argparse import ArgumentParser
from argparse import Namespace
from datetime import datetime
from textwrap import dedent

from anemoi.models.migrations import MIGRATION_PATH

from . import Command


def _get_migration_name(name: str) -> str:
    name = name.replace("-", "_").replace(" ", "_")
    now = int(datetime.now().timestamp())
    return f"{now}_{name}.py"


class CreateMigration(Command):
    """Migrate a checkpoint"""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        command_parser.add_argument("name", help="Name of the migration")

    def run(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        name = _get_migration_name(args.name)
        with open(MIGRATION_PATH / name, "w") as f:
            f.write(
                dedent(
                    """
                    from anemoi.models.migrations import CkptType

                    version = "1.0.0"


                    def migrate(ckpt: CkptType) -> CkptType:
                        # Migrate the checkpoint
                        print(ckpt)
                        return ckpt
                """
                ).strip()
            )
        print(f"Created migration {name} in {MIGRATION_PATH}")


command = CreateMigration
