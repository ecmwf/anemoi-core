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
from pathlib import Path
from textwrap import dedent

from .. import __version__ as version_anemoi_models
from ..migrations import MIGRATION_PATH
from . import Command


def _get_migration_name(name: str) -> str:
    name = name.lower().replace("-", "_").replace(" ", "_")
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
        command_parser.add_argument("--path", type=Path, default=MIGRATION_PATH, help="Path to the migration folder")

    def run(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        name = _get_migration_name(args.name)
        with open(args.path / name, "w") as f:
            f.write(
                dedent(
                    f"""
                    from anemoi.models.migrations import CkptType
                    from anemoi.models.migrations import Versions

                    versions: Versions = {{
                        "migration": "1.0.0",
                        "anemoi-models": "{version_anemoi_models}",
                    }}


                    def upgrade(ckpt: CkptType) -> CkptType:
                        \"\"\"Migrate the model\"\"\"
                        print(ckpt)
                        return ckpt


                    def downgrade(ckpt: CkptType) -> CkptType:
                        \"\"\"Cancels the upgrade function\"\"\"
                        return ckpt
                """
                ).strip()
                + "\n"
            )
        print(f"Created migration {name} in {args.path}")


command = CreateMigration
