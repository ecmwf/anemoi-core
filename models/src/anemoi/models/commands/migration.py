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
from datetime import datetime
from pathlib import Path

from .. import __version__ as version_anemoi_models
from ..migrations import MIGRATION_PATH
from ..migrations import Migrator
from ..migrations import registered_migrations
from . import Command

LOGGER = logging.getLogger(__name__)


def _get_migration_name(name: str) -> str:
    name = name.lower().replace("-", "_").replace(" ", "_")
    now = int(datetime.now().timestamp())
    return f"{now}_{name}.py"


class Migration(Command):
    """Commands to interact with migrations"""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : ArgumentParser
            The argument parser to which the arguments will be added.
        """
        subparsers = command_parser.add_subparsers(required=True)
        help_create = "Create a new migration script."
        create_parser = subparsers.add_parser("create", help=help_create, description=help_create)
        create_parser.add_argument("name", help="Name of the migration")
        create_parser.add_argument("--path", type=Path, default=MIGRATION_PATH, help="Path to the migration folder")

        help_apply = "Apply migrations to a checkpoint."
        apply_parser = subparsers.add_parser("apply", help=help_apply, description=help_apply)
        apply_parser.add_argument("ckpt", help="Path to the checkpoint to migrate")
        apply_parser.add_argument(
            "--steps",
            default=None,
            type=int,
            help=(
                "Relative number of steps to execute. Positive migrates, negative rollbacks. "
                "Defaults to execute all migrations."
                "Mutually exclusive with --n_migrations and --target"
            ),
        )
        apply_parser.add_argument(
            "--n-migrations",
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
        apply_parser.add_argument(
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
        if args.subcommand == "create":
            return self.run_create(args)
        elif args.subcommand == "apply":
            return self.run_appy(args)
        raise ValueError(f"{args.subcommand} does not exist.")

    def run_create(self, args: Namespace) -> None:
        """Create a new migration

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        from textwrap import dedent

        name = _get_migration_name(args.name)
        with open(args.path / name, "w") as f:
            f.write(
                dedent(
                    f"""
                    # (C) Copyright 2024 Anemoi contributors.
                    #
                    # This software is licensed under the terms of the Apache Licence Version 2.0
                    # which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
                    #
                    # In applying this licence, ECMWF does not waive the privileges and immunities
                    # granted to it by virtue of its status as an intergovernmental organisation
                    # nor does it submit to any jurisdiction.


                    from anemoi.models.migrations import CkptType
                    from anemoi.models.migrations import MigrationMetadata


                    metadata = MigrationMetadata(
                        versions={{
                            "migration": "1.0.0",
                            "anemoi-models": "{version_anemoi_models}",
                        }}
                    )


                    def migrate(ckpt: CkptType) -> CkptType:
                        \"\"\"Migrate the checkpoint\"\"\"
                        print(ckpt)
                        return ckpt


                    def rollback(ckpt: CkptType) -> CkptType:
                        \"\"\"Rollback the migration\"\"\"
                        return ckpt
                """
                ).strip()
                + "\n"
            )
        print(f"Created migration {name} in {args.path}")

    def run_appy(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        import torch

        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        new_ckpt, done_migrations, done_rollbacks = Migrator().sync(
            ckpt, steps=args.steps, n_migrations=args.n_migrations, target=args.target
        )
        if len(done_migrations) or len(done_rollbacks):
            version = len(registered_migrations(ckpt))
            ckpt_path = Path(args.ckpt)
            new_path = ckpt_path.with_stem(f"{ckpt_path.stem}-v{version}")
            torch.save(ckpt, new_path)
            LOGGER.info("Saved previous checkpoint here: %s", str(new_path.resolve()))
            torch.save(new_ckpt, ckpt_path)
        if len(done_migrations):
            LOGGER.info(
                "Executed %s migrations: %s", len(done_migrations), [migration.name for migration in done_migrations]
            )
        if len(done_rollbacks):
            LOGGER.info(
                "Executed %s migration rollbacks: %s",
                len(done_rollbacks),
                [migration.name for migration in done_rollbacks],
            )


command = Migration
