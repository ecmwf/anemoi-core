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

from rich import print as rprint

from .. import __version__ as version_anemoi_models
from ..migrations import MIGRATION_PATH
from ..migrations import IncompatibleCheckpointException
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
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)
        help_create = "Create a new migration script."
        create_parser = subparsers.add_parser("create", help=help_create, description=help_create)
        create_parser.add_argument("name", help="Name of the migration.")
        create_parser.add_argument("--path", type=Path, default=MIGRATION_PATH, help="Path to the migration folder.")
        create_parser.add_argument(
            "--final",
            action="store_true",
            default=False,
            help="Set this as the final migration. Older checkpoints cannot be migrated past this.",
        )

        help_sync = "Apply migrations to a checkpoint."
        sync_parser = subparsers.add_parser("sync", help=help_sync, description=help_sync)
        sync_parser.add_argument("ckpt", help="Path to the checkpoint to migrate.")
        sync_parser.add_argument(
            "--steps",
            default=None,
            type=int,
            help=(
                "Relative number of steps to execute. Positive migrates, negative rollbacks. "
                "Defaults to execute all migrations."
            ),
        )

        help_inspect = "Inspect migrations in a checkpoint."
        inspect_parser = subparsers.add_parser("inspect", help=help_inspect, description=help_inspect)
        inspect_parser.add_argument("ckpt", help="Path to the checkpoint to inspect.")

    def run(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        if args.subcommand == "create":
            return self.run_create(args)
        elif args.subcommand == "sync":
            return self.run_sync(args)
        elif args.subcommand == "inspect":
            return self.run_inspect(args)
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

        imports_items = ["from anemoi.models.migrations import CkptType"]
        if args.final:
            imports_items.append("from anemoi.models.migrations import IncompatibleCheckpointException")
        imports_items.append("from anemoi.models.migrations import MigrationMetadata")
        imports = "\n".join(imports_items)

        content = "return ckpt"
        if args.final:
            content = "raise IncompatibleCheckpointException"

        with open(args.path / name, "w") as f:
            f.write(
                dedent(
                    """\
                    # (C) Copyright 2024 Anemoi contributors.
                    #
                    # This software is licensed under the terms of the Apache Licence Version 2.0
                    # which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
                    #
                    # In applying this licence, ECMWF does not waive the privileges and immunities
                    # granted to it by virtue of its status as an intergovernmental organisation
                    # nor does it submit to any jurisdiction.\n
                    """
                )
            )
            f.write(imports)

            f.write(
                dedent(
                    f"""

                    metadata = MigrationMetadata(
                        versions={{
                            "migration": "1.0.0",
                            "anemoi-models": "{version_anemoi_models}",
                        }}"""
                )
            )
            if args.final:
                f.write(f",\n    final={args.final},")
            f.write(
                dedent(
                    f"""
                    )


                    def migrate(ckpt: CkptType) -> CkptType:
                        \"\"\"Migrate the checkpoint.\"\"\"
                        {content}


                    def rollback(ckpt: CkptType) -> CkptType:
                        \"\"\"Rollback the migration.\"\"\"
                        {content}
                """
                )
            )
        print(f"Created migration {args.path}/{name}")

    def run_sync(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        import torch

        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        try:
            new_ckpt, done_migrations, done_rollbacks = Migrator().sync(ckpt, steps=args.steps)
            if len(done_migrations) or len(done_rollbacks):
                version = len(registered_migrations(ckpt))
                ckpt_path = Path(args.ckpt)
                new_path = ckpt_path.with_stem(f"{ckpt_path.stem}-v{version}")
                torch.save(ckpt, new_path)
                print("Saved backed-up checkpoint here:", str(new_path.resolve()))
                torch.save(new_ckpt, ckpt_path)
            if len(done_migrations):
                print(f"Executed {len(done_migrations)} migration(s):")
            for migration in done_migrations:
                rprint(f"[green]+ [bold]{migration}[/bold][/green]")
            if len(done_rollbacks):
                print(f"Executed {len(done_rollbacks)} rollback(s):")
            for migration in done_rollbacks:
                rprint(f"[red]- [bold]{migration}[/red]")
        except IncompatibleCheckpointException as e:
            LOGGER.error(str(e))

    def run_inspect(self, args: Namespace) -> None:
        """Inspects the checkpoint.
        It will show:
        * the migrations already registered in the checkpoint
        * the missing migrations to execute
        * the extra migrations to rollback

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        import torch

        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        migrator = Migrator()
        try:
            executed_migrations, missing_migrations, extra_migrations = migrator.inspect(ckpt)
            if len(executed_migrations):
                print("Registered migrations:")
            for migration in executed_migrations:
                rprint(
                    f"[cyan]* [bold]{migration.name} \\[{migration.metadata.versions['anemoi-models']}][/bold][/cyan]"
                )
            if len(missing_migrations):
                print("Missing migrations:")
            else:
                print("No missing migration to execute.")
            for migration in missing_migrations:
                rprint(
                    f"[green]+ [bold]{migration.name} \\[{migration.metadata.versions['anemoi-models']}][/bold][/green]"
                )
            if len(extra_migrations):
                print("Extra migrations to rollback:")
            for migration in extra_migrations:
                rprint(f"[red]- [bold]{migration}[/red]")
        except IncompatibleCheckpointException:
            print("No compatible migrations available. (Checkpoint too old).")


command = Migration
