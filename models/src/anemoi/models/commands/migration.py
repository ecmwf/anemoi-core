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

from rich.console import Console

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


def maybe_plural(count: int, text: str) -> str:
    if count >= 2:
        return text + "s"
    return text


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
        create_parser.add_argument(
            "--no-rollback",
            action="store_true",
            default=False,
            help="Set this if you do not plan to support rollbacking.",
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
        sync_parser.add_argument("--no-color", action="store_true", help="Disables terminal colors.")

        help_inspect = "Inspect migrations in a checkpoint."
        inspect_parser = subparsers.add_parser("inspect", help=help_inspect, description=help_inspect)
        inspect_parser.add_argument("ckpt", help="Path to the checkpoint to inspect.")
        inspect_parser.add_argument("--no-color", action="store_true", help="Disables terminal colors.")

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

        imports_items: list[str] = []
        if not args.final:
            imports_items.append("from anemoi.models.migrations import CkptType")
        imports_items.append("from anemoi.models.migrations import MigrationMetadata")
        imports = "\n".join(imports_items)

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
                    """
                    )
                """
                )
            )

            if not args.final:
                f.write(
                    dedent(
                        """

                        def migrate(ckpt: CkptType) -> CkptType:
                            \"\"\"Migrate the checkpoint.\"\"\"
                            return ckpt
                    """
                    )
                )
            if not args.no_rollback and not args.final:
                f.write(
                    dedent(
                        """

                        def rollback(ckpt: CkptType) -> CkptType:
                            \"\"\"Rollback the migration.\"\"\"
                            return ckpt
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
        console = Console(force_terminal=not args.no_color)
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
                console.print(f"[green]+ [bold]{migration}[/bold][/green]")
            if len(done_rollbacks):
                print(f"Executed {len(done_rollbacks)} rollback(s):")
            for migration in done_rollbacks:
                console.print(f"[red]- [bold]{migration}[/red]")
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
        console = Console(force_terminal=not args.no_color, highlight=False)
        try:
            executed_migrations, missing_migrations, extra_migrations = migrator.inspect(ckpt)
            if len(executed_migrations):
                print(
                    len(executed_migrations),
                    " registered ",
                    maybe_plural(len(executed_migrations), "migration"),
                    ":",
                    sep="",
                )
            for migration in executed_migrations:
                console.print(
                    f"  [cyan]* [bold]{migration.name}[/bold] \\[v{migration.metadata.versions['anemoi-models']}][/cyan]"
                )
            if len(executed_migrations):
                console.print("  [italic]These migrations are already executed and part of the checkpoint[/italic]")
            if len(missing_migrations):
                print(
                    len(missing_migrations),
                    " missing ",
                    maybe_plural(len(missing_migrations), "migration"),
                    ":",
                    sep="",
                )
            else:
                console.print("Your checkpoint is already compatible :party_popper:! No missing migration to execute.")
            for migration in missing_migrations:
                console.print(
                    f"  [green]+ [bold]{migration.name}[/bold] \\[v{migration.metadata.versions['anemoi-models']}][/green]"
                )
            if len(extra_migrations):
                print(
                    len(executed_migrations),
                    "extra",
                    maybe_plural(len(executed_migrations), "migration"),
                    "to rollback:",
                )
                print("Extra migrations to rollback:")
            for migration in extra_migrations:
                console.print(f"  [red]- [bold]{migration}[/red]")
            if len(missing_migrations) or len(extra_migrations):
                console.print("[italic]To update your checkpoint, run:[/italic]")
                console.print(f"[italic]anemoi-models migration sync {args.ckpt}[/italic]")
        except IncompatibleCheckpointException:
            print("No compatible migration available: the checkpoint is too old.")


command = Migration
