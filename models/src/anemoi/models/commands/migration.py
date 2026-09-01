# (C) Copyright 2025-2026 Anemoi contributors.
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
from shutil import copy2
from textwrap import dedent

from jinja2 import Environment
from rich.console import Console

from anemoi.utils.migrations import added_migrations_compared_to_main_branch
from anemoi.utils.migrations import get_migration_name
from anemoi.utils.migrations import migrations_in_incorrect_order

from ..migrations import MIGRATION_PATH
from ..migrations import CkptMigrator
from ..migrations import IncompatibleCheckpointException
from ..migrations.migrator import LOGGER as migrator_logger
from . import Command

here = Path(__file__).parent
root_folder = here.parent.parent.parent.parent.parent


def maybe_plural(count: int, text: str) -> str:
    if count >= 2:
        return text + "s"
    return text


def get_migration_template() -> str:
    return dedent("""\
        {% for import in imports %}
        {{import}}
        {% endfor %}

        # DO NOT CHANGE -->
        metadata = MigrationMetadata(
            versions={
                "migration": "{{migration_version}}",
                "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
            },
            {% if final %}
            final=True,
            {% endif %}
        )
        # <-- END DO NOT CHANGE
        {% if not final %}


        {% if with_setup %}
        def migrate_setup(context: MigrationContext) -> None:
            \"""Migrate setup callback to be run before loading the checkpoint.

            Parameters
            ----------
            context : MigrationContext
               A MigrationContext instance
            \"""


        {% endif %}
        def migrate(ckpt: CkptType) -> CkptType:
            \"""Migrate the checkpoint.


            Parameters
            ----------
            ckpt : CkptType
                The checkpoint dict.

            Returns
            -------
            CkptType
                The migrated checkpoint dict.
            \"""
            return ckpt
        {% endif %}
        """)


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
        create_parser.add_argument(
            "--final",
            "-f",
            action="store_true",
            default=False,
            help="Set this as the final migration. Older checkpoints cannot be migrated past this.",
        )
        create_parser.add_argument(
            "--with-setup",
            "-s",
            action="store_true",
            default=False,
            help="Set this if need the migrate_setup and rollback_setup callback.",
        )

        help_sync = "Apply migrations to a checkpoint."
        sync_parser = subparsers.add_parser("sync", help=help_sync, description=help_sync)
        sync_parser.add_argument("ckpt", help="Path to the checkpoint to migrate.")
        sync_parser.add_argument(
            "--dry-run",
            default=False,
            action="store_true",
            help="Perform a dry-run, without saving the updated checkpoint.",
        )
        sync_parser.add_argument("--no-color", action="store_true", help="Disables terminal colors.")
        sync_parser.add_argument(
            "--log-level",
            default="NOTSET",
            choices=logging.getLevelNamesMapping(),
            help="Log level",
        )

        help_inspect = "Inspect migrations in a checkpoint."
        inspect_parser = subparsers.add_parser("inspect", help=help_inspect, description=help_inspect)
        inspect_parser.add_argument("ckpt", help="Path to the checkpoint to inspect.")
        inspect_parser.add_argument("--no-color", action="store_true", help="Disables terminal colors.")

        help_fix_order = "Fix the order of migrations after a git merge."
        subparsers.add_parser("fix-order", help=help_fix_order, description=help_fix_order)

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
        elif args.subcommand == "fix-order":
            return self.run_fix_order()
        raise ValueError(f"{args.subcommand} does not exist.")

    def run_create(self, args: Namespace) -> None:
        """Create a new migration.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """

        if args.final and args.with_setup:
            raise ValueError("Final migration cannot have setup callbacks.")

        name = get_migration_name(args.name)

        imports: list[str] = []
        if not args.final:
            imports.append("from anemoi.models.migrations import CkptType")
        if args.with_setup:
            imports.append("from anemoi.models.migrations import MigrationContext")
        imports.append("from anemoi.models.migrations import MigrationMetadata")
        template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(get_migration_template())

        with open(MIGRATION_PATH / name, "w") as f:
            f.write(
                template.render(
                    {
                        "migration_version": "1.0.0",
                        "imports": imports,
                        "final": args.final,
                        "with_setup": args.with_setup,
                    }
                )
            )

        print(f"Created migration {MIGRATION_PATH}/{name}")

    def run_sync(self, args: Namespace) -> None:
        """Execute the command with the provided arguments.

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        import torch

        migrator_logger.setLevel(args.log_level)

        console = Console(force_terminal=not args.no_color, highlight=False)
        migrator = CkptMigrator()
        ckpt_path = Path(args.ckpt)
        try:
            old_ckpt, new_ckpt, executed_migrations = migrator.sync(ckpt_path)
            if len(executed_migrations) and not args.dry_run:
                registered_migrations = migrator.registered_migrations(old_ckpt)
                version = ""
                if len(registered_migrations):
                    version = registered_migrations[-1].metadata.versions["anemoi-models"] + "-"
                version += f"{len(registered_migrations)}"

                new_path = ckpt_path.with_stem(f"{ckpt_path.stem}-v{version}")
                copy2(ckpt_path, new_path)
                print("Saved backed-up checkpoint here:", str(new_path.resolve()))
                torch.save(new_ckpt, ckpt_path)
                print(
                    "Executed ",
                    len(executed_migrations),
                    " ",
                    maybe_plural(len(executed_migrations), "operation"),
                    ":",
                    sep="",
                )
            if len(executed_migrations) and args.dry_run:
                print(
                    "Would execute ",
                    len(executed_migrations),
                    " ",
                    maybe_plural(len(executed_migrations), "operation"),
                    ":",
                    sep="",
                )
            if not len(executed_migrations):
                console.print("Your checkpoint is already compatible :party_popper:! No missing migration to execute.")
            for migration in executed_migrations:
                console.print(
                    f"  [green]+ MIGRATE [bold]{migration.name}[/bold] \\[v{migration.metadata.versions['anemoi-models']}][/green]"
                )
        except IncompatibleCheckpointException as e:
            print(str(e))

    def run_inspect(self, args: Namespace) -> None:
        """Inspects the checkpoint.

        It will show:
        * the migrations already registered in the checkpoint
        * the missing migrations to execute

        Parameters
        ----------
        args : Namespace
            The arguments passed to the command.
        """
        migrator = CkptMigrator()
        console = Console(force_terminal=not args.no_color, highlight=False)
        try:
            executed_migrations, missing_migrations = migrator.inspect(args.ckpt)
            if not len(missing_migrations):
                console.print("Your checkpoint is already compatible :party_popper:! No missing migration to execute.")
            if len(executed_migrations):
                print(
                    len(executed_migrations),
                    " registered ",
                    maybe_plural(len(executed_migrations), "migration"),
                    ":",
                    sep="",
                )
                console.print("  [italic]These migrations are already executed and part of the checkpoint[/italic]")
            for migration in executed_migrations:
                console.print(
                    f"  [cyan]* [bold]{migration.name}[/bold] \\[v{migration.metadata.versions['anemoi-models']}][/cyan]"
                )
            if len(missing_migrations):
                print(
                    len(missing_migrations),
                    " missing ",
                    maybe_plural(len(missing_migrations), "migration"),
                    ":",
                    sep="",
                )
            for migration in missing_migrations:
                console.print(
                    f"  [green]+ [bold]{migration.name}[/bold] \\[v{migration.metadata.versions['anemoi-models']}][/green]"
                )
            if len(missing_migrations):
                console.print("\n[italic]To update your checkpoint, run:[/italic]")
                console.print(f"  [italic]anemoi-models migration sync {args.ckpt}[/italic]")
        except IncompatibleCheckpointException as e:
            print(str(e))

    def run_fix_order(self) -> None:
        """Fixes the order of the new migration scripts.

        It uses the earliest possible time with the last migration name in origin/main.
        """
        new_migrations = added_migrations_compared_to_main_branch(root_folder, MIGRATION_PATH)
        all_migrations = sorted(
            [file.name for file in MIGRATION_PATH.iterdir() if file.is_file() and file.name != "__init__.py"]
        )
        incorrect_order, last_upstream_name = migrations_in_incorrect_order(all_migrations, new_migrations)

        if last_upstream_name is None or not len(incorrect_order):
            print("No migration to rename.")
            return

        new_timestamp = int(last_upstream_name.partition("_")[0]) + 1
        for k, name in enumerate(new_migrations):
            path = MIGRATION_PATH / name
            _, _, new_name = name.partition("_")
            new_name = f"{new_timestamp + k}_{new_name}"
            print(f"Renaming {name} to {new_name}.")
            path.rename(path.with_name(new_name))


command = Migration
