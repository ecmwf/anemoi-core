# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import contextlib
import importlib.resources as pkg_resources
import logging
import os
import re
import shutil
from collections.abc import Generator
from pathlib import Path
from textwrap import dedent
from typing import Any

from hydra import compose
from hydra import initialize
from hydra import initialize_config_dir
from jinja2 import Environment
from jinja2 import select_autoescape
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel
from rich.console import Console

from anemoi.training.commands import Command
from anemoi.training.migrations.migrator import MIGRATION_PATH
from anemoi.training.migrations.migrator import ConfigMigrator
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.utils.migrations import added_migrations_compared_to_main_branch
from anemoi.utils.migrations import get_migration_name
from anemoi.utils.migrations import migrations_in_incorrect_order

LOGGER = logging.getLogger(__name__)

here = Path(__file__).parent
root_folder = here.parent.parent.parent.parent.parent


def get_migration_template() -> str:
    return dedent("""\
        {% for import in imports %}
        {{import}}
        {% endfor %}

        # DO NOT CHANGE -->
        metadata = MigrationMetadata(
            versions={
                "migration": "{{migration_version}}",
                "anemoi-training": "%NEXT_ANEMOI_TRAINING_VERSION%",
            },
            {% if final %}
            final=True,
            {% endif %}
        )
        # <-- END DO NOT CHANGE
        {% if not final %}


        def migrate(config: Config) -> Config:
            \"""Migrate the config.

            Parameters
            ----------
            config : Config
                The config object to migrated.

            Returns
            -------
            Config
                The migrated config.
            \"""
            return config
        {% endif %}
        """)


class ConfigGenerator(Command):
    """Commands to interact with training configs."""

    @staticmethod
    def add_arguments(command_parser: argparse.ArgumentParser) -> None:
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        help_msg = "Generate the Anemoi training configs."
        generate = subparsers.add_parser(
            "generate",
            help=help_msg,
            description=help_msg,
        )
        generate.add_argument("--output", "-o", default=Path.cwd(), help="Output directory")
        generate.add_argument("--overwrite", "-f", action="store_true")

        help_msg = "Validate the Anemoi training configs."
        validate = subparsers.add_parser("validate", help=help_msg, description=help_msg)

        validate.add_argument("--config-name", help="Name of the primary config file")
        validate.add_argument("--config-path", "-i", type=Path, default=None, help="Configuration directory")
        validate.add_argument("--overwrite", "-f", action="store_true")
        validate.add_argument(
            "--mask_env_vars",
            "-m",
            help="Mask environment variables from config. Default False",
            action="store_true",
        )

        help_msg = "Dump Anemoi configs to a YAML file."
        dump = subparsers.add_parser(
            "dump",
            help=help_msg,
            description=help_msg,
        )
        dump.add_argument("--config-path", "-i", default=None, type=Path, help="Configuration directory")
        dump.add_argument("--config-name", "-n", default="dev", help="Name of the configuration")
        dump.add_argument("--output", "-o", default="./config.yaml", type=Path, help="Output file path")
        dump.add_argument("--overwrite", "-f", action="store_true")
        dump.add_argument("--sort", "-s", action="store_true", help="Sort keys alphabetically in the output YAML")

        help_msg = "Export the JSON schema of the training configuration."
        subparsers.add_parser(
            "schema",
            help=help_msg,
            description=help_msg,
        )

        help_msg = "Config migration commands."
        migration = subparsers.add_parser(
            "migration",
            help=help_msg,
            description=help_msg,
        )
        migration_subcommands = migration.add_subparsers(dest="migration_subcommand", required=True)

        help_msg = "Apply all migrations to your config dump."
        migration_sync = migration_subcommands.add_parser(
            "sync",
            help=help_msg,
            description=help_msg,
        )
        migration_sync.add_argument("path", help="Path to the config dump to migrate.")
        migration_sync.add_argument(
            "--output",
            "-o",
            default="./migrated-config.yaml",
            type=Path,
            help="Path to the migrated config dump.",
        )
        migration_sync.add_argument(
            "--dry-run",
            action="store_true",
            help="Perform a dry-run, without saving the updated config.",
        )
        migration_sync.add_argument("--no-color", action="store_true", help="Disables terminal colors.")

        help_msg = "Generate a config migration script."
        migration_create = migration_subcommands.add_parser(
            "create",
            help=help_msg,
            description=help_msg,
        )
        migration_create.add_argument("name", help="Name of the migration.")
        migration_create.add_argument(
            "--final",
            "-f",
            action="store_true",
            default=False,
            help="Set this as the final migration. Older configs cannot be migrated past this.",
        )

        help_msg = "Fix the order of migrations after a git merge."
        migration_subcommands.add_parser("fix-order", help=help_msg, description=help_msg)

    def run(self, args: argparse.Namespace) -> None:

        if args.subcommand == "migration" and args.migration_subcommand == "sync":
            LOGGER.info("Migrating config %s in %s.", args.path, args.output, args.dry_run, args.no_color)
            self.migrate_config(args.path, args.output)
            return
        if args.subcommand == "migration" and args.migration_subcommand == "create":
            LOGGER.info("Creating migration script %s.", args.name)
            self.create_config_migration(args.name, args.final)
            return
        if args.subcommand == "migration" and args.migration_subcommand == "fix-order":
            self.fix_order_config_migration()
            return

        self.overwrite = args.overwrite

        if args.subcommand == "generate":
            LOGGER.info(
                "Generating configs, please wait.",
            )
            self.traverse_config(args.output)
            return

        if args.subcommand == "validate":
            LOGGER.info("Validating configs.")
            LOGGER.warning(
                "Note that this command is not taking into account if your config has set \
                    the config_validation flag to false."
                "So this command will validate the config regardless of the flag.",
            )
            self.validate_config(args.config_name, args.mask_env_vars, args.config_path)
            LOGGER.info("Config files validated.")
            return

        if args.subcommand == "dump":
            LOGGER.info("Dumping config to %s", args.output)
            self.dump_config(args.config_path, args.config_name, args.output, sort=args.sort)
            return

        if args.subcommand == "schema":
            import json

            print(json.dumps(BaseSchema.model_json_schema(), indent=2))  # noqa: T201
            return

    def traverse_config(self, destination_dir: Path | str) -> None:
        """Writes the given configuration data to the specified file path."""
        config_package = "anemoi.training.config"

        # Ensure the destination directory exists
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Traverse through the package's config directory
        with pkg_resources.as_file(pkg_resources.files(config_package)) as config_path:
            self.copy_files(config_path, destination_dir)

    @staticmethod
    def copy_file(item: Path, file_path: Path) -> None:
        """Copies the file to the destination directory."""
        try:
            shutil.copy2(item, file_path)
            LOGGER.debug("Copied %s to %s", item.name, file_path)
        except Exception:
            LOGGER.exception("Failed to copy %s", item.name)

    def copy_files(self, source_directory: Path, target_directory: Path) -> None:
        """Copies directory files to a target directory."""
        for data in source_directory.rglob("*yaml"):  # Recursively walk through all files and directories
            item = Path(data)
            if item.is_file():
                file_path = Path(target_directory, item.relative_to(source_directory))

                file_path.parent.mkdir(parents=True, exist_ok=True)

                if not file_path.exists() or self.overwrite:
                    self.copy_file(item, file_path)
                else:
                    LOGGER.info("File %s already exists, skipping", file_path)

    def _mask_slurm_env_variables(self, cfg: DictConfig) -> None:
        """Mask environment variables are set."""
        # Convert OmegaConf dict to YAML format (raw string)
        raw_cfg = OmegaConf.to_yaml(cfg)
        # To extract and replace environment variables, loop through the matches
        updated_cfg = raw_cfg
        primitive_type_hints = extract_primitive_type_hints(BaseSchema)

        patterns = [
            r"(\w+):\s*\$\{oc\.env:([A-Z0-9_]+)\}(?!\})",
            r"(\w+):\s*\$\{oc\.decode:\$\{oc\.env:([A-Z0-9_]+)\}\}",
        ]
        replaces = ["${{oc.env:{match}}}", "${{oc.decode:${{oc.env:{match}}}}}"]
        # Find all matches in the raw_cfg string
        for pattern, replace in zip(patterns, replaces, strict=False):
            matches = re.findall(pattern, raw_cfg)
            # Find the corresponding type hints for each extracted key
            for extracted_key, match in matches:
                corresponding_keys = next(iter([key for key in primitive_type_hints if extracted_key in key]))
                # Check if the environment variable exists
                env_value = os.getenv(match)

                # If environment variable doesn't exist, replace with default string
                if env_value is None:
                    def_str = "default"
                    def_int = 0
                    def_bool = True
                    if primitive_type_hints[corresponding_keys] is str:
                        env_value = def_str
                    elif primitive_type_hints[corresponding_keys] in [int, float]:
                        env_value = def_int
                    elif primitive_type_hints[corresponding_keys] is bool:
                        env_value = def_bool
                    elif primitive_type_hints[corresponding_keys] is Path:
                        env_value = Path(def_str)
                    else:
                        msg = "Type not supported for masking environment variables"
                        raise TypeError(msg)
                    LOGGER.warning("Environment variable %s not found, masking with %s", match, env_value)
                    # Replace the pattern with the actual value or the default string
                    updated_cfg = updated_cfg.replace(replace.format(match=match), str(env_value))

        return OmegaConf.create(updated_cfg)

    def validate_config(self, config_name: Path | str, mask_env_vars: bool, config_path: Path | None = None) -> None:
        """Validate a configuration, composed from ``config_path`` when given (see ``_initialize_hydra``)."""
        with _initialize_hydra(config_path):
            cfg = compose(config_name=config_name)
            if mask_env_vars:
                cfg = self._mask_slurm_env_variables(cfg)
            OmegaConf.resolve(cfg)
            BaseSchema(**cfg)

    def dump_config(self, config_path: Path | None, name: str, output: Path, sort: bool = False) -> None:
        """Dump config files in one YAML file, composed from ``config_path`` when given."""
        with _initialize_hydra(config_path):
            cfg = compose(config_name=name)

        LOGGER.info("Dumping file in %s.", output)
        with output.open("w") as f:
            f.write(OmegaConf.to_yaml(cfg, sort_keys=sort))

    def migrate_config(
        self,
        config_path: Path,
        output: Path | None,
        dry_run: bool = False,
        no_color: bool = False,
    ) -> None:
        """Migrate a config dump."""
        from difflib import unified_diff

        original_config, migrated_config, executed_migrations = ConfigMigrator().sync(config_path)
        original_conifg_lines = original_config.to_yaml().split("\n")
        migrated_config_content = migrated_config.to_yaml()
        migrated_config_lines = migrated_config_content.split("\n")

        console = Console(force_terminal=not no_color, highlight=False)
        prefix = "Executed"
        if dry_run:
            prefix = "Would execute"
        if len(executed_migrations):
            console.print(f"{prefix}", len(executed_migrations), "operation(s):")
        if not len(executed_migrations):
            console.print("Your config is already compatible :party_popper:! No missing migration to execute.")
        for migration in executed_migrations:
            console.print(f"  [green]+ MIGRATE [bold]{migration.name}[/bold][/green]")

        print("\n".join(unified_diff(original_conifg_lines, migrated_config_lines)))  # noqa: T201
        if output is not None and not dry_run:
            output.write_text(migrated_config_content)

    def create_config_migration(self, name: str, final: bool = False) -> None:
        """Create a new migration script."""
        name = get_migration_name(name)

        imports: list[str] = ["from anemoi.utils.migrations import MigrationMetadata", ""]
        if not final:
            imports.append("from anemoi.training.migrations.config import Config")
        template = Environment(trim_blocks=True, lstrip_blocks=True, autoescape=select_autoescape()).from_string(
            get_migration_template(),
        )

        migration_path = MIGRATION_PATH / name

        MIGRATION_PATH.mkdir(exist_ok=True, parents=True)

        migration_path.write_text(
            template.render(
                {
                    "migration_version": "1.0.0",
                    "imports": imports,
                    "final": final,
                },
            ),
        )

        print(f"Created migration {MIGRATION_PATH}/{name}")  # noqa: T201

    def fix_order_config_migration(self) -> None:
        """Fixes the order of the new migration scripts.

        It uses the earliest possible time with the last migration name in origin/main.
        """
        new_migrations = added_migrations_compared_to_main_branch(root_folder, MIGRATION_PATH)
        all_migrations = sorted(
            [file.name for file in MIGRATION_PATH.iterdir() if file.is_file() and file.name != "__init__.py"],
        )
        incorrect_order, last_upstream_name = migrations_in_incorrect_order(all_migrations, new_migrations)

        if last_upstream_name is None or not len(incorrect_order):
            print("No migration to rename.")  # noqa: T201
            return

        new_timestamp = int(last_upstream_name.partition("_")[0]) + 1
        for k, name in enumerate(new_migrations):
            path = MIGRATION_PATH / name
            _, _, new_name = name.partition("_")
            new_name = f"{new_timestamp + k}_{new_name}"
            print(f"Renaming {name} to {new_name}.")  # noqa: T201
            path.rename(path.with_name(new_name))


@contextlib.contextmanager
def _initialize_hydra(config_path: Path | None) -> Generator[None, None, None]:
    """Initialise Hydra for config composition, shared by ``validate`` and ``dump``.

    With ``config_path`` the configs are composed from that directory, installed as Hydra's
    primary (``main``) search-path entry so it takes precedence over the current working
    directory and the packaged defaults — the same model ``@hydra.main`` gives ``train``'s
    ``--config-path``. Without it, discovery is left to the ``AnemoiSearchPathPlugin`` (the
    current working directory and the packaged defaults). ``initialize_config_dir`` requires
    an absolute path.
    """
    if config_path is not None:
        with initialize_config_dir(version_base=None, config_dir=str(Path(config_path).absolute())):
            yield
    else:
        with initialize(version_base=None):
            yield


def extract_primitive_type_hints(model: type[BaseModel], prefix: str = "") -> dict[str, Any]:
    field_types = {}

    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        full_field_name = f"{prefix}.{field_name}" if prefix else field_name

        # Check if the field type has 'model_fields' (indicating a nested Pydantic model)
        if hasattr(field_type, "model_fields"):
            field_types.update(extract_primitive_type_hints(field_type, full_field_name))
        else:
            try:
                field_types[full_field_name] = field_type.__args__[0]
            except AttributeError:
                field_types[full_field_name] = field_type

    return field_types


command = ConfigGenerator
