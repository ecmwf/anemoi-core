# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CLI command to show a human-readable summary of checkpoint metadata."""

import json
from argparse import ArgumentParser
from argparse import Namespace

from anemoi.utils.cli import Command

_PREVIEW_LIMIT = 10


class InfoCommand(Command):
    """Show a summary of checkpoint metadata."""

    def add_arguments(self, command_parser: ArgumentParser) -> None:
        """Register command-line arguments.

        Parameters
        ----------
        command_parser : ArgumentParser
            Parser to which arguments are added.
        """
        command_parser.add_argument("path", help="Path to the checkpoint file.")
        command_parser.add_argument(
            "--no-migrate",
            action="store_true",
            help="Do not migrate metadata to the latest schema version.",
        )

    def run(self, args: Namespace) -> None:
        """Execute the info command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from ..interface import Metadata

        migrate = not args.no_migrate
        metadata = Metadata.from_checkpoint(args.path, migrate=migrate)
        _print_summary(metadata)


def _print_summary(metadata) -> None:
    """Print a human-readable summary of typed metadata fields.

    Displays envelope fields (schema version, creation timestamp), inference
    contract fields (datasets, task, timestep, multi-step counts), a variable
    preview, and a full indented JSON dump.

    Parameters
    ----------
    metadata : Metadata
        Loaded metadata instance to summarise.
    """
    # --- Envelope ---
    print(f"Schema version: {metadata.schema_version}")
    print(f"Created at:     {metadata.created_at}")
    print()

    # --- Inference contract ---
    print(f"Datasets:           {metadata.dataset_names}")
    print(f"Task:               {metadata.task}")
    print(f"Timestep:           {metadata.timestep}")
    print(f"Multi-step input:   {metadata.multi_step_input}")
    print(f"Multi-step output:  {metadata.multi_step_output}")
    print()

    # --- Variable preview ---
    variables = metadata.variables
    total = metadata.num_variables
    preview = variables[:_PREVIEW_LIMIT]

    print(f"Variables ({total} total, first {min(total, _PREVIEW_LIMIT)}):")
    for var in preview:
        print(f"  - {var}")
    if total > _PREVIEW_LIMIT:
        print(f"  ... and {total - _PREVIEW_LIMIT} more")
    print()

    # --- Full dump ---
    print("Full metadata:")
    print(json.dumps(metadata.to_dict(), indent=2, default=str))


command = InfoCommand
