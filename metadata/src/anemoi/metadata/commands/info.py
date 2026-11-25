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
from typing import TYPE_CHECKING

from anemoi.metadata.components.task import TaskTimingError
from anemoi.utils.cli import Command

if TYPE_CHECKING:
    from anemoi.metadata.interface import Metadata

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

    def run(self, args: Namespace) -> None:
        """Execute the info command.

        Parameters
        ----------
        args : Namespace
            Parsed command-line arguments.
        """
        from anemoi.metadata.interface import Metadata

        metadata = Metadata.from_checkpoint(args.path)
        _print_summary(metadata)


def _print_summary(metadata: "Metadata") -> None:
    """Print a human-readable summary of typed metadata fields.

    Displays the checkpoint level (version, component versions, creation
    timestamp, task and its timing), then per dataset the data kind, grid size
    and variable counts with a variable preview, and a full indented JSON dump.
    Nothing is taken from "the first dataset".

    Parameters
    ----------
    metadata : Metadata
        Loaded metadata instance to summarise.
    """
    # --- Checkpoint level ---
    print(f"Version:            {metadata.version}")
    print(f"Component versions: {metadata.component_versions}")
    print(f"Created at:         {metadata.sections.timestamp}")
    print()

    # --- Task ---
    task = metadata.task
    print(f"Task:               {task.task_name}")
    try:
        timing = task.timing
    except TaskTimingError as exc:
        print(f"Timing:             differs between datasets ({exc})")
        timings = task.timings()
    else:
        print(f"Timestep:           {timing.timestep}")
        print(f"Steps in / out:     {timing.n_step_input} / {timing.n_step_output}")
        timings = {}
    for name, dataset_timing in timings.items():
        print(
            f"  {name}: timestep {dataset_timing.timestep}, "
            f"steps in / out {dataset_timing.n_step_input} / {dataset_timing.n_step_output}",
        )
    print()

    # --- Datasets ---
    print(f"Datasets:           {metadata.dataset_names}")
    for name, view in metadata.datasets.items():
        data = view.data
        variables = data.variables()
        total = len(variables)
        print()
        print(f"[{name}]")
        print(f"  Data type:        {data.data_type()}")
        print(f"  Grid size:        {data.grid_size()}")
        print(
            f"  Variables:        {total} "
            f"({len(data.variable_indices())} input, {len(data.output_variable_indices())} output)",
        )
        for var in variables[:_PREVIEW_LIMIT]:
            print(f"    - {var}")
        if total > _PREVIEW_LIMIT:
            print(f"    ... and {total - _PREVIEW_LIMIT} more")
    print()

    # --- Full dump ---
    print("Full metadata:")
    print(json.dumps(metadata.to_dict(), indent=2, default=str))


command = InfoCommand
