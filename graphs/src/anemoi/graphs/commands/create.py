# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging
from pathlib import Path

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.describe import GraphDescriptor

from . import Command

LOGGER = logging.getLogger(__name__)


class Create(Command):
    """Create a graph."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files. This will delete the target graph if it already exists.",
        )
        command_parser.add_argument(
            "--description",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Show the description of the graph.",
        )
        command_parser.add_argument(
            "config",
            type=Path,
            help="Configuration yaml file path defining the recipe to create the graph.",
        )
        command_parser.add_argument("save_path", type=Path, help="Path to store the created graph.")

    def run(self, args):
        if args.save_path.exists() and not args.overwrite:
            LOGGER.warning(
                "Graph already exists at %s. Skipping creation. Use --overwrite to regenerate.",
                args.save_path,
            )
            if args.description:
                GraphDescriptor(args.save_path).describe()
            return

        graph_creator = GraphCreator(config=args.config)
        graph_creator.create(save_path=args.save_path, overwrite=args.overwrite)

        if args.description:
            if args.save_path.exists():
                GraphDescriptor(args.save_path).describe()
            else:
                print("Graph description is not shown if the graph is not saved.")


command = Create
