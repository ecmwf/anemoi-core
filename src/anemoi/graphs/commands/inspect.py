# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse

from anemoi.graphs.describe import GraphDescriptor
from anemoi.graphs.inspect import GraphInspector

from . import Command


class Inspect(Command):
    """Inspect a graph."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "--show_attribute_distributions",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Hide distribution plots of edge/node attributes.",
        )
        command_parser.add_argument(
            "--show_nodes",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Show the nodes of the graph.",
        )
        command_parser.add_argument(
            "--description",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Hide the description of the graph.",
        )
        command_parser.add_argument("path", help="Path to the graph (a .PT file).")
        command_parser.add_argument("output_path", help="Path to store the inspection results.")

    def run(self, args):
        kwargs = vars(args)

        if kwargs.get("description", False):
            GraphDescriptor(kwargs["path"]).describe()

        inspector = GraphInspector(**kwargs)
        inspector.inspect()


command = Inspect
