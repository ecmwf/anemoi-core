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


def parse_str_area(area_str: str) -> tuple[float, float, float, float]:
    # Remove parentheses if present, then split by comma
    area_str = area_str.strip("()")
    parts = area_str.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Area must be four comma-separated values: north,west,south,east")
    try:
        return [float(x) for x in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("All area values must be floats.")


class Inspect(Command):
    """Inspect a graph.

    Example
    -------
    To save all visualizations in a folder, use the following command:
    ```
    anemoi-graphs inspect graph.pt output_dir/
    ```
    For high-resolution graphs, you can specify an area to crop the graph using the `--area` option:
    ```
    anemoi-graphs inspect graph.pt output_dir/ --area=40,10,20,30
    ```
    """

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
            "--area",
            type=parse_str_area,
            default=None,
            help="Area of interest to crop the nodes, (north, west, south, east).",
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
