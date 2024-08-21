import argparse

from anemoi.graphs.descriptor import GraphDescriptor
from anemoi.graphs.inspector import GraphInspector

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
