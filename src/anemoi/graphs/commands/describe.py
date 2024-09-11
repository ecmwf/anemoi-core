from anemoi.graphs.descriptor import GraphDescriptor

from . import Command


class Describe(Command):
    """Describe a graph."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument("graph_file", help="Path to the graph (a .PT file).")

    def run(self, args):
        kwargs = vars(args)

        GraphDescriptor(kwargs["graph_file"]).describe()


command = Describe
