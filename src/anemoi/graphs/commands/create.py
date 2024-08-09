from pathlib import Path

from anemoi.graphs.create import GraphCreator

from . import Command


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
            "config", type=Path, help="Configuration yaml file path defining the recipe to create the graph."
        )
        command_parser.add_argument("save_path", type=Path, help="Path to store the created graph.")

    def run(self, args):

        graph_creator = GraphCreator(config=args.config)
        graph_creator.create(save_path=args.save_path, overwrite=args.overwrite)


command = Create
