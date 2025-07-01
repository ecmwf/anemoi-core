# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.graphs.export import GraphExporter

from . import Command


class ExportToSparse(Command):
    """Export a graph edges to a sparse format.

    Example usage specifying an edge attribute:
    ```
    anemoi-graphs export-to-sparse graph.pt output_path --edge_attribute_name edge_attr
    ```

    Example usage specifying a subset of edges:
    ```
    anemoi-graphs export-to-sparse graph.pt output_path --edges-name data down --edges-name down data
    ```
    """

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument("graph", help="Path to the graph (a .PT file) or a config file defining the graph.")
        command_parser.add_argument("output_path", help="Path to store the inspection results.")
        command_parser.add_argument("--edge_attribute_name", default=None, help="Name of the edge attribute to export.")
        command_parser.add_argument(
            "--edges-name",
            nargs=2,
            action="append",
            metavar=("NODE1", "NODE2"),
            help="Specify an edge by its two node names. Can be used multiple times.",
        )

    def run(self, args):
        kwargs = vars(args)
        edges_name = kwargs.get("edges_name", None)
        if edges_name is not None:
            # Convert list of lists to list of tuples
            kwargs["edges_name"] = [tuple(pair) for pair in edges_name]

        GraphExporter(
            graph=kwargs["graph"],
            output_path=kwargs["output_path"],
            edge_attribute_name=kwargs["edge_attribute_name"],
            edges_name=kwargs["edges_name"],
        ).export()


command = ExportToSparse
