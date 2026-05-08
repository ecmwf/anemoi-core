# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.graphs.describe import GraphDescriptor

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
