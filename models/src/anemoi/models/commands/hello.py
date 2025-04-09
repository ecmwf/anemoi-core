# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Command place holder. Delete when we have real commands."""

from . import Command


def say_hello(greetings, who):
    print(greetings, who)


class Hello(Command):

    def add_arguments(self, command_parser):
        command_parser.add_argument("--greetings", default="hello")
        command_parser.add_argument("--who", default="world")

    def run(self, args):
        say_hello(args.greetings, args.who)


command = Hello
