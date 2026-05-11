# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import logging

from anemoi.training.commands import Command

LOG = logging.getLogger(__name__)


class Schema(Command):
    """Export the JSON schema of the training configuration."""

    def add_arguments(self, command_parser: argparse.ArgumentParser) -> None:
        pass

    def run(self, args: argparse.Namespace) -> None:  # noqa: ARG002
        import json

        from anemoi.training.schemas.base_schema import BaseSchema

        print(json.dumps(BaseSchema.model_json_schema(), indent=2))  # noqa: T201


command = Schema
