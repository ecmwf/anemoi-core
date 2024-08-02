# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import sys
from typing import TYPE_CHECKING

from . import Command

if TYPE_CHECKING:
    import argparse

LOGGER = logging.getLogger(__name__)


class Train(Command):
    """Commands to train Anemoi models."""

    accept_unknown_args = True

    @staticmethod
    def add_arguments(parser: "argparse.ArgumentParser") -> "argparse.ArgumentParser":
        return parser

    @staticmethod
    def run(args, unknown_args=None) -> None:  # noqa: ANN001
        del args

        if unknown_args is not None:
            sys.argv = [sys.argv[0], *unknown_args]
        else:
            sys.argv = [sys.argv[0]]

        LOGGER.info("Running anemoi training command with overrides: %s", sys.argv[1:])
        from anemoi.training.train.train import main as anemoi_train

        anemoi_train()


command = Train
