# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import logging
import os
import sys
from abc import ABC

from anemoi.training.commands._base import HydraCommand

LOGGER = logging.getLogger(__name__)


class TrainBase(HydraCommand, ABC):
    env_var = "ANEMOI_TRAINING_CMD"


class Train(TrainBase):
    """Commands to train Anemoi models."""

    def run(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None:
        self.prepare_sysargv(args, unknown_args)
        LOGGER.info("Running anemoi training command with overrides: %s", sys.argv[1:])
        main()


def main() -> None:
    # Use the environment variable to check if main is being called from the subcommand, not from the ddp entrypoint
    if not os.environ.get("ANEMOI_TRAINING_CMD"):
        error = "This entrypoint should not be called directly. Use `anemoi-training train` instead."
        raise RuntimeError(error)

    from anemoi.training.train.train import main as anemoi_train

    anemoi_train()


command = Train
