# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import sys

from anemoi.training.commands.train import TrainBase

LOGGER = logging.getLogger(__name__)


class Profile(TrainBase):
    """Commands to profile Anemoi models."""

    accept_unknown_args = True
    command = "profile"

    def run(self, args: list[str], unknown_args: list[str] | None = None) -> None:
        # This will be picked up by the logger
        self.prepare_sysargv(args, unknown_args)

        LOGGER.info("Running anemoi profile command with overrides: %s", sys.argv[1:])
        main()


def main() -> None:
    # Use the environment variable to check if main is being called from the subcommand, not from the ddp entrypoint
    if not os.environ.get("ANEMOI_TRAINING_CMD"):
        error = "This entrypoint should not be called directly. Use `anemoi-training profiler` instead."
        raise RuntimeError(error)

    from anemoi.training.train.profiler import main as anemoi_profile

    anemoi_profile()


command = Profile
