#!/usr/bin/env python
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

from anemoi.training.train.train import main as anemoi_train

from . import Command

LOGGER = logging.getLogger(__name__)


class Train(Command):
    accept_unknown_args = True

    def add_arguments(self, parser):
        return parser

    def run(self, args, unknown_args=None):
        del args

        if unknown_args is not None:
            sys.argv = [sys.argv[0]] + unknown_args
        else:
            sys.argv = [sys.argv[0]]

        LOGGER.info(f"Running anemoi training command with overrides: {sys.argv[1:]}")
        anemoi_train()


command = Train
