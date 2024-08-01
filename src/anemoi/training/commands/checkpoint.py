# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging

from . import Command

LOG = logging.getLogger(__name__)


class Checkpoint(Command):
    """Commands to interact with training checkpoints."""

    def add_arguments(self, command_parser: argparse.ArgumentParser):
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        help = "Save an anemoi training checkpoint for inference."
        inference = subparsers.add_parser(
            "inference",
            help=help,
            description=help,
        )
        inference.add_argument("--input", "-i", required=True, metavar="input.ckpt")
        inference.add_argument("--output", "-o", required=True, metavar="output.ckpt")

    def run(self, args):
        if args.subcommand == "inference":
            LOG.info(f"Loading training checkpoint from {args.input}, please wait...")

            from anemoi.training.utils.checkpoint import load_and_prepare_model
            from anemoi.training.utils.checkpoint import save_inference_checkpoint

            module, metadata = load_and_prepare_model(args.input)
            path = save_inference_checkpoint(module, metadata, args.output)

            LOG.info(f"Inference checkpoint saved to {path}")
            return


command = Checkpoint
