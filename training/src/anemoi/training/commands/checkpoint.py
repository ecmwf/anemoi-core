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

from anemoi.training.commands import Command

LOG = logging.getLogger(__name__)


class Checkpoint(Command):
    """Commands to interact with training checkpoints."""

    @staticmethod
    def add_arguments(command_parser: argparse.ArgumentParser) -> None:
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        help_msg = "Save an anemoi training checkpoint for inference."
        inference = subparsers.add_parser(
            "inference",
            help=help_msg,
            description=help_msg,
        )
        inference.add_argument("--input", "-i", required=True, metavar="training.ckpt")
        inference.add_argument("--output", "-o", required=True, metavar="inference.ckpt")

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        if args.subcommand == "inference":
            LOG.info("Loading training checkpoint from %s, please wait...", args.input)

            from anemoi.training.utils.checkpoint import load_and_prepare_model
            from anemoi.training.utils.checkpoint import save_inference_checkpoint

            module, metadata = load_and_prepare_model(args.input)
            path = save_inference_checkpoint(module, metadata, args.output)

            LOG.info("Inference checkpoint saved to %s", path)
            return


command = Checkpoint
