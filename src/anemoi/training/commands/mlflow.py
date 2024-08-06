# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse

from anemoi.training.commands import Command


class MlFlow(Command):
    """Commands to interact with MLflow."""

    @staticmethod
    def add_arguments(command_parser: argparse.ArgumentParser) -> None:
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        login = subparsers.add_parser(
            "login",
            help="Log in and acquire a token from keycloak.",
        )
        login.add_argument(
            "--url",
            help="The URL of the authentication server",
            required=True,
            # TODO (Gert Mertes): once we have a config file, make this optional
            # and load default value from config
        )
        login.add_argument(
            "--force-credentials",
            "-f",
            action="store_true",
            help="Force a credential login even if a token is available.",
        )

        subparsers.add_parser(
            "sync",
            help="Synchronise an offline run with an MLflow server (placeholder, not implemented).",
        )

    @staticmethod
    def run(args: "argparse.Namespace") -> None:
        if args.subcommand == "login":
            from anemoi.training.diagnostics.mlflow.auth import TokenAuth

            TokenAuth(url=args.url).login(force_credentials=args.force_credentials)
            return

        if args.subcommand == "sync":
            raise NotImplementedError


command = MlFlow
