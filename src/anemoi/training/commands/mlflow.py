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

        help_msg = "Log in and acquire a token from keycloak."
        login = subparsers.add_parser(
            "login",
            help=help_msg,
            description=help_msg,
        )
        login.add_argument(
            "--url",
            help="The URL of the authentication server. If not provided, the last used URL will be tried.",
        )
        login.add_argument(
            "--force-credentials",
            "-f",
            action="store_true",
            help="Force a credential login even if a token is available.",
        )

        help_msg = "Synchronise an offline run with an MLflow server."
        sync = subparsers.add_parser(
            "sync",
            help=help_msg,
            description=help_msg,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        sync.add_argument(
            "--source",
            "-s",
            help="The MLflow logs source directory.",
            required=True,
            default=argparse.SUPPRESS,
        )
        sync.add_argument(
            "--destination",
            "-d",
            help="The destination MLflow tracking URI.",
            required=True,
            default=argparse.SUPPRESS,
        )
        sync.add_argument("--run-id", "-r", help="The run ID to sync.", required=True, default=argparse.SUPPRESS)
        sync.add_argument(
            "--experiment-name",
            "-e",
            help="The experiment name to sync to.",
            default="anemoi-debug",
        )
        sync.add_argument(
            "--export-deleted-runs",
            "-x",
            action="store_true",
        )
        sync.add_argument(
            "--verbose",
            "-v",
            action="store_true",
        )

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        if args.subcommand == "login":
            from anemoi.training.diagnostics.mlflow.auth import TokenAuth

            url = args.url or TokenAuth.load_config().get("url")

            if not url:
                msg = "No URL provided and no past URL found. Rerun the command with --url"
                raise ValueError(msg)

            TokenAuth(url=url).login(force_credentials=args.force_credentials)
            return

        if args.subcommand == "sync":
            from anemoi.training.utils.mlflow_sync import MlFlowSync

            log_level = "DEBUG" if args.verbose else "INFO"

            MlFlowSync(
                args.source,
                args.destination,
                args.run_id,
                args.experiment_name,
                args.export_deleted_runs,
                log_level,
            ).sync()
            return


command = MlFlow
