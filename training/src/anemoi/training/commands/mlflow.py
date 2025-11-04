# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import getpass
import json
import logging
from pathlib import Path

from anemoi.training.commands import Command

LOGGER = logging.getLogger(__name__)


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
        login_group = login.add_mutually_exclusive_group()
        login_group.add_argument(
            "--url",
            "-u",
            help="The URL of the authentication server. If not provided, the last used URL will be tried.",
        )
        login_group.add_argument(
            "--all",
            "-a",
            action="store_true",
            help="Log in to all known servers.",
        )
        login_group.add_argument(
            "--list",
            "-l",
            action="store_true",
            help="List all known servers.",
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
            metavar="DIR",
            required=True,
            default=argparse.SUPPRESS,
        )
        sync.add_argument(
            "--destination",
            "-d",
            help="The destination MLflow tracking URI.",
            metavar="URI",
            required=True,
            default=argparse.SUPPRESS,
        )
        sync.add_argument(
            "--run-id",
            "-r",
            help="The run ID to sync.",
            metavar="ID",
            required=True,
            default=argparse.SUPPRESS,
        )
        sync.add_argument(
            "--experiment-name",
            "-e",
            help="The experiment name to sync to.",
            metavar="NAME",
            default="anemoi-debug",
        )
        sync.add_argument(
            "--authentication",
            "-a",
            action="store_true",
            help="The destination server requires authentication.",
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

        help_msg = "Create an MLflow run_id given a training configuration."
        prepare = subparsers.add_parser(
            "prepare",
            help=help_msg,
            description=help_msg,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        prepare.add_argument(
            "--config-name",
            default="dev",
            help="Name of the training configuration.",
        )
        prepare.add_argument(
            "--owner",
            default=getpass.getuser(),
            help="Name of the training configuration.",
        )
        prepare.add_argument(
            "--run-name",
            default=None,
            help="MLflow run name.",
        )
        prepare.add_argument(
            "--output",
            default="./mlflow_metadata.json",
            type=Path,
            help="Output file path.",
        )
        prepare.add_argument(
            "--verbose",
            "-v",
            action="store_true",
        )

    @staticmethod
    def run(args: argparse.Namespace) -> None:  # noqa: C901
        if args.subcommand == "login":
            from datetime import datetime
            from datetime import timezone

            from anemoi.utils.mlflow.auth import TokenAuth

            if args.list:
                servers = TokenAuth.get_servers()
                if not servers:
                    LOGGER.error("No known servers. Rerun the login command with --url")
                    return
                LOGGER.info("Known servers:")
                for url, refresh_expires in servers:
                    expires = datetime.fromtimestamp(refresh_expires, tz=timezone.utc)
                    if expires < datetime.now(tz=timezone.utc):
                        LOGGER.info(" - %s (expired, login required)", url)
                        continue
                    LOGGER.info(" - %s (expires: %s UTC)", url, expires.strftime("%Y-%m-%d %H:%M:%S"))
                return

            if args.url:
                urls = [args.url]  # specific url
            else:
                urls = [url for url, _ in TokenAuth.get_servers()]  # all urls
                if not args.all and len(urls) > 0:
                    urls = [urls[0]]  # last used url

            if not urls:
                msg = "No URL provided and no past URL found. Rerun the command with --url"
                raise ValueError(msg)

            for url in urls:
                TokenAuth(url=url).login(force_credentials=args.force_credentials)
            return

        if args.subcommand == "sync":
            from anemoi.training.utils.mlflow_sync import MlFlowSync
            from anemoi.utils.mlflow.utils import health_check

            if args.authentication:
                from anemoi.utils.mlflow.auth import TokenAuth

                auth = TokenAuth(url=args.destination)
                auth.login()
                auth.authenticate()

            health_check(args.destination)

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

        if args.subcommand == "prepare":
            import mlflow
            from hydra import compose
            from hydra import initialize

            from anemoi.training.diagnostics.mlflow.logger import AnemoiMLflowLogger
            from anemoi.utils.mlflow.client import AnemoiMlflowClient

            # Load configuration and resolve schema
            with initialize(version_base=None, config_path="./"):
                config = compose(config_name=args.config_name)

            # Create MLflow client and get experiment
            client = AnemoiMlflowClient(config.diagnostics.log.mlflow.tracking_uri, authentication=True)
            experiment = client.get_experiment_by_name(config.diagnostics.log.mlflow.experiment_name)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else client.create_experiment(config.diagnostics.log.mlflow.experiment_name)
            )

            # Parse configuration
            if config.training.run_id is not None:  # Existing run_id
                LOGGER.info("Existing run_id: %s", config.training.run_id)
                try:
                    client.get_run(config.training.run_id)
                except ValueError as e:
                    msg = "Invalid run_id provided."
                    raise ValueError(msg) from e
                return

            # Create a new run attached to the experiment
            run_name = args.run_name if args.run_name is not None else config.diagnostics.log.mlflow.run_name
            run = client.create_run(experiment_id, run_name=run_name)
            run_id = run.info.run_id
            LOGGER.info("Creating new run_id: %s", run_id)

            # Log metadata to MLflow server
            mlflow.set_tracking_uri(config.diagnostics.log.mlflow.tracking_uri)
            client.set_tag(run_id, "mlflow.user", args.owner)
            client.set_tag(run_id, "dry_run", True)
            client.set_tag(run_id, "mlflow.source.name", "anemoi-training mlflow prepare")
            AnemoiMLflowLogger.log_hyperparams_in_mlflow(
                client,
                run_id,
                config,
                expand_keys=config.diagnostics.log.mlflow.expand_hyperparams,
                log_hyperparams=True,
                clean_params=False,
            )

            # Dump run ID in output file
            LOGGER.info("Saving run id in file in %s.", args.output)
            mlflow_metadata = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "experiment_name": config.diagnostics.log.mlflow.experiment_name,
                "tracking_uri": config.diagnostics.log.mlflow.tracking_uri,
            }
            with Path.open(args.output, "w") as fp:
                json.dump(mlflow_metadata, fp)

            return
        return


command = MlFlow
