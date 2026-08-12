# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import sys
from unittest import mock

import pytest

from anemoi.training.commands._base import HydraCommand


class ConcreteCommand(HydraCommand):
    """Minimal concrete subclass for testing."""

    env_var = "ANEMOI_TEST_CMD"

    def run(self, args: argparse.Namespace, unknown_args: list[str] | None = None) -> None:
        pass


@pytest.fixture
def cmd() -> ConcreteCommand:
    return ConcreteCommand()


@pytest.fixture
def args() -> argparse.Namespace:
    return argparse.Namespace(command="train")


def test_merge_sysargv_basic(cmd: ConcreteCommand, args: argparse.Namespace) -> None:
    with mock.patch.object(sys, "argv", ["/env/bin/anemoi-training", "train"]):
        result = cmd._merge_sysargv(args)
    assert result == "/env/bin/.anemoi-training-train"


def test_merge_sysargv_with_subcommand(cmd: ConcreteCommand) -> None:
    args = argparse.Namespace(command="evaluate", subcommand="deterministic")
    with mock.patch.object(sys, "argv", ["/env/bin/anemoi-training", "evaluate"]):
        result = cmd._merge_sysargv(args)
    assert result == "/env/bin/.anemoi-training-evaluate-deterministic"


def test_prepare_sysargv_sets_env_var(cmd: ConcreteCommand, args: argparse.Namespace) -> None:
    with (
        mock.patch.object(sys, "argv", ["/env/bin/anemoi-training", "train"]),
        mock.patch.dict("os.environ", {}, clear=True),
    ):
        cmd.prepare_sysargv(args)
        import os

        assert os.environ["ANEMOI_TEST_CMD"] == "/env/bin/anemoi-training train"


def test_prepare_sysargv_no_unknown_args(cmd: ConcreteCommand, args: argparse.Namespace) -> None:
    with mock.patch.object(sys, "argv", ["/env/bin/anemoi-training", "train"]):
        cmd.prepare_sysargv(args)
        assert sys.argv == ["/env/bin/.anemoi-training-train"]


def test_prepare_sysargv_with_unknown_args(cmd: ConcreteCommand, args: argparse.Namespace) -> None:
    unknown = ["model=transformer", "training.lr=1e-4"]
    with mock.patch.object(sys, "argv", ["/env/bin/anemoi-training", "train"]):
        cmd.prepare_sysargv(args, unknown_args=unknown)
        assert sys.argv == ["/env/bin/.anemoi-training-train", "model=transformer", "training.lr=1e-4"]


def test_env_var_differs_per_subclass() -> None:
    class OtherCommand(HydraCommand):
        env_var = "OTHER_CMD"

        def run(self, args: list[str], unknown_args: list[str] | None = None) -> None:
            pass

    assert ConcreteCommand.env_var != OtherCommand.env_var
