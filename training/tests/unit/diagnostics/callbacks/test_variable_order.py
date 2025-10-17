# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.sanity import CheckVariableOrder
from anemoi.training.train.tasks.forecaster import GraphForecaster
from anemoi.training.train.train import AnemoiTrainer


@pytest.fixture
def name_to_index() -> dict:
    return {"a": 0, "b": 1, "c": 2}


@pytest.fixture
def name_to_index_permute() -> dict:
    return {"a": 0, "b": 2, "c": 1}


@pytest.fixture
def name_to_index_rename() -> dict:
    return {"a": 0, "b": 1, "d": 2}


@pytest.fixture
def name_to_index_partial_rename_permute() -> dict:
    return {"a": 2, "b": 1, "d": 0}


@pytest.fixture
def name_to_index_rename_permute() -> dict:
    return {"x": 2, "b": 1, "d": 0}


@pytest.fixture
def fake_trainer(mocker: Any, name_to_index: dict) -> AnemoiTrainer:
    trainer = mocker.Mock(spec=AnemoiTrainer)
    trainer.datamodule.data_indices.name_to_index = name_to_index
    trainer.datamodule.data_indices.compare_variables = types.MethodType(
        IndexCollection.compare_variables,
        trainer.datamodule.data_indices,
    )
    return trainer


@pytest.fixture
def fake_pl_module(mocker: Any, name_to_index: dict) -> MagicMock:
    pl_module = mocker.Mock()
    pl_module._ckpt_model_name_to_index = name_to_index
    return pl_module


@pytest.fixture
def callback() -> CheckVariableOrder:
    callback = CheckVariableOrder()
    assert callback is not None
    assert hasattr(callback, "on_train_start")
    assert hasattr(callback, "on_validation_start")
    assert hasattr(callback, "on_test_start")

    return callback


def test_on_epoch(
    fake_trainer: AnemoiTrainer,
    fake_pl_module: MagicMock,
    callback: CheckVariableOrder,
    name_to_index: dict,
) -> None:
    """Test all epoch functions with "working" indices."""
    fake_trainer.datamodule.ds_train.name_to_index = name_to_index
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index
    callback.on_train_start(fake_trainer, fake_pl_module)
    callback.on_validation_start(fake_trainer, fake_pl_module)
    callback.on_test_start(fake_trainer, fake_pl_module)

    assert (
        fake_trainer.datamodule.data_indices.compare_variables(
            fake_pl_module._ckpt_model_name_to_index,
            name_to_index,
        )
        is None
    )


def test_on_epoch_permute(
    fake_trainer: AnemoiTrainer,
    fake_pl_module: MagicMock,
    callback: CheckVariableOrder,
    name_to_index_permute: dict,
) -> None:
    """Test all epoch functions with permuted indices.

    Expecting errors in all cases.
    """
    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_permute
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_permute
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_train_start(fake_trainer, fake_pl_module)
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_validation_start(fake_trainer, fake_pl_module)
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_test_start(fake_trainer, fake_pl_module)
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)

    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        fake_trainer.datamodule.data_indices.compare_variables(
            fake_pl_module._ckpt_model_name_to_index,
            name_to_index_permute,
        )
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)


def test_on_epoch_rename(
    fake_trainer: AnemoiTrainer,
    fake_pl_module: MagicMock,
    callback: CheckVariableOrder,
    name_to_index_rename: dict,
) -> None:
    """Test all epoch functions with renamed indices.

    Expecting passes in all cases.
    """
    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_rename
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_rename
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_rename
    callback.on_train_start(fake_trainer, fake_pl_module)
    callback.on_validation_start(fake_trainer, fake_pl_module)
    callback.on_test_start(fake_trainer, fake_pl_module)

    fake_trainer.datamodule.data_indices.compare_variables(
        fake_pl_module._ckpt_model_name_to_index,
        name_to_index_rename,
    )


def test_on_epoch_rename_permute(
    fake_trainer: AnemoiTrainer,
    fake_pl_module: MagicMock,
    callback: CheckVariableOrder,
    name_to_index_rename_permute: dict,
) -> None:
    """Test all epoch functions with renamed and permuted indices.

    Expects all passes (but warnings).
    """
    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_rename_permute
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_rename_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_rename_permute
    callback.on_train_start(fake_trainer, fake_pl_module)
    callback.on_validation_start(fake_trainer, fake_pl_module)
    callback.on_test_start(fake_trainer, fake_pl_module)

    fake_trainer.datamodule.data_indices.compare_variables(
        fake_pl_module._ckpt_model_name_to_index,
        name_to_index_rename_permute,
    )


def test_on_epoch_partial_rename_permute(
    fake_trainer: AnemoiTrainer,
    fake_pl_module: MagicMock,
    callback: CheckVariableOrder,
    name_to_index_partial_rename_permute: dict,
) -> None:
    """Test all epoch functions with partially renamed and permuted indices.

    Expects all errors.
    """
    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_partial_rename_permute
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_partial_rename_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_partial_rename_permute
    with pytest.raises(ValueError, match=r"The variable order in the model and data is different."):
        callback.on_train_start(fake_trainer, fake_pl_module)
    with pytest.raises(ValueError, match=r"The variable order in the model and data is different."):
        callback.on_validation_start(fake_trainer, fake_pl_module)
    with pytest.raises(ValueError, match=r"The variable order in the model and data is different."):
        callback.on_test_start(fake_trainer, fake_pl_module)

    with pytest.raises(ValueError, match=r"The variable order in the model and data is different."):
        fake_trainer.datamodule.data_indices.compare_variables(
            fake_pl_module._ckpt_model_name_to_index,
            name_to_index_partial_rename_permute,
        )


def test_on_epoch_wrong_validation(
    fake_trainer: AnemoiTrainer,
    fake_pl_module: MagicMock,
    callback: CheckVariableOrder,
    name_to_index: dict,
    name_to_index_permute: dict,
    name_to_index_rename: dict,
) -> None:
    """Test all epoch functions with "working" indices, but different validation indices."""
    fake_trainer.datamodule.ds_train.name_to_index = name_to_index
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_rename
    callback.on_train_start(fake_trainer, fake_pl_module)
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_validation_start(fake_trainer, fake_pl_module)
    assert " {'c': (2, 1), 'b': (1, 2)}" in str(
        exc_info.value,
    ) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)
    callback.on_test_start(fake_trainer, fake_pl_module)

    assert (
        fake_trainer.datamodule.data_indices.compare_variables(
            fake_pl_module._ckpt_model_name_to_index,
            name_to_index,
        )
        is None
    )


def test_on_load_checkpoint_restores_name_to_index() -> None:

    model = GraphForecaster.__new__(GraphForecaster)

    model.on_load_checkpoint = types.MethodType(GraphForecaster.on_load_checkpoint, GraphForecaster)

    mock_name_to_index = {"var1": 0, "var2": 1}
    mock_checkpoint = {"hyper_parameters": {"data_indices": MagicMock(name_to_index=mock_name_to_index)}}
    # Act
    model.on_load_checkpoint(mock_checkpoint)

    # Assert
    assert model._ckpt_model_name_to_index == mock_name_to_index
