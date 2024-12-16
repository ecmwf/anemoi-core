# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import pytest
from anemoi.models.data_indices.collection import IndexCollection

from anemoi.training.diagnostics.callbacks.sanity import CheckVariableOrder
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
    return trainer


@pytest.fixture
def checkpoint(mocker: Any, name_to_index: dict) -> dict[str, dict[str, IndexCollection]]:
    data_index = mocker.Mock(spec=IndexCollection)
    data_index.name_to_index = name_to_index
    return {"hyper_parameters": {"data_indices": data_index}}


@pytest.fixture
def callback() -> CheckVariableOrder:
    callback = CheckVariableOrder()
    assert callback is not None
    assert hasattr(callback, "on_load_checkpoint")
    assert hasattr(callback, "on_sanity_check_start")
    assert hasattr(callback, "on_train_epoch_start")
    assert hasattr(callback, "on_validation_epoch_start")
    assert hasattr(callback, "on_test_epoch_start")

    assert callback._model_name_to_index is None

    return callback


def test_on_load_checkpoint(
    fake_trainer: AnemoiTrainer,
    callback: CheckVariableOrder,
    checkpoint: dict,
    name_to_index: dict,
) -> None:
    assert callback._model_name_to_index is None
    callback.on_load_checkpoint(fake_trainer, None, checkpoint)
    assert callback._model_name_to_index == name_to_index

    assert callback._compare_variables(name_to_index) is None


def test_on_sanity(fake_trainer: AnemoiTrainer, callback: CheckVariableOrder, name_to_index: dict) -> None:
    assert callback._model_name_to_index is None
    callback.on_sanity_check_start(fake_trainer, None)
    assert callback._model_name_to_index == name_to_index

    assert callback._compare_variables(name_to_index) is None


def test_on_epoch(fake_trainer: AnemoiTrainer, callback: CheckVariableOrder, name_to_index: dict) -> None:
    """Test all epoch functions with "working" indices."""
    assert callback._model_name_to_index is None
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)
    assert callback._model_name_to_index is None

    assert callback._compare_variables(name_to_index) is None

    # Test with initialised model_name_to_index
    callback.on_sanity_check_start(fake_trainer, None)
    assert callback._model_name_to_index == name_to_index

    fake_trainer.datamodule.ds_train.name_to_index = name_to_index
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)

    assert callback._compare_variables(name_to_index) is None


def test_on_epoch_permute(
    fake_trainer: AnemoiTrainer,
    callback: CheckVariableOrder,
    name_to_index: dict,
    name_to_index_permute: dict,
) -> None:
    """Test all epoch functions with permuted indices.

    Expecting errors in all cases.
    """
    assert callback._model_name_to_index is None
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)
    assert callback._model_name_to_index is None

    assert callback._compare_variables(name_to_index) is None

    # Test with initialised model_name_to_index
    callback.on_sanity_check_start(fake_trainer, None)
    assert callback._model_name_to_index == name_to_index

    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_permute
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_permute
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_train_epoch_start(fake_trainer, None)
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_validation_epoch_start(fake_trainer, None)
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_test_epoch_start(fake_trainer, None)
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)

    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback._compare_variables(name_to_index_permute)
    assert "{'c': (2, 1), 'b': (1, 2)}" in str(exc_info.value) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)


def test_on_epoch_rename(
    fake_trainer: AnemoiTrainer,
    callback: CheckVariableOrder,
    name_to_index: dict,
    name_to_index_rename: dict,
) -> None:
    """Test all epoch functions with renamed indices.

    Expecting passes in all cases.
    """
    assert callback._model_name_to_index is None
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)
    assert callback._model_name_to_index is None

    assert callback._compare_variables(name_to_index) is None

    # Test with initialised model_name_to_index
    callback.on_sanity_check_start(fake_trainer, None)
    assert callback._model_name_to_index == name_to_index

    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_rename
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_rename
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_rename
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)

    callback._compare_variables(name_to_index_rename)


def test_on_epoch_rename_permute(
    fake_trainer: AnemoiTrainer,
    callback: CheckVariableOrder,
    name_to_index: dict,
    name_to_index_rename_permute: dict,
) -> None:
    """Test all epoch functions with renamed and permuted indices.

    Expects all passes (but warnings).
    """
    assert callback._model_name_to_index is None
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)
    assert callback._model_name_to_index is None

    assert callback._compare_variables(name_to_index) is None

    # Test with initialised model_name_to_index
    callback.on_sanity_check_start(fake_trainer, None)
    assert callback._model_name_to_index == name_to_index

    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_rename_permute
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_rename_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_rename_permute
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)

    callback._compare_variables(name_to_index_rename_permute)


def test_on_epoch_partial_rename_permute(
    fake_trainer: AnemoiTrainer,
    callback: CheckVariableOrder,
    name_to_index: dict,
    name_to_index_partial_rename_permute: dict,
) -> None:
    """Test all epoch functions with partially renamed and permuted indices.

    Expects all errors.
    """
    assert callback._model_name_to_index is None
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)
    assert callback._model_name_to_index is None

    assert callback._compare_variables(name_to_index) is None

    # Test with initialised model_name_to_index
    callback.on_sanity_check_start(fake_trainer, None)
    assert callback._model_name_to_index == name_to_index

    fake_trainer.datamodule.ds_train.name_to_index = name_to_index_partial_rename_permute
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_partial_rename_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_partial_rename_permute
    with pytest.raises(ValueError, match="The variable order in the model and data is different."):
        callback.on_train_epoch_start(fake_trainer, None)
    with pytest.raises(ValueError, match="The variable order in the model and data is different."):
        callback.on_validation_epoch_start(fake_trainer, None)
    with pytest.raises(ValueError, match="The variable order in the model and data is different."):
        callback.on_test_epoch_start(fake_trainer, None)

    with pytest.raises(ValueError, match="The variable order in the model and data is different."):
        callback._compare_variables(name_to_index_partial_rename_permute)


def test_on_epoch_wrong_validation(
    fake_trainer: AnemoiTrainer,
    callback: CheckVariableOrder,
    name_to_index: dict,
    name_to_index_permute: dict,
    name_to_index_rename: dict,
) -> None:
    """Test all epoch functions with "working" indices, but different validation indices."""
    assert callback._model_name_to_index is None
    callback.on_train_epoch_start(fake_trainer, None)
    callback.on_validation_epoch_start(fake_trainer, None)
    callback.on_test_epoch_start(fake_trainer, None)
    assert callback._model_name_to_index is None

    assert callback._compare_variables(name_to_index) is None

    # Test with initialised model_name_to_index
    callback.on_sanity_check_start(fake_trainer, None)
    assert callback._model_name_to_index == name_to_index

    fake_trainer.datamodule.ds_train.name_to_index = name_to_index
    fake_trainer.datamodule.ds_valid.name_to_index = name_to_index_permute
    fake_trainer.datamodule.ds_test.name_to_index = name_to_index_rename
    callback.on_train_epoch_start(fake_trainer, None)
    with pytest.raises(ValueError, match="Detected a different sort order of the same variables:") as exc_info:
        callback.on_validation_epoch_start(fake_trainer, None)
    assert " {'c': (2, 1), 'b': (1, 2)}" in str(
        exc_info.value,
    ) or "{'b': (1, 2), 'c': (2, 1)}" in str(exc_info.value)
    callback.on_test_epoch_start(fake_trainer, None)

    assert callback._compare_variables(name_to_index) is None
