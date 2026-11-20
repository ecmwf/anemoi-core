# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pickle

import pytest
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.utils import check_loss_and_metrics_ignore_nans
from anemoi.training.losses.utils import check_loss_tree_ignores_nans
from anemoi.training.losses.utils import check_loss_tree_variable_units


def test_check_loss_tree_recurses_into_combined_loss() -> None:
    """Test that the tree walker finds variable mappers inside CombinedLoss."""
    data_indices = IndexCollection(
        DictConfig({"forcing": ["forcing"], "diagnostic": [], "target": ["imerg"]}),
        {"tp": 0, "forcing": 1, "imerg": 2},
    )
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "losses": [
                    {
                        "_target_": "anemoi.training.losses.MAELoss",
                        "predicted_variables": ["tp"],
                        "target_variables": ["imerg"],
                    },
                ],
            },
        ),
        data_indices=data_indices,
    )

    # Compatible units - should not raise
    variables_metadata = {"tp": {"units": "kg m**-2"}, "imerg": {"units": "kg m**-2"}}
    check_loss_tree_variable_units(loss, variables_metadata)

    # Incompatible units - should raise
    variables_metadata_bad = {"tp": {"units": "kg m**-2"}, "imerg": {"units": "mm"}}
    with pytest.raises(ValueError, match="Units are not compatible"):
        check_loss_tree_variable_units(loss, variables_metadata_bad)


NAN_NAME_TO_INDEX = {"x": 0, "tc_msl": 1, "tc_ws": 2}


@pytest.fixture
def nan_data_indices() -> IndexCollection:
    return IndexCollection(
        DictConfig({"forcing": [], "diagnostic": ["tc_msl", "tc_ws"], "allow_nans": ["tc_msl"]}),
        NAN_NAME_TO_INDEX,
    )


def _mse_config(ignore_nans: bool, **kwargs: object) -> DictConfig:
    return DictConfig({"_target_": "anemoi.training.losses.MSELoss", "ignore_nans": ignore_nans, **kwargs})


def _combined_config(*ignore_nans: bool) -> DictConfig:
    return DictConfig(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [_mse_config(flag) for flag in ignore_nans],
        },
    )


def _time_aggregate_config(loss_fn: DictConfig) -> DictConfig:
    return DictConfig(
        {
            "_target_": "anemoi.training.losses.aggregate.TimeAggregateLossWrapper",
            "time_aggregation_types": ["mean", "diff"],
            "loss_fn": loss_fn,
        },
    )


def test_check_loss_tree_ignores_nans_passes_when_loss_ignores_nans(nan_data_indices: IndexCollection) -> None:
    loss = get_loss_function(_mse_config(True), data_indices=nan_data_indices)
    check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


def test_check_loss_tree_ignores_nans_raises_when_loss_does_not_ignore_nans(
    nan_data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(_mse_config(False), data_indices=nan_data_indices)
    with pytest.raises(ValueError, match=r"training loss of dataset 'data' .*does not ignore NaNs.*tc_msl"):
        check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss of dataset 'data'")


def test_check_loss_tree_ignores_nans_noop_without_allow_nans(nan_data_indices: IndexCollection) -> None:
    loss = get_loss_function(_mse_config(False), data_indices=nan_data_indices)
    check_loss_tree_ignores_nans(loss, [], "training loss")


def test_check_loss_tree_ignores_nans_raises_for_loss_that_cannot_ignore_nans(
    nan_data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(
        DictConfig({"_target_": "anemoi.training.losses.spectral.LogFFT2Distance", "x_dim": 4, "y_dim": 4}),
        data_indices=nan_data_indices,
    )
    with pytest.raises(ValueError, match=r"LogFFT2Distance\) does not ignore NaNs"):
        check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


@pytest.mark.parametrize(
    ("variables", "raises"),
    [(["x"], False), (["tc_ws"], False), (["tc_msl"], True), (["x", "tc_msl"], True)],
)
def test_check_loss_tree_ignores_nans_only_checks_variables_the_loss_sees(
    nan_data_indices: IndexCollection,
    variables: list[str],
    raises: bool,
) -> None:
    loss = get_loss_function(
        _mse_config(False, predicted_variables=variables, target_variables=variables),
        data_indices=nan_data_indices,
    )
    if raises:
        with pytest.raises(ValueError, match="does not ignore NaNs"):
            check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")
    else:
        check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


@pytest.mark.parametrize(("flags", "raises"), [((True, True), False), ((True, False), True), ((False, False), True)])
def test_check_loss_tree_ignores_nans_recurses_into_combined_loss(
    nan_data_indices: IndexCollection,
    flags: tuple[bool, ...],
    raises: bool,
) -> None:
    loss = get_loss_function(_combined_config(*flags), data_indices=nan_data_indices)
    if raises:
        with pytest.raises(ValueError, match="does not ignore NaNs"):
            check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")
    else:
        check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


def test_check_loss_tree_ignores_nans_keeps_variable_selection_inside_wrapper(
    nan_data_indices: IndexCollection,
) -> None:
    """A mapper restricted to other variables inside a transparent wrapper must not reject `allow_nans`."""
    loss = get_loss_function(
        _time_aggregate_config(_mse_config(False, predicted_variables=["x"], target_variables=["x"])),
        data_indices=nan_data_indices,
    )
    check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


def test_check_loss_tree_ignores_nans_raises_for_unrestricted_loss_inside_wrapper(
    nan_data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(_time_aggregate_config(_mse_config(False)), data_indices=nan_data_indices)
    with pytest.raises(ValueError, match=r"MSELoss\) does not ignore NaNs"):
        check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


def test_check_loss_tree_ignores_nans_passes_for_ignoring_loss_inside_wrapper(
    nan_data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(_time_aggregate_config(_mse_config(True)), data_indices=nan_data_indices)
    check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


def test_check_loss_tree_ignores_nans_wrapper_inside_combined_loss(nan_data_indices: IndexCollection) -> None:
    config = DictConfig(
        {
            "_target_": "anemoi.training.losses.combined.CombinedLoss",
            "losses": [
                _time_aggregate_config(_mse_config(False, predicted_variables=["x"], target_variables=["x"])),
                _mse_config(True),
            ],
        },
    )
    loss = get_loss_function(config, data_indices=nan_data_indices)
    check_loss_tree_ignores_nans(loss, nan_data_indices.allow_nans, "training loss")


# Startup validation of a dataset: `check_loss_and_metrics_ignore_nans` is what `BaseTrainingModule.__init__` calls
# once the training loss and the validation metrics of a dataset have been built.


def test_startup_check_passes_when_loss_and_metrics_ignore_nans(nan_data_indices: IndexCollection) -> None:
    loss = get_loss_function(_mse_config(True), data_indices=nan_data_indices)
    metrics = {"mse": get_loss_function(_mse_config(True), data_indices=nan_data_indices)}
    check_loss_and_metrics_ignore_nans(loss, metrics, nan_data_indices.allow_nans, "data")


def test_startup_check_raises_for_training_loss(nan_data_indices: IndexCollection) -> None:
    loss = get_loss_function(_mse_config(False), data_indices=nan_data_indices)
    metrics = {"mse": get_loss_function(_mse_config(True), data_indices=nan_data_indices)}
    with pytest.raises(ValueError, match=r"training loss of dataset 'data' \(MSELoss\) does not ignore NaNs"):
        check_loss_and_metrics_ignore_nans(loss, metrics, nan_data_indices.allow_nans, "data")


def test_startup_check_raises_for_validation_metric(nan_data_indices: IndexCollection) -> None:
    loss = get_loss_function(_mse_config(True), data_indices=nan_data_indices)
    metrics = {
        "mse": get_loss_function(_mse_config(True), data_indices=nan_data_indices),
        "mae": get_loss_function(_mse_config(False), data_indices=nan_data_indices),
    }
    with pytest.raises(ValueError, match=r"validation metric 'mae' of dataset 'data' \(MSELoss\) does not ignore NaNs"):
        check_loss_and_metrics_ignore_nans(loss, metrics, nan_data_indices.allow_nans, "data")


def test_startup_check_ignores_metric_that_does_not_see_the_variables(nan_data_indices: IndexCollection) -> None:
    loss = get_loss_function(_mse_config(True), data_indices=nan_data_indices)
    metrics = {
        "mse_x": get_loss_function(
            _mse_config(False, predicted_variables=["x"], target_variables=["x"]),
            data_indices=nan_data_indices,
        ),
    }
    check_loss_and_metrics_ignore_nans(loss, metrics, nan_data_indices.allow_nans, "data")


def test_startup_check_noop_without_allow_nans() -> None:
    indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": ["tc_msl", "tc_ws"]}), NAN_NAME_TO_INDEX)
    loss = get_loss_function(_mse_config(False), data_indices=indices)
    metrics = {"mse": get_loss_function(_mse_config(False), data_indices=indices)}
    check_loss_and_metrics_ignore_nans(loss, metrics, indices.allow_nans, "data")


def _legacy_index_collection() -> IndexCollection:
    """Index collection as unpickled from a checkpoint created before `allow_nans` existed."""
    indices = IndexCollection(DictConfig({"forcing": [], "diagnostic": ["tc_msl", "tc_ws"]}), NAN_NAME_TO_INDEX)
    del indices.__dict__["allow_nans"]
    legacy = pickle.loads(pickle.dumps(indices))  # noqa: S301
    assert "allow_nans" not in legacy.__dict__
    return legacy


def test_startup_check_with_legacy_index_collection() -> None:
    legacy = _legacy_index_collection()
    loss = get_loss_function(_mse_config(False), data_indices=legacy)
    metrics = {"mse": get_loss_function(_mse_config(False), data_indices=legacy)}

    check_loss_and_metrics_ignore_nans(loss, metrics, legacy.allow_nans, "data")
