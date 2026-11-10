# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.training.data.data_reader import BaseAnemoiReader


class _Store:
    """Dataset with tendency statistics for a subset of deltas."""

    def __init__(self, variables: list[str], stdev: list[float], tendencies: dict[str, list[float]]) -> None:
        self.variables = variables
        self._stdev = stdev
        self._tendencies = tendencies

    @property
    def statistics(self) -> dict:
        return {"stdev": np.array(self._stdev)}

    def statistics_tendencies(self, delta: str) -> dict:
        if delta not in self._tendencies:
            raise KeyError(delta)
        return {"stdev": np.array(self._tendencies[delta])}


class _Static:
    """Auxiliary dataset with no tendency statistics, e.g. static orography fields."""

    def __init__(self, variables: list[str], stdev: list[float]) -> None:
        self.variables = variables
        self._stdev = stdev

    @property
    def statistics(self) -> dict:
        return {"stdev": np.array(self._stdev)}

    def statistics_tendencies(self, delta: str) -> dict:
        raise KeyError(delta)


class _Join:
    """Mimics ``anemoi.datasets`` Join: concatenates and re-raises any source error."""

    def __init__(self, datasets: list) -> None:
        self.datasets = datasets

    @property
    def variables(self) -> list[str]:
        return [variable for dataset in self.datasets for variable in dataset.variables]

    @property
    def statistics(self) -> dict:
        return {"stdev": np.concatenate([dataset.statistics["stdev"] for dataset in self.datasets])}

    def statistics_tendencies(self, delta: str) -> dict:
        return {
            "stdev": np.concatenate([dataset.statistics_tendencies(delta)["stdev"] for dataset in self.datasets]),
        }


class _Reader(BaseAnemoiReader):
    """Minimal reader exposing only the dataset under test."""

    def __init__(self, data: object) -> None:
        self.data = data


@pytest.fixture
def joined() -> _Join:
    trajectory = _Store(["2t", "sp"], [10.0, 200.0], {"1h": [0.4, 1.6], "6h": [2.0, 8.0]})
    auxiliary = _Static(["sdor", "slor"], [3.0, 4.0])
    return _Join([trajectory, auxiliary])


def test_join_lookup_fails_without_the_fallback(joined: _Join) -> None:
    """The joined dataset itself cannot serve tendency statistics."""
    with pytest.raises(KeyError):
        joined.statistics_tendencies("1h")


def test_partial_tendency_statistics_are_recovered(joined: _Join) -> None:
    """Sources that have tendency statistics are used despite a source that has none."""
    statistics = _Reader(joined).statistics_tendencies("1h")

    assert statistics is not None
    np.testing.assert_allclose(statistics["stdev"], [0.4, 1.6, 3.0, 4.0])


def test_variables_without_tendencies_are_left_unscaled(joined: _Join) -> None:
    """The fallback yields a scaling of exactly 1.0 for the variables it covers."""
    statistics = _Reader(joined).statistics_tendencies("1h")

    scaling = joined.statistics["stdev"] / statistics["stdev"]
    np.testing.assert_allclose(scaling, [25.0, 125.0, 1.0, 1.0])


def test_other_deltas_still_resolve(joined: _Join) -> None:
    """A delta present in the store is picked up as well."""
    statistics = _Reader(joined).statistics_tendencies("6h")

    np.testing.assert_allclose(statistics["stdev"], [2.0, 8.0, 3.0, 4.0])


def test_delta_absent_everywhere_returns_none(joined: _Join) -> None:
    """A delta no source provides still yields None, as before the change."""
    assert _Reader(joined).statistics_tendencies("2h") is None


def test_unjoined_dataset_is_returned_unchanged() -> None:
    """The direct lookup remains the fast path for a plain dataset."""
    store = _Store(["2t"], [10.0], {"1h": [0.4]})

    np.testing.assert_allclose(_Reader(store).statistics_tendencies("1h")["stdev"], [0.4])
    assert _Reader(store).statistics_tendencies("2h") is None


def test_wrapped_source_is_traversed() -> None:
    """Wrappers such as select/subset expose the wrapped dataset as `forward`."""

    class _Wrapper:
        def __init__(self, forward: object) -> None:
            self.forward = forward
            self.variables = forward.variables

        @property
        def statistics(self) -> dict:
            return self.forward.statistics

        def statistics_tendencies(self, delta: str) -> dict:
            raise AttributeError(delta)

    trajectory = _Store(["2t", "sp"], [10.0, 200.0], {"1h": [0.4, 1.6]})
    joined = _Join([_Wrapper(trajectory), _Static(["sdor", "slor"], [3.0, 4.0])])

    statistics = _Reader(joined).statistics_tendencies("1h")

    np.testing.assert_allclose(statistics["stdev"], [0.4, 1.6, 3.0, 4.0])
