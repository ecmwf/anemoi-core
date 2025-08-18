# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from types import SimpleNamespace

import numpy as np

# The function lives in utils.py (a module file, not a package directory)
from anemoi.training.losses.utils import print_variable_scaling

# Use the logger from that module (LOGGER = logging.getLogger(__name__) there)
LOGGER = logging.getLogger(print_variable_scaling.__module__)


# -------------------------- Minimal fakes -----------------------------------


class _FakeScalerSubset:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr)

    def get_scaler(self, *_args, **_kwargs) -> np.ndarray:
        # Return exactly what we seeded (shape preserved); the function will reshape(-1)
        return self._arr.copy()


class _FakeScaler:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.asarray(arr)

    def subset_by_dim(self, _dim) -> "_FakeScalerSubset":
        # dim is ignored in this fake; real code uses TensorDim.VARIABLE.value
        return _FakeScalerSubset(self._arr)


class _FakeLoss:
    def __init__(self, arr: np.ndarray) -> None:
        self.scaler = _FakeScaler(arr)


def _fake_indices(names: list[str]):
    # Must provide data_indices.model.output.name_to_index (ordered)
    name_to_index = {name: i for i, name in enumerate(names)}
    model = SimpleNamespace(output=SimpleNamespace(name_to_index=name_to_index))
    return SimpleNamespace(model=model)


class _ListHandler(logging.Handler):
    """Capture log records as plain strings."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _count_logged_pairs(message: str) -> int:
    """Count 'name: value' pairs in the function's log line."""
    prefix = "Final Variable Scaling: "
    payload = message.removeprefix(prefix)
    # The function ends with a trailing ", " — filter empty tail
    parts = [p for p in payload.split(", ") if p]
    return len(parts)


# ------------------------------ Tests ---------------------------------------


def test_print_variable_scaling_single_var_flattens_but_not_scalar() -> None:
    """For a single variable, various shapes should flatten to a length-1 vector (never 0-D scalar)."""
    names = ["swh"]
    indices = _fake_indices(names)
    shapes = [(), (1,), (1, 1), (1, 1, 1)]  # shapes that squeeze() could collapse to 0-D

    for shape in shapes:
        arr = np.array([1.23]).reshape(shape) if shape else np.array(1.23)
        loss = _FakeLoss(arr)

        handler = _ListHandler()
        old_level = LOGGER.level
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.addHandler(handler)
        try:
            # If reshape(-1) were not used, some shapes would turn into 0-D and indexing would blow up.
            print_variable_scaling(loss, indices)
        finally:
            LOGGER.removeHandler(handler)
            LOGGER.setLevel(old_level)

        msgs = [m for m in handler.messages if m.startswith("Final Variable Scaling: ")]
        assert msgs, f"No log captured for shape {shape}"
        assert _count_logged_pairs(msgs[-1]) == 1, f"Expected 1 pair for shape {shape}"


def test_print_variable_scaling_multi_var_flattens_to_correct_length() -> None:
    """For multiple variables, various shapes should flatten to a vector with the correct length."""
    names = ["swh", "mwp"]
    indices = _fake_indices(names)
    shapes = [(2,), (1, 2), (2, 1), (1, 1, 2)]  # all contain exactly 2 elements

    for shape in shapes:
        base = np.array([1.0, 2.5]).reshape(shape)
        loss = _FakeLoss(base)

        handler = _ListHandler()
        old_level = LOGGER.level
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.addHandler(handler)
        try:
            print_variable_scaling(loss, indices)
        finally:
            LOGGER.removeHandler(handler)
            LOGGER.setLevel(old_level)

        msgs = [m for m in handler.messages if m.startswith("Final Variable Scaling: ")]
        assert msgs, f"No log captured for shape {shape}"
        assert _count_logged_pairs(msgs[-1]) == 2, f"Expected 2 pairs for shape {shape}"
