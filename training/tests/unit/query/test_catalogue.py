# (C) Copyright 2026 Anemoi contributors.

"""Tests for physical field semantics recovered by the query catalogue."""

from anemoi.training.query.catalogue import _implicit_pressure_level


def test_implicit_pressure_level_from_atmospheric_field_name() -> None:
    assert _implicit_pressure_level("t_500", "t", None, None) == 500
    assert _implicit_pressure_level("q_850", "q", None, None) == 850


def test_implicit_pressure_level_does_not_override_declared_metadata() -> None:
    assert _implicit_pressure_level("t_500", "t", "sfc", None) is None
    assert _implicit_pressure_level("t_500", "t", None, 700) is None


def test_implicit_pressure_level_rejects_non_pressure_parameters() -> None:
    assert _implicit_pressure_level("tp_6", "tp", None, None) is None
    assert _implicit_pressure_level("100u", "u", None, None) is None
