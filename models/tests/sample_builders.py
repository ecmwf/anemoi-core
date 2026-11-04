# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Builders for per-sample payloads in tests, as a data reader returns them.

Both builders go through :func:`~anemoi.models.data.sample.create_batched_struct`, the
path the readers and :meth:`AnemoiModelInterface.predict_step` take, so the samples are
validated and their coordinates converted from degrees. Pass ``data=None`` (plus the
template sizes) to build a template instead.

``n_points`` and ``n_variables`` shape the defaults. Any other keyword is passed to
:func:`create_batched_struct` as is and replaces the default, after the defaults are
built: ``variables=["a"]`` keeps data with ``n_variables`` channels, which is how the
validation tests build inconsistent payloads.

Use :func:`~tests.batch_builders.build_batch` instead for an already-collated batch.
"""

from typing import Any

import torch

from anemoi.models.data import create_batched_struct
from anemoi.models.data.sample import BaseSample
from anemoi.models.data.sources import BaseTemplate

GRIDDED_AXES = ("time", "ensemble", "grid", "variables")
TABULAR_AXES = ("ensemble", "grid", "variables")

#: Seconds between the reference time and each successive tabular time window.
WINDOW_SPACING = 3600.0


def latitudes(n_points: int) -> list[float]:
    """Return ``n_points`` latitudes in degrees, evenly spaced from 0 to 90."""
    return [90.0 * i / max(n_points - 1, 1) for i in range(n_points)]


def longitudes(n_points: int) -> list[float]:
    """Return ``n_points`` longitudes in degrees, evenly spaced from 0 to 180."""
    return [180.0 * i / max(n_points - 1, 1) for i in range(n_points)]


def variable_names(n_variables: int) -> list[str]:
    """Return the default variable names ``a``, ``b``, ``c``, ..."""
    return [chr(ord("a") + i) for i in range(n_variables)]


def statistics(n_variables: int) -> dict[str, torch.Tensor]:
    """Return a ``mean`` statistic of ``0, 1, ...`` so selections can be traced by value."""
    return {"mean": torch.arange(n_variables, dtype=torch.float32)}


def gridded_sample(
    *,
    n_points: int = 4,
    n_variables: int = 3,
    n_times: int = 2,
    **overrides: Any,
) -> BaseSample | BaseTemplate:
    """Build a gridded sample of shape ``(n_times, 1, n_points, n_variables)``, data ``0, 1, ...``."""
    kwargs = {
        "data_type": "gridded",
        "data": torch.arange(n_times * n_points * n_variables, dtype=torch.float32).reshape(
            n_times, 1, n_points, n_variables
        ),
        "variables": variable_names(n_variables),
        "layout": GRIDDED_AXES,
        "latitudes": latitudes(n_points),
        "longitudes": longitudes(n_points),
        "statistics": statistics(n_variables),
    }
    return create_batched_struct(**(kwargs | overrides))


def tabular_sample(
    *,
    n_points: int = 4,
    n_variables: int = 2,
    **overrides: Any,
) -> BaseSample | BaseTemplate:
    """Build a tabular sample of shape ``(1, n_points, n_variables)``, data ``0, 1, ...``.

    The points are split into two time windows, the first holding ``ceil(n_points / 2)``,
    and every point's timedelta is its window's offset: ``0`` and :data:`WINDOW_SPACING`.
    """
    split = (n_points + 1) // 2
    kwargs = {
        "data_type": "tabular",
        "data": torch.arange(n_points * n_variables, dtype=torch.float32).reshape(1, n_points, n_variables),
        "variables": variable_names(n_variables),
        "layout": TABULAR_AXES,
        "latitudes": latitudes(n_points),
        "longitudes": longitudes(n_points),
        "statistics": statistics(n_variables),
        "timedeltas": [0.0] * split + [WINDOW_SPACING] * (n_points - split),
        "boundaries": [(0, split), (split, n_points)],
    }
    return create_batched_struct(**(kwargs | overrides))
