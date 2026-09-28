# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.data.layout import TensorLayout

from .base import Source
from .gridded import GriddedSource
from .tabular import TabularSource

__all__ = [
    "GriddedSource",
    "TabularSource",
    "Source",
    "make_source",
]


def make_source(layout: TensorLayout, **kwargs) -> Source:
    """Build a source of the kind ``layout`` describes.

    >>> make_source(name="era5", variables=["t"], layout=layout, data=x, coordinates=coords)
    """
    if layout.time_in_grid:
        return TabularSource(layout=layout, **kwargs)

    return GriddedSource(layout=layout, **kwargs)
