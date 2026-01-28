# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from torch_geometric.data import HeteroData


class GraphBundle:
    """Bundle of graphs for a dataset.

    Attributes
    ----------
    main : HeteroData
        Primary graph used by the model.
    assets : dict[str, HeteroData]
        Named auxiliary graphs (e.g., loss smoothing, truncation, interpolation).
    """

    def __init__(self, main: HeteroData, assets: dict[str, HeteroData] | None = None) -> None:
        self.main = main
        self.assets = {} if assets is None else assets

    def get(self, name: str | None) -> HeteroData:
        """Return a graph by name (defaults to main)."""
        if name in (None, "", "main"):
            return self.main
        if name not in self.assets:
            raise KeyError(f"Graph asset not found: {name}")
        return self.assets[name]
