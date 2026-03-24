# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import torch

from anemoi.models.data_indices.tensor import BaseTensorIndex


class DownscalingBaseTensorIndex(BaseTensorIndex):
    """Indexing for variables in index as Tensor."""

    def __init__(self, *, includes: list[str], excludes: list[str], name_to_index: dict[str, int]):
        super().__init__(includes=includes, excludes=excludes, name_to_index=name_to_index)

    def _build_idx_from_includes(self, includes=None) -> "torch.Tensor[int]":
        if includes is None:
            includes = self.includes
        return torch.Tensor(sorted(self.name_to_index[name] for name in includes if name in self.name_to_index)).to(
            torch.int
        )


class InputTensorIndex(DownscalingBaseTensorIndex):
    """Indexing for input variables."""

    def __init__(self, *, includes: list[str], excludes: list[str], name_to_index: dict[str, int]):
        super().__init__(includes=includes, excludes=excludes, name_to_index=name_to_index)
        self.forcing = self._only
        self.diagnostic = self._removed


class OutputTensorIndex(DownscalingBaseTensorIndex):
    """Indexing for output variables."""

    def __init__(self, *, includes: list[str], excludes: list[str], name_to_index: dict[str, int]):
        super().__init__(includes=includes, excludes=excludes, name_to_index=name_to_index)
        self.forcing = self._removed
        self.diagnostic = self._only
