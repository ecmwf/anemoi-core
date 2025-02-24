# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.models.data_indices.collection import IndexCollection
from datetime import timedelta


class AnemoiDataHandler:
    dataset: AnemoiDatasetsDataModule  # torch.utils.data.DataLoader
    timestep: timedelta
    forcing: list[str]
    diagnostic: list[str]
    #remapped: dict[str, list[str]]
    name_to_index: dict[str, int]
    processors: list # normalizer, imputer, remapper?

    @property
    def data_indices(self) -> IndexCollection: ...
    # name_to_index, forcing, diagnostic

    @property
    def metadata(self) -> dict: ...

    @property
    def supporting_arrays(self) -> dict: ...

    @property
    def statistics(self) -> dict: ...
