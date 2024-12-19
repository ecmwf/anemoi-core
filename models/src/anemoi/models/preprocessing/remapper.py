# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from typing import Optional

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.monomapper import Monomapper
from anemoi.models.preprocessing.multimapper import Multimapper

LOGGER = logging.getLogger(__name__)


class Remapper(BasePreprocessor, ABC):
    """Remap and convert variables for single variables."""

    def __new__(
        cls,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        _, _, method_config = cls._process_config(config)
        monomappings = Monomapper.supported_methods
        multimappings = Multimapper.supported_methods
        if all(method in monomappings for method in method_config):
            return Monomapper(config, data_indices, statistics)
        elif all(method in multimappings for method in method_config):
            return Multimapper(config, data_indices, statistics)
        elif not (
            any(method in monomappings for method in method_config)
            or any(method in multimappings for method in method_config)
        ):
            raise ValueError("No valid remapping method found.")
        else:
            raise NotImplementedError(
                f"Not implemented: method_config contains a mix of monomapper and multimapper methods: {list(method_config.keys())}"
            )
