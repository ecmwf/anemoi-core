# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Callable
from typing import Any

import torch

from anemoi.models.data.sources import Source


def apply_pairwise(pred: Source, target: Source, func: Callable, *args: Any, **kwargs) -> torch.Tensor:
    """Apply a tensor-level function to two aligned source views (see :meth:`Source.pairwise`)."""
    if not isinstance(pred, Source):
        raise TypeError(f"Pairwise losses do not support source type {type(pred).__name__}.")
    return pred.pairwise(target, func, *args, **kwargs)
