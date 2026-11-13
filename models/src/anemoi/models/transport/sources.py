# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Optional

from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.data import Batch


def gaussian_source(templates: Batch, model_comm_group: Optional[ProcessGroup] = None) -> Batch:
    """Draw the Gaussian source field that a transport objective moves from.

    ``templates`` describes the field to build, one source per dataset: structure, per-sample
    node counts, variables, device, dtype and grid shard sizes. Only its payload shapes are
    read, so a zero-stride template payload is enough. For a grid-sharded payload the noise is
    drawn once for the full grid, and each rank of ``model_comm_group`` keeps its own part.
    """
    return templates.with_sources(
        {name: template.randn_like(model_comm_group) for name, template in templates.items()},
    )
