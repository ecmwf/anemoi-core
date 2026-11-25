# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Component adapters.

Each versioned component of the checkpoint metadata has its own module of
adapters.  Importing this package imports every component module, which
registers its adapters with :class:`~anemoi.metadata.registry.ComponentRegistry`.

Data and task adapters vary on two axes, the layout version and a name (the
dataset's ``data_type``, the task name).  To add a component version of either:

1. In the component's module, subclass the nearest existing version reader
   (``DataReaderV1_0``, ``TaskReaderV1_0``) and override only what differs.
2. Register it with ``register_data_version(version, reader)`` or
   ``register_task_version(version, reader)``, which combines it with every
   behaviour.
3. Add a golden fixture and bump the stamp written for the component.

The lineage component has one adapter per version, registered with
``ComponentRegistry.register("lineage", version)``.  A value that older
component versions do not record and cannot fall back on is gated with
:func:`~anemoi.metadata.components.base.requires`.

Each component module also provides a ``build_*`` function that selects and
builds its adapters from the raw dictionary and the component version
(``build_task_adapter``, ``build_data_adapters``, ``build_lineage_adapter``);
``CheckpointAdapter`` only calls them.  A new component module provides one
too, and must be imported below.
"""

from . import data
from . import legacy
from . import lineage
from . import task
from .base import ComponentAdapter
from .base import requires

__all__ = ["ComponentAdapter", "data", "legacy", "lineage", "requires", "task"]
