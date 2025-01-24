# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from __future__ import annotations

import logging
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Field

LOGGER = logging.getLogger(__name__)


class RemoveUnconnectedNodesSchema(BaseModel):
    target_: Literal["anemoi.graphs.processors.RemoveUnconnectedNodes"] = Field(..., alias="_target_")
    "Post processor to remove unconnected nodes."
    nodes_name: str
    "Nodes from which to remove the unconnected nodes."
    ignore: str = Field(default=None)
    "Attribute name of nodes to be ignored."
    save_mask_indices_to_attr: str = Field(default=None)
    "New attribute name to store the mask indices."


ProcessorSchema = Union[RemoveUnconnectedNodesSchema]
