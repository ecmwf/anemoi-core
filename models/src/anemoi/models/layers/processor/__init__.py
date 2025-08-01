# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.models.layers.processor.dynamic import DynamicGraphTransformerProcessor
from anemoi.models.layers.processor.static import GNNProcessor
from anemoi.models.layers.processor.static import GraphTransformerProcessor
from anemoi.models.layers.processor.static import TransformerProcessor

__all__ = ["TransformerProcessor", "GNNProcessor", "GraphTransformerProcessor", "DynamicGraphTransformerProcessor"]
