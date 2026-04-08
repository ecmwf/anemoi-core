# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Re-export shim — graph config utilities have moved to anemoi.graphs."""

from anemoi.graphs.graph_config import expand_projections_into_graph_config

__all__ = ["expand_projections_into_graph_config"]
