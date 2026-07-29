# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Legacy edge-materializing graph scores retained for equivalence tests.

This package is intentionally not exported from :mod:`anemoi.training.losses`.
Remove it together with ``test_graph_scores_legacy_equivalence.py`` once the
CSR migration no longer needs an implementation-level oracle.
"""

from .graph_edge_crps import LegacyGraphEdgeCRPSLoss
from .graph_edge_energy_score import LegacyGraphEdgeEnergyScoreLoss
from .graph_energy_score import LegacyGraphEnergyScoreLoss
from .graph_variogram_score import LegacyGraphVariogramScoreLoss

__all__ = [
    "LegacyGraphEdgeCRPSLoss",
    "LegacyGraphEdgeEnergyScoreLoss",
    "LegacyGraphEnergyScoreLoss",
    "LegacyGraphVariogramScoreLoss",
]
