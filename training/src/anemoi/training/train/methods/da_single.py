# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Deterministic training method with data-assimilation cycling and correctors.

Pairs with :class:`anemoi.training.tasks.da_forecaster.DAForecaster`. The DA cycling,
corrector and loss-target machinery lives in
:class:`anemoi.training.train.methods.da_mixin.DATrainingMixin`; this class is the
deterministic binding of it, and takes the mixin's hooks at their defaults.

See :class:`anemoi.training.train.methods.da_ensemble.DAEnsembleTraining` for the
kCRPS-ensemble binding.
"""

from anemoi.training.train.methods.da_mixin import DATrainingMixin
from anemoi.training.train.methods.single import SingleTraining


class DASingleTraining(DATrainingMixin, SingleTraining):
    """Deterministic training method for the DA forecaster with correctors."""
