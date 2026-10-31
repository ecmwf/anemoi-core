# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.forcings import Forcings
from anemoi.inference.runner import Runner
from anemoi.inference.runner import RunnerClasses
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.runners.simple import SimpleTensorHandler

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

# `SimpleRunner` is the low-level API: it loads *computed* forcings (insolation, julian day, ...)
# but refuses to load *coupled* ones, which come from data.
#
# The runner carries a forcing through the rollout only if the metadata marks it
# `constant_in_time`. In this checkpoint the static fields lsm, z, slor and sdor carry no such
# flag - their variables_metadata holds only `mars` and `units` - so the runner treats them as
# time-varying coupled forcings. `SimpleTensorHandler` declines to supply those, nothing refills
# their slots after the first model call, and the rollout dies with
# "Missing variables in input tensor: ['lsm', 'sdor', 'slor', 'z']".
#
# These fields really are constant, so holding them at their analysis values is exact, not an
# approximation. That is only true because they are static; see the warning below.


class PersistedForcings(Forcings):
    """Repeats the initial value of a dynamic coupled forcing at every step."""

    def __init__(self, context, variables, mask):
        super().__init__(context)
        self.variables = list(variables)
        self.mask = mask
        self.kinds = {"persisted": True}

    def load_forcings_array(self, dates, current_state):
        # `current_state` only holds what the model output, so the values come from the snapshot
        # the script takes of the input state before the forecast starts.
        source = self.context.context.persisted_forcings
        missing = [v for v in self.variables if v not in source]
        if missing:
            raise ValueError(
                f"No initial values to persist for {sorted(missing)}. These are dynamic forcings "
                "that the model needs at every step; add them to the input state.",
            )
        # float32 to match the input tensor, as ComputedForcings does: `add_dynamic_forcings_to_
        # input_tensor` copies straight into it without casting, and open data arrives as float64.
        values = np.stack([np.asarray(source[v], dtype=np.float32) for v in self.variables])  # (variables, values)
        return np.repeat(values[:, np.newaxis, :], len(dates), axis=1)


# Fields that are physically static, so persisting them introduces no error even when the
# checkpoint metadata forgets to mark them constant_in_time.
STATIC_FORCINGS = {"lsm", "z", "slor", "sdor", "sdfor", "cl", "cvl", "cvh", "tvl", "tvh", "anor", "isor"}


class PersistingTensorHandler(SimpleTensorHandler):
    def create_dynamic_coupled_forcings(self, variables, mask):
        if len(mask) == 0:
            return []
        varying = sorted(set(variables) - STATIC_FORCINGS)
        LOGGER.warning("Holding these forcings at their analysis values for the whole rollout: %s", sorted(variables))
        if varying:
            LOGGER.warning(
                "  %s are not known to be static, so persisting them is an approximation that grows with lead time.",
                varying,
            )
        return [PersistedForcings(self, variables, mask)]


class PersistingRunner(SimpleRunner):
    """SimpleRunner that persists dynamic coupled forcings instead of failing on them."""

    def __init__(self, checkpoint, **kwargs):
        self.persisted_forcings = {}
        config = RunConfiguration(
            checkpoint=checkpoint,
            input=kwargs.pop("input", "empty"),
            output=kwargs.pop("output", "none"),
            **kwargs,
        )
        # Skip SimpleRunner.__init__, which hard-codes the handler that refuses coupled forcings.
        Runner.__init__(self, config, classes=RunnerClasses(tensor_handler=PersistingTensorHandler))
