# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

LOGGER = logging.getLogger(__name__)


def compute_relative_date_indices(config, val_rollout: int = 1) -> list:
    """Compute the list of relative time indices to load for each batch.

    Accounts for rollout configuration and explicit time overrides.
    """
    if hasattr(config.training, "explicit_times"):
        return sorted(set(config.training.explicit_times.input + config.training.explicit_times.target))

    rollout_cfg = getattr(getattr(config, "training", None), "rollout", None)
    rollout_max = getattr(rollout_cfg, "max", None)
    rollout_start = getattr(rollout_cfg, "start", 1)
    rollout_epoch_increment = getattr(rollout_cfg, "epoch_increment", 0)

    rollout_value = rollout_start
    if rollout_cfg and rollout_epoch_increment > 0 and rollout_max is not None:
        rollout_value = rollout_max
    else:
        LOGGER.warning("Falling back rollout to: %s", rollout_value)

    rollout = max(rollout_value, val_rollout)
    n_step_input = config.training.multistep_input
    n_step_output = config.training.multistep_output
    return list(range(n_step_input + rollout * n_step_output))
