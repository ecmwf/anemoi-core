# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import ListConfig

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
    },
)
# <-- END DO NOT CHANGE

_PROVIDER = "model.model.processor_graph_provider."


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    The processor graph provider of hierarchical models (``hidden_nodes_name`` is a list) is a
    ``ModuleDict`` keyed by hidden mesh name, like in the other models. It holds one provider, for
    the last hidden level, where the main processor runs, so its parameters move one level down
    under that level's name. The level processors' providers
    (``down_level_processor_graph_providers``, ``up_level_processor_graph_providers``) already
    were keyed by level and are left untouched, as are checkpoints of the other models. Keys
    already under a hidden level name are kept as they are.

    Before: ``model.model.processor_graph_provider.trainable.trainable``
    After:  ``model.model.processor_graph_provider.<last hidden level>.trainable.trainable``

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """
    hidden_nodes_name = ckpt["hyper_parameters"]["config"].model.model.hidden_nodes_name
    if not isinstance(hidden_nodes_name, (list, ListConfig)):
        return ckpt

    # The levels the model was built with: in order, each named once.
    levels = list(dict.fromkeys(hidden_nodes_name))
    state_dict = ckpt["state_dict"]
    renames = {
        key: f"{_PROVIDER}{levels[-1]}.{key.removeprefix(_PROVIDER)}"
        for key in state_dict
        if key.startswith(_PROVIDER) and key.removeprefix(_PROVIDER).split(".")[0] not in levels
    }
    for old_key, new_key in renames.items():
        state_dict[new_key] = state_dict.pop(old_key)
    return ckpt
