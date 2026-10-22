# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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

_PROVIDER = ".processor_graph_provider."


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    The processor graph provider is now always a ``ModuleDict`` keyed by hidden
    mesh name. Models with a single hidden mesh (``hidden_nodes_name`` is a
    string) used to hold one provider directly, so its parameters move one level
    down. Models with a dataset mapping already had this layout, and hierarchical
    models (``hidden_nodes_name`` is a list) are left untouched.

    Before: ``*.processor_graph_provider.trainable.trainable``
    After:  ``*.processor_graph_provider.<hidden_nodes_name>.trainable.trainable``

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """
    hidden_nodes_name = ckpt["hyper_parameters"]["config"].model.hidden_nodes_name
    if not isinstance(hidden_nodes_name, str):
        return ckpt

    state_dict = ckpt["state_dict"]
    renames = {
        key: key.replace(_PROVIDER, f"{_PROVIDER}{hidden_nodes_name}.", 1) for key in state_dict if _PROVIDER in key
    }
    for old_key, new_key in renames.items():
        state_dict[new_key] = state_dict.pop(old_key)
    return ckpt
