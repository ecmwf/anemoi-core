# (C) Copyright 2025 Anemoi contributors.
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


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    ``GraphTransformerBaseBlock.node_dst_mlp`` and
    ``GraphTransformerMapperBlock.node_src_mlp`` were bare ``nn.Sequential``
    modules and are now ``MLP`` instances whose layers live under an inner
    ``.mlp`` sequential. This inserts the extra path component.

    Before: ``*.node_dst_mlp.0.weight``
    After:  ``*.node_dst_mlp.mlp.0.weight``

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """
    state_dict = ckpt["state_dict"]
    renames = {
        k: k.replace(".node_dst_mlp.", ".node_dst_mlp.mlp.").replace(".node_src_mlp.", ".node_src_mlp.mlp.")
        for k in list(state_dict)
        if ".node_dst_mlp." in k or ".node_src_mlp." in k
    }
    for old_key, new_key in renames.items():
        state_dict[new_key] = state_dict.pop(old_key)
    return ckpt
