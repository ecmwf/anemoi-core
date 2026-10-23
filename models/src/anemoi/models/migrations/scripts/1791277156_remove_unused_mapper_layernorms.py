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


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    ``TransformerMapperBlock`` used to carry two layer norms it never used,
    ``layer_norm_attention`` and ``layer_norm_mlp``, and called its MLP layer
    norm ``layer_norm_mpl``. This removes the two unused norms and renames
    ``layer_norm_mpl`` to ``layer_norm_mlp``.

    Before: ``*.proc.layer_norm_attention.weight`` (removed)
            ``*.proc.layer_norm_mlp.weight``       (removed)
            ``*.proc.layer_norm_mpl.weight``
    After:  ``*.proc.layer_norm_mlp.weight``

    Only blocks that have a ``layer_norm_mpl`` are touched, so the
    ``layer_norm_attention`` and ``layer_norm_mlp`` of processor blocks are
    kept.

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

    # Every key with a ".layer_norm_mpl." belongs to a transformer mapper block;
    # the part in front of it is the path to that block.
    mapper_blocks = {k.split(".layer_norm_mpl.")[0] for k in state_dict if ".layer_norm_mpl." in k}
    if not mapper_blocks:
        return ckpt

    # Drop the unused norms first so the renamed MLP norm does not collide with them.
    unused_prefixes = tuple(
        f"{block}.{name}." for block in mapper_blocks for name in ("layer_norm_attention", "layer_norm_mlp")
    )
    for key in [k for k in state_dict if k.startswith(unused_prefixes)]:
        del state_dict[key]

    renames = {k: k.replace(".layer_norm_mpl.", ".layer_norm_mlp.") for k in state_dict if ".layer_norm_mpl." in k}
    for old_key, new_key in renames.items():
        state_dict[new_key] = state_dict.pop(old_key)
    return ckpt
