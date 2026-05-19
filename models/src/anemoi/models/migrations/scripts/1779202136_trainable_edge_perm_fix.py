# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch

from anemoi.models.layers.graph_provider import StaticGraphProvider
from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationMetadata

LOGGER = logging.getLogger(__name__)

_TRAINABLE_SUFFIX = ".trainable.trainable"

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
    },
)
# <-- END DO NOT CHANGE


def migrate(ckpt: CkptType, model: torch.nn.Module | None = None) -> CkptType:
    """Migrate the checkpoint.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.
    """

    if model is None:
        LOGGER.info("Skipping trainable edge permutation migration because no model was provided.")
        return ckpt

    state_dict = ckpt["state_dict"]

    for key in [k for k in list(state_dict.keys()) if "graph_provider" in k and k.endswith(_TRAINABLE_SUFFIX)]:
        provider_path = key[: -len(_TRAINABLE_SUFFIX)]

        try:
            graph_provider = model.get_submodule(provider_path)
        except AttributeError:
            LOGGER.debug("Skipping missing graph provider %s while migrating %s", provider_path, key)
            continue

        if not isinstance(graph_provider, StaticGraphProvider):
            continue

        layout_version_key = f"{provider_path}.{graph_provider._TRAINABLE_LAYOUT_VERSION_KEY}"
        layout_version = state_dict.get(layout_version_key, 0)
        if isinstance(layout_version, torch.Tensor):
            layout_version = int(layout_version.item())
        else:
            layout_version = int(layout_version)

        if layout_version < graph_provider._TRAINABLE_LAYOUT_VERSION:
            LOGGER.info("Permuting legacy trainable edge parameters for %s", provider_path)
            trainable = state_dict[key]
            if trainable.shape[0] != graph_provider.perm.shape[0]:
                msg = (
                    "Cannot permute legacy graph-provider trainable tensor for "
                    f"{provider_path}: expected first dimension {graph_provider.perm.shape[0]}, "
                    f"got {trainable.shape[0]}."
                )
                raise RuntimeError(msg)

            state_dict[key] = trainable.index_select(0, graph_provider.perm.to(device=trainable.device))

        state_dict[layout_version_key] = graph_provider.trainable_layout_version.clone()

    return ckpt
