# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationContext
from anemoi.models.migrations import MigrationMetadata

import logging
import torch
from typing import Any
import yaml

LOG = logging.getLogger(__name__)

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
    },
)
# <-- END DO NOT CHANGE


def migrate_setup(context: MigrationContext) -> None:
    """Migrate setup callback to be run before loading the checkpoint.

    Parameters
    ----------
    context : MigrationContext
       A MigrationContext instance
    """


def develop_mapping(
    model_config: dict[str, Any], chunks: int = 0, blocks: int = 0
) -> dict[str, str]:
    mapping = {}
    for chk_idx in range(chunks):
        for block_idx in range(blocks):
            for k, v in model_config.items():
                kproc = ".".join([str(chk_idx), "blocks", str(block_idx), k])
                vproc = ".".join([str(chk_idx), "blocks", str(block_idx), v])
                mapping[kproc] = vproc
    return mapping


def get_chunks_and_blocks(
    model_config: dict[str, Any], state_dict: dict[str, Any], module: str
) -> dict[str, int]:
    prefix = model_config["prefixes"][module]
    names = [k.split(".") for k in state_dict if str(k).beginswith(prefix)]
    prefix_kw_length = len(prefix.split("."))

    chunk_idxs = {int(name[prefix_kw_length + 2]) for name in names}
    block_idxs = {int(name[prefix_kw_length + 2]) for name in names}

    return {"chunks": max(chunk_idxs), "blocks": max(block_idxs)}


def migrate(ckpt: CkptType) -> CkptType:
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
    with open("graphtransformer_1760709058.yaml") as f:
        ckptconfig = yaml.safe_load(f)

    state_dict = ckpt["state_dict"]

    for part in ckptconfig["prefixes"]:
        prefix = ckptconfig["prefixes"][part]

        if part != "processor":
            mapping = ckptconfig["old_to_new_mapping"][part]
        else:
            components = get_chunks_and_blocks(ckptconfig, state_dict, "processor")
            mapping = develop_mapping(
                ckptconfig["old_to_new_mapping"][part],
                chunks=components["chunks"],
                blocks=components["blocks"],
            )

        # performing simple name changes
        for k in mapping:
            LOG.debug(f"Mapping {k}")
            old_bias = ".".join([prefix, k, "bias"])
            old_weight = ".".join([prefix, k, "weight"])
            new_bias = ".".join([prefix, mapping[k], "bias"])
            new_weight = ".".join([prefix, mapping[k], "weight"])

            state_dict[new_bias] = state_dict.pop(old_bias)
            state_dict[new_weight] = state_dict.pop(old_weight)

        # new keys corresponding to no data in old checkpoint
        additions = ckptconfig["add_in_new"]["part"]

        for k in additions:
            new_bias = ".".join([prefix, k, "bias"])
            new_weight = ".".join([prefix, k, "weight"])
            LOG.warning(
                f"Adding {new_bias, new_weight} without supporting tensor (new key) : setting 0-tensor"
            )
            state_dict[new_bias] = torch.tensor([0], dtype=torch.float32)
            state_dict[new_weight] = torch.tensor([0], dtype=torch.float32)

        repeats = ckptconfig["repeats"][part]

        # code copies (different names, same data)
        for k in repeats:
            source_bias = ".".join([prefix, k, "bias"])
            source_weight = ".".join([prefix, k, "weight"])
            new_bias = ".".join([prefix, mapping[k], "bias"])
            new_weight = ".".join([prefix, mapping[k], "weight"])
            LOG.warning(
                f"Copying {new_bias, new_weight} from {source_bias, source_weight}"
            )
            state_dict[new_bias] = state_dict[source_bias]
            state_dict[new_weight] = state_dict[source_weight]

    return ckpt


def rollback(ckpt: CkptType) -> CkptType:
    """Rollback the checkpoint.


    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The rollbacked checkpoint dict.
    """
    return ckpt
