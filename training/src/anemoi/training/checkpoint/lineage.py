# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Checkpoint load lineage stored on the training metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


def record_checkpoint_lineage(module: nn.Module, checkpoint: dict[str, Any], loading_strategy: str) -> None:
    """Append one ckpt load to ``module.hparams['metadata']['checkpoint_lineage']``.

    The inherited list is the loaded file's ``hyper_parameters.metadata.checkpoint_lineage``.
    A missing list, including every checkpoint saved before lineage was recorded, is ``[]``.
    ``parent_uuid`` is that file's ``uuid``, or ``None`` when the file has none. It is also
    stored as the scalar ``metadata['parent_uuid']``, the immediate parent of this run.

    Parameters
    ----------
    module : nn.Module
        The training module. ``hparams['metadata']`` is the dict written into the sidecar.
    checkpoint : dict
        The loaded checkpoint.
    loading_strategy : str
        How this file was applied (``warm_start``, ``weights_only``, ``transfer_learning``, ``cold_start``).

    Examples
    --------
    After a warm start from ``0dc0e962-102b-47dd-9f35-2e3dbd285b9a`` and a cold start from
    ``0e761c2f-2224-4450-a2f1-623f26b618c7``::

        metadata["checkpoint_lineage"] == [
            {"loading_strategy": "warm_start", "parent_uuid": "0dc0e962-102b-47dd-9f35-2e3dbd285b9a"},
            {"loading_strategy": "cold_start", "parent_uuid": "0e761c2f-2224-4450-a2f1-623f26b618c7"},
        ]
        metadata["parent_uuid"] == "0e761c2f-2224-4450-a2f1-623f26b618c7"
    """
    # Get existing checkpoint metadata.
    stored = (checkpoint.get("hyper_parameters") or {}).get("metadata") or {}
    inherited = stored.get("checkpoint_lineage", [])
    lineage = [dict(pair) for pair in inherited]

    # Get parent UUID from checkpoint metadata.
    parent_uuid = stored.get("uuid")
    lineage.append({"loading_strategy": loading_strategy, "parent_uuid": parent_uuid})

    # Write ckpt load lineage to the module's metadata.
    metadata = module.hparams["metadata"]
    metadata["checkpoint_lineage"] = lineage
    metadata["parent_uuid"] = parent_uuid  # The immediate parent of this run.
