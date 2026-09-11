# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-batch metadata carried next to the tensors of a batch.

A dataset may attach metadata to a sample under the reserved key
:data:`META_KEY`: ``{"data": tensor, "__meta__": {"participant": "h1"}}``.
``default_collate`` leaves strings alone, so after collation the batch holds
``{"__meta__": {"participant": ["h1", "h1", ...]}}`` (one entry per sample).

The training module strips the metadata off before anything else looks at the
batch (:func:`split_meta` in ``on_after_batch_transfer``), so the rest of the
training loop keeps seeing a batch keyed by dataset name only.

This is a seam, designed to be replaced by ``Batch.metadata`` once a rich batch
type lands.
"""

from typing import Any

META_KEY = "__meta__"
"""Reserved batch key holding the metadata mapping."""

PARTICIPANT_FIELD = "participant"
"""Metadata field naming the participant the batch was sampled from."""

BatchMeta = dict[str, Any]


def split_meta(batch: Any) -> tuple[Any, BatchMeta | None]:
    """Return ``batch`` without :data:`META_KEY` and the metadata (``None`` if absent).

    The input is not mutated; a batch that is not a dict is returned unchanged.
    """
    if not isinstance(batch, dict) or META_KEY not in batch:
        return batch, None
    meta = batch[META_KEY]
    return {key: value for key, value in batch.items() if key != META_KEY}, meta


def meta_participant(meta: BatchMeta | None) -> str | None:
    """Return the single participant a batch was sampled from (``None`` if not recorded).

    Raises
    ------
    ValueError
        If the collated batch holds samples of several participants: every
        batch must be participant-pure.
    """
    if meta is None or PARTICIPANT_FIELD not in meta:
        return None
    values = meta[PARTICIPANT_FIELD]
    if isinstance(values, str):  # a single, uncollated sample
        return values
    unique = set(values)
    if len(unique) != 1:
        msg = f"Batch mixes participants {sorted(unique)}; every batch must hold samples of one participant."
        raise ValueError(msg)
    return next(iter(unique))
