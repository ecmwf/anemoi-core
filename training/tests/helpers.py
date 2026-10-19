# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Plain helpers that more than one test module uses."""

from __future__ import annotations

_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
_LOADERS = "anemoi.training.checkpoint.loading.strategies"


class FakeIndexCollection:
    """Picklable stand-in for ``IndexCollection``. It has only ``name_to_index``."""

    def __init__(self, name_to_index: dict[str, int] | None = None) -> None:
        self.name_to_index = dict(name_to_index or {})


def resume_checkpoint_cfg(run_id: str, loading: str | None = "WarmStartLoader", *, fork: bool = False) -> dict:
    """Return a ``training.checkpoint`` block that starts from the ``last.ckpt`` of run ``run_id``.

    ``loading`` is the name of a class in ``checkpoint.loading.strategies``. If it is ``None``,
    there is no loading block, which is also a resume.
    """
    checkpoint: dict = {"source": {"_target_": _RUN_SOURCE, "run_id": run_id, "fork": fork}}
    if loading is not None:
        checkpoint["loading"] = {"_target_": f"{_LOADERS}.{loading}"}
    return checkpoint
