# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

from __future__ import annotations

import datetime

from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import as_timedelta


def _parse_offsets(offsets: list[str | datetime.timedelta], name: str) -> list[datetime.timedelta]:
    if not offsets:
        raise ValueError(f"{name} must contain at least one physical time offset.")

    parsed = []
    for value in offsets:
        if isinstance(value, datetime.timedelta):
            parsed.append(value)
            continue
        text = str(value).strip()
        sign = -1 if text.startswith("-") else 1
        unsigned = text[1:] if text[:1] in "+-" else text
        parsed.append(sign * as_timedelta(unsigned))
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate offsets: {offsets!r}.")
    return parsed


class FixedOffsetsTask(BaseTask):
    """Task with explicitly configured physical input and output offsets."""

    name = "fixed-offsets"

    def __init__(
        self,
        input_offsets: list[str | datetime.timedelta],
        output_offsets: list[str | datetime.timedelta],
        **_kwargs,
    ) -> None:
        self._configured_input_offsets = _parse_offsets(input_offsets, "input_offsets")
        self._configured_output_offsets = _parse_offsets(output_offsets, "output_offsets")
        super().__init__(
            input_offsets=self._configured_input_offsets,
            output_offsets=self._configured_output_offsets,
            preserve_order=True,
        )

    def _get_timestep_for_metadata(self) -> str:
        """Return a compatibility value; exact physical offsets are recorded too."""
        return "fixed-offsets"
