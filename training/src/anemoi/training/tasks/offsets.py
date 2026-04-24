# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging

from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


def _parse_offset(s: str) -> datetime.timedelta:
    """Parse a duration string into a ``timedelta``, with optional leading ``-``."""
    s = s.strip()
    if s.startswith("-"):
        return -frequency_to_timedelta(s[1:])
    return frequency_to_timedelta(s)


def _to_timedeltas(values: list[datetime.timedelta | str]) -> list[datetime.timedelta]:
    """Coerce a list of values to ``timedelta``, parsing strings via ``_parse_offset``."""
    return [v if isinstance(v, datetime.timedelta) else _parse_offset(v) for v in values]


class BaseTaskOffsets:
    """Collects a task's input and output time offsets.

    Attributes
    ----------
    input : list[datetime.timedelta]
        Sorted list of input time offsets.
    output : list[datetime.timedelta]
        Sorted list of output time offsets.
    """

    def __init__(
        self,
        input_offsets: list[datetime.timedelta | str],
        output_offsets: list[datetime.timedelta | str],
    ) -> None:
        self.input = sorted(_to_timedeltas(input_offsets))
        self.output = sorted(_to_timedeltas(output_offsets))

    @property
    def all(self) -> list[datetime.timedelta]:
        """Sorted union of input and output offsets."""
        return sorted(set(self.input + self.output))


class ForecastOffsets(BaseTaskOffsets):
    """Task offsets for autoregressive forecasting tasks.

    Includes step_shift, and its validation, for rollout advancement.

    Parameters
    ----------
    input_offsets, output_offsets :
        Forwarded to :class:`BaseTaskOffsets`.
    step_shift :
        Shift performed per rollout step.
    """

    def __init__(
        self,
        input_offsets: list[datetime.timedelta | str],
        output_offsets: list[datetime.timedelta | str],
        step_shift: datetime.timedelta | str | None = None,
    ) -> None:
        super().__init__(input_offsets, output_offsets)
        if isinstance(step_shift, str):
            step_shift = frequency_to_timedelta(step_shift)
        self.step_shift = self._validate_shift(step_shift)

    def _validate_shift(self, step_shift: datetime.timedelta | None) -> datetime.timedelta:
        """Return a validated step-shift timedelta.

        A shift S is valid if it is strictly positive and
        when the shifted input is included in the union of input and output.
        None gets replaced by the maximum valid shift, if it exists.

        Parameters
        ----------
        step_shift :
            Explicit rollout step shift to validate, or ``None`` to infer the largest
            valid shift automatically.

        Raises
        ------
        ValueError
            If the explicit shift is invalid, or if no valid shift exists.
        """
        candidates = {x - i for x in self.all for i in self.input if x > i}
        valid = sorted(s for s in candidates if all(i + s in self.all for i in self.input))

        if step_shift is None:
            if not valid:
                msg = (
                    "No valid autoregressive rollout step_shift exists. "
                    "Input offsets and output offsets are incompatible for a forecasting task. "
                    f"input_offsets={self.input}, output_offsets={self.output}"
                )
                raise ValueError(msg)
            LOGGER.info("Inferred step_shift=%s (maximum valid shift).", valid[-1])
            return valid[-1]

        if step_shift not in valid:
            msg = (
                f"step_shift={step_shift!r} is not a valid autoregressive rollout shift "
                "for the chosen input and output offsets."
                f" (valid shifts are: {valid}). "
                f"input_offsets={self.input}, output_offsets={self.output}"
            )
            raise ValueError(msg)
        return step_shift

    def slot_mapping(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Pre-compute slot mappings for autoregressive input advancement during rollout.

        For each position in the next input window (``input + step_shift``),
        determine whether the data comes from a model prediction (output slot)
        or is preserved unchanged from the current input (input slot).
        Output takes priority when a slot falls in both.

        Returns
        -------
        preserve : list[tuple[int, int]]
            ``(new_slot, old_slot)`` pairs to copy unchanged from current input.
        predict : list[tuple[int, int]]
            ``(new_slot, output_slot)`` pairs to fill from model predictions.
        """
        output_index = {o: j for j, o in enumerate(self.output)}
        input_index = {inp: j for j, inp in enumerate(self.input)}
        preserve: list[tuple[int, int]] = []
        predict: list[tuple[int, int]] = []
        for new_slot, i in enumerate(self.input):
            src = i + self.step_shift
            if src in output_index:
                predict.append((new_slot, output_index[src]))
            else:
                preserve.append((new_slot, input_index[src]))
        return preserve, predict
