# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Task component adapters.

The task component covers what the training task writes: the task name
(``metadata_inference.task``) and, per dataset, ``timesteps``.  A checkpoint
has one task, so the adapter is checkpoint level.  Training copies the same
``timesteps`` block into every dataset; the adapter reads all of them and
exposes one value when they agree.  :meth:`TaskAdapter.for_dataset` returns the
timing of one dataset, and global access raises :class:`TaskTimingError` when
the datasets disagree.

Adapters are registered by ``(task name, version)``; the unnamed adapter is the
generic one used for task names without an adapter of their own (it warns).
They vary on two axes, the layout version and the task name, so each one is
built from a *version reader* (where the values are stored) and a *task
behaviour* (what the task derives or stores) on top of :class:`TaskAdapter`;
:func:`register_task_version` combines a reader with every behaviour.  A change
for a new component version is one reader that reaches every task name.  The
behaviours are:

* ``forecaster``: offsets, ``rollout_shift`` and ``advance_map`` are derived
  from the timestep and the number of input and output steps;
* ``offset-forecaster``: the same values are stored by training and read as is
  (``rollout_shift`` is the last output offset when not stored);
* ``temporal-downscaler``: boundary inputs at ``0`` and one timestep;
  ``lagged`` looks ahead, from the input explicit times and the data
  frequency.  ``time-interpolator`` is an alias of it (see
  :data:`TASK_NAME_ALIASES`): older checkpoints record temporal downscalers
  under that name, and they are read as ``temporal-downscaler``;
* ``autoencoder`` and ``timeless``: a single snapshot, timestep ``0H``.

The task name is read from ``metadata_inference.task``, else the top-level
``task`` written by training for a while (format F4), else it is
``"forecaster"``, which is what inference assumes.

Component versions:

* ``0.2``: layout of metadata format ``2.0``.
* ``1.0``: layout of metadata format ``3.0`` (same task layout; the stamp
  versions the task block independently of the data component).
"""

import datetime
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import cast

from anemoi.metadata.exceptions import MetadataError
from anemoi.metadata.registry import ComponentRegistry
from anemoi.utils.dates import as_timedelta

from .base import ComponentAdapter

#: Task assumed when the metadata records none (as inference does).
DEFAULT_TASK_NAME = "forecaster"


class TaskTimingError(MetadataError):
    """Raised when a checkpoint-level timing value is requested but datasets disagree."""


def resolve_task_name(raw: Mapping[str, Any]) -> str | None:
    """Return the task name recorded in the raw metadata, or ``None``.

    Reads ``metadata_inference.task``, falling back to the top-level ``task``
    written by training during a short window (format F4).
    """
    inference = raw.get("metadata_inference")
    if isinstance(inference, dict) and isinstance(inference.get("task"), str):
        return inference["task"]
    if isinstance(raw.get("task"), str):
        return raw["task"]
    return None


def effective_task_name(raw: Mapping[str, Any]) -> str:
    """Return the task name to use: the recorded one, else :data:`DEFAULT_TASK_NAME`."""
    return resolve_task_name(raw) or DEFAULT_TASK_NAME


def _timedelta(value: Any) -> datetime.timedelta | None:
    """Convert a frequency string to a timedelta; ``None`` if absent or unreadable."""
    if isinstance(value, datetime.timedelta):
        return value
    if not isinstance(value, str):
        return None
    try:
        return as_timedelta(value)
    except (ValueError, TypeError):
        return None


def _timedelta_list(value: Any) -> list[datetime.timedelta]:
    """Convert a list of frequency strings to a sorted list of timedeltas, dropping unreadable ones."""
    if not isinstance(value, list):
        return []
    return sorted(td for td in map(_timedelta, value) if td is not None)


def _int_list(value: Any) -> list[int]:
    """Return *value* as a list of ints; a bare int (seen in real files) becomes a one-item list."""
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return []


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class TaskTiming:
    """Timing of the task for one dataset.

    Values the task does not define, or that cannot be derived from what the
    metadata records, are ``None``.  Lists and dicts are returned as copies.

    Attributes
    ----------
    timestep : str | None
        Model timestep as recorded (e.g. ``"6h"``; ``"0H"`` for timeless tasks).
    timestep_delta : datetime.timedelta | None
        The same, as a timedelta.
    relative_date_indices_training : list[int]
        Relative date indices of the training batch (may repeat an index).
    input_relative_date_indices, output_relative_date_indices : list[int]
        Relative date indices of the model input and output.  For the offset
        forecaster they are positions in the offsets lists.
    n_step_input, n_step_output : int | None
        Number of input and output steps; ``None`` when the metadata records
        nothing to derive them from.
    lagged : list[datetime.timedelta] | None
        Input times relative to the last input, ascending and not positive:
        ``-s * timestep`` for ``s < n_step_input``; the stored input offsets
        for the offset forecaster.
    input_offsets, output_offsets : list[datetime.timedelta] | None
        Input and output times relative to the last input.
    rollout_shift : datetime.timedelta | None
        Time by which each rollout step advances.
    advance_map : dict[str, list[tuple[int, int]]] | None
        Index mapping that advances input and output into the next input
        (keys ``"inin"`` and ``"outin"``).
    input_explicit_times, target_explicit_times : list[int] | None
        Times of the input and target steps, in timesteps of the data: the
        explicit times of the training config, else the relative date indices.
        For the temporal downscaler and time interpolator ``lagged`` is
        derived from the input times.
    """

    def __init__(
        self,
        *,
        timestep: str | None,
        relative_date_indices_training: list[int],
        input_relative_date_indices: list[int],
        output_relative_date_indices: list[int],
        n_step_input: int | None,
        n_step_output: int | None,
        lagged: list[datetime.timedelta] | None,
        input_offsets: list[datetime.timedelta] | None = None,
        output_offsets: list[datetime.timedelta] | None = None,
        rollout_shift: datetime.timedelta | None = None,
        advance_map: dict[str, list[tuple[int, int]]] | None = None,
        input_explicit_times: list[int] | None = None,
        target_explicit_times: list[int] | None = None,
    ) -> None:
        self._timestep = timestep
        self._training = list(relative_date_indices_training)
        self._input = list(input_relative_date_indices)
        self._output = list(output_relative_date_indices)
        self._n_input = n_step_input
        self._n_output = n_step_output
        self._lagged = None if lagged is None else list(lagged)
        self._input_offsets = None if input_offsets is None else list(input_offsets)
        self._output_offsets = None if output_offsets is None else list(output_offsets)
        self._rollout_shift = rollout_shift
        self._advance_map = None if advance_map is None else {k: list(v) for k, v in advance_map.items()}
        self._input_explicit_times = None if input_explicit_times is None else list(input_explicit_times)
        self._target_explicit_times = None if target_explicit_times is None else list(target_explicit_times)

    def _key(self) -> tuple[Any, ...]:
        advance = (
            None if self._advance_map is None else tuple(sorted((k, tuple(v)) for k, v in self._advance_map.items()))
        )
        return (
            self._timestep,
            tuple(self._training),
            tuple(self._input),
            tuple(self._output),
            self._n_input,
            self._n_output,
            None if self._lagged is None else tuple(self._lagged),
            None if self._input_offsets is None else tuple(self._input_offsets),
            None if self._output_offsets is None else tuple(self._output_offsets),
            self._rollout_shift,
            advance,
            None if self._input_explicit_times is None else tuple(self._input_explicit_times),
            None if self._target_explicit_times is None else tuple(self._target_explicit_times),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TaskTiming) and self._key() == other._key()

    def __repr__(self) -> str:
        return (
            f"TaskTiming(timestep={self._timestep!r}, n_step_input={self._n_input}, n_step_output={self._n_output}, "
            f"input_relative_date_indices={self._input}, output_relative_date_indices={self._output})"
        )

    @property
    def timestep(self) -> str | None:
        """Model timestep as recorded."""
        return self._timestep

    @property
    def timestep_delta(self) -> datetime.timedelta | None:
        """Model timestep as a timedelta."""
        return _timedelta(self._timestep)

    @property
    def relative_date_indices_training(self) -> list[int]:
        """Relative date indices of the training batch."""
        return list(self._training)

    @property
    def input_relative_date_indices(self) -> list[int]:
        """Relative date indices of the model input."""
        return list(self._input)

    @property
    def output_relative_date_indices(self) -> list[int]:
        """Relative date indices of the model output."""
        return list(self._output)

    @property
    def n_step_input(self) -> int | None:
        """Number of input steps."""
        return self._n_input

    @property
    def n_step_output(self) -> int | None:
        """Number of output steps."""
        return self._n_output

    @property
    def lagged(self) -> list[datetime.timedelta] | None:
        """Input times relative to the last input (ascending, not positive)."""
        return None if self._lagged is None else list(self._lagged)

    @property
    def input_offsets(self) -> list[datetime.timedelta] | None:
        """Input times relative to the last input."""
        return None if self._input_offsets is None else list(self._input_offsets)

    @property
    def output_offsets(self) -> list[datetime.timedelta] | None:
        """Output times relative to the last input."""
        return None if self._output_offsets is None else list(self._output_offsets)

    @property
    def rollout_shift(self) -> datetime.timedelta | None:
        """Time by which each rollout step advances."""
        return self._rollout_shift

    @property
    def advance_map(self) -> dict[str, list[tuple[int, int]]] | None:
        """Index mapping that advances input and output into the next input."""
        return None if self._advance_map is None else {k: list(v) for k, v in self._advance_map.items()}

    @property
    def input_explicit_times(self) -> list[int] | None:
        """Explicit times of the input steps, in timesteps of the data (see :meth:`TaskReaderV0_2._explicit_times`)."""
        return None if self._input_explicit_times is None else list(self._input_explicit_times)

    @property
    def target_explicit_times(self) -> list[int] | None:
        """Explicit times of the target steps, in timesteps of the data."""
        return None if self._target_explicit_times is None else list(self._target_explicit_times)


class TaskAdapter(ComponentAdapter):
    """Machinery shared by every task adapter, whatever its version or task name.

    Checkpoint level: the timing of every dataset is read and, when they
    agree, exposed as one value (``n_step_input``, ``timing``, ...).

    A task adapter varies on two axes: the layout of the metadata (the
    component *version*) and the task *name*.  Each axis is a mixin placed in
    front of this class:

    * a *version reader* (:class:`TaskReaderV0_2` and its subclasses) knows
      where and how the metadata is stored: :meth:`_timesteps`,
      :meth:`_config_timestep` and the ``_default_n_step_*`` fallbacks;
    * a *task behaviour* (:class:`ForecasterBehaviour`, ...) adds what the task
      derives or stores, through :meth:`_extend`.

    :func:`register_task_version` combines one version reader with every task
    behaviour, so a change made in a version reader reaches every named task.
    When both axes define the same method, the task behaviour wins.  This class
    is not usable on its own: it needs a version reader, which overrides the
    reading hooks declared here.
    """

    # ------------------------------------------------------------------
    # Name
    # ------------------------------------------------------------------

    @property
    def task_name(self) -> str:
        """The task name: recorded, else the top-level ``task`` (F4), else ``"forecaster"``."""
        return effective_task_name(self._raw)

    # ------------------------------------------------------------------
    # Per dataset and checkpoint-level timing
    # ------------------------------------------------------------------

    def for_dataset(self, dataset_name: str) -> TaskTiming:
        """Return the timing of one dataset.

        Raises
        ------
        KeyError
            If *dataset_name* is not one of the dataset names.
        """
        names = self.dataset_names()
        if dataset_name not in names:
            msg = f"Dataset {dataset_name!r} not found. Available datasets: {names!r}"
            raise KeyError(msg)
        return self._build_timing(self._timesteps(dataset_name), dataset_name)

    def timings(self) -> dict[str, TaskTiming]:
        """Return the timing of every dataset, by dataset name."""
        return {name: self.for_dataset(name) for name in self.dataset_names()}

    @property
    def timing(self) -> TaskTiming:
        """The timing shared by all datasets.

        Raises
        ------
        TaskTimingError
            If the datasets have different timings; use :meth:`for_dataset`.
        """
        timings = self.timings()
        if not timings:
            return self._build_timing({})
        first_name, first = next(iter(timings.items()))
        for name, other in timings.items():
            if other != first:
                msg = (
                    f"The timing differs between datasets ({first_name!r} and {name!r}), so there is no "
                    "checkpoint-level value; use task.for_dataset(name) for the timing of one dataset."
                )
                raise TaskTimingError(msg)
        return first

    def _timing_of(self, dataset_name: str | None) -> TaskTiming:
        """Timing of *dataset_name*, or the shared timing when it is ``None``."""
        return self.timing if dataset_name is None else self.for_dataset(dataset_name)

    def timestep(self, dataset_name: str | None = None) -> str | None:
        """Return the model timestep (e.g. ``"6h"``).

        Parameters
        ----------
        dataset_name : str | None, optional
            Dataset to read; ``None`` for the value shared by all datasets
            (raises :class:`TaskTimingError` when they disagree).
        """
        return self._timing_of(dataset_name).timestep

    def input_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return the relative date indices used as model input (see :meth:`timestep`)."""
        return self._timing_of(dataset_name).input_relative_date_indices

    def output_relative_date_indices(self, dataset_name: str | None = None) -> list[int]:
        """Return the relative date indices produced as model output (see :meth:`timestep`)."""
        return self._timing_of(dataset_name).output_relative_date_indices

    @property
    def n_step_input(self) -> int | None:
        """Number of input steps shared by all datasets."""
        return self.timing.n_step_input

    @property
    def n_step_output(self) -> int | None:
        """Number of output steps shared by all datasets."""
        return self.timing.n_step_output

    @property
    def lagged(self) -> list[datetime.timedelta] | None:
        """Input times relative to the last input, shared by all datasets."""
        return self.timing.lagged

    @property
    def input_offsets(self) -> list[datetime.timedelta] | None:
        """Input offsets shared by all datasets."""
        return self.timing.input_offsets

    @property
    def output_offsets(self) -> list[datetime.timedelta] | None:
        """Output offsets shared by all datasets."""
        return self.timing.output_offsets

    @property
    def rollout_shift(self) -> datetime.timedelta | None:
        """Rollout shift shared by all datasets."""
        return self.timing.rollout_shift

    @property
    def advance_map(self) -> dict[str, list[tuple[int, int]]] | None:
        """Advance map shared by all datasets."""
        return self.timing.advance_map

    @property
    def input_explicit_times(self) -> list[int] | None:
        """Explicit input times shared by all datasets."""
        return self.timing.input_explicit_times

    @property
    def target_explicit_times(self) -> list[int] | None:
        """Explicit target times shared by all datasets."""
        return self.timing.target_explicit_times

    # ------------------------------------------------------------------
    # Reading hooks (implemented by the version reader)
    # ------------------------------------------------------------------

    def _timesteps(self, dataset_name: str) -> dict[str, Any]:
        """The ``timesteps`` block of one dataset (empty when absent)."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def _config_timestep(self) -> Any:
        """The timestep recorded in the config, for files whose ``timesteps`` block has none."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def _explicit_times(self, which: str, indices: list[int]) -> list[int]:
        """Times of the ``"input"`` or ``"target"`` steps in timesteps of the data; *indices* when none are recorded."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def _data_frequency(self, dataset_name: str | None) -> str | None:
        """The frequency of the data of one dataset, as recorded, or ``None``."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def _default_n_step_input(self) -> int | None:
        """Number of input steps when no input indices are recorded."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    def _default_n_step_output(self, n_input: int | None) -> int | None:
        """Number of output steps when no output indices are recorded."""
        msg = f"{type(self).__name__} has no version reader"
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Building the timing
    # ------------------------------------------------------------------

    def _build_timing(self, timesteps: dict[str, Any], dataset_name: str | None = None) -> TaskTiming:
        training = _int_list(timesteps.get("relative_date_indices_training"))
        input_indices = _int_list(timesteps.get("input_relative_date_indices"))
        output_indices = _int_list(timesteps.get("output_relative_date_indices"))

        n_input = len(input_indices) or self._default_n_step_input()
        n_output = len(output_indices) or self._default_n_step_output(n_input)
        if not input_indices and n_input is not None:
            input_indices = list(range(n_input))
        if not output_indices and n_output is not None:
            output_indices = list(range(n_input or 0, (n_input or 0) + n_output))
        if not training:
            training = sorted(input_indices + output_indices)

        timestep = timesteps.get("timestep")
        if not isinstance(timestep, str):
            timestep = self._config_timestep()
            timestep = timestep if isinstance(timestep, str) else None
        delta = _timedelta(timestep)
        lagged = None if delta is None or n_input is None else sorted(-s * delta for s in range(n_input))

        fields: dict[str, Any] = {
            "timestep": timestep,
            "relative_date_indices_training": training,
            "input_relative_date_indices": input_indices,
            "output_relative_date_indices": output_indices,
            "n_step_input": n_input,
            "n_step_output": n_output,
            "lagged": lagged,
            "input_explicit_times": self._explicit_times("input", input_indices),
            "target_explicit_times": self._explicit_times("target", output_indices),
        }
        fields.update(self._extend(fields, timesteps, dataset_name))
        return TaskTiming(**fields)

    def _extend(self, fields: dict[str, Any], timesteps: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
        """Return task specific values that add to or replace the generic *fields*."""
        del fields, timesteps, dataset_name  # Hook: the generic task has no specific values.
        return {}


# --------------------------------------------------------------------------
# Version axis: readers
# --------------------------------------------------------------------------


class TaskReaderV0_2(TaskAdapter):
    """Version reader for the component version ``0.2`` (metadata format ``2.0``).

    Knows where the task values are stored: the per dataset ``timesteps``
    block and the ``config`` fallbacks for the oldest files.  A reader for
    another layout subclasses the nearest one and overrides what differs; it
    is then combined with every task name by :func:`register_task_version`.
    """

    def _timesteps(self, dataset_name: str) -> dict[str, Any]:
        """The ``timesteps`` block of one dataset (empty when absent)."""
        timesteps = self._dataset_block(dataset_name).get("timesteps")
        return timesteps if isinstance(timesteps, dict) else {}

    def _config_value(self, key: str) -> Any:
        """Return ``config.task.<key>``, else ``config.training.<key>``, else ``None``."""
        config = self._section("config")
        for part in ("task", "training"):
            block = config.get(part)
            if isinstance(block, dict) and block.get(key) is not None:
                return block[key]
        return None

    def _config_timestep(self) -> Any:
        """The timestep recorded in the config, for files whose ``timesteps`` block has none."""
        task = self._section("config").get("task")
        if isinstance(task, dict):
            for key in ("timestep", "input_timestep"):
                if task.get(key) is not None:
                    return task[key]
        data = self._section("config").get("data")
        return data.get("timestep") if isinstance(data, dict) else None

    def _explicit_times(self, which: str, indices: list[int]) -> list[int]:
        """Times of the ``"input"`` or ``"target"`` steps, in timesteps of the data.

        With several datasets (``metadata_inference.dataset_names``) they are the
        relative date indices.  Otherwise ``config.training.explicit_times``,
        then ``config.task.explicit_times``, give them; the relative date
        indices when the config has none.
        """
        if "dataset_names" not in self._inference:
            config = self._section("config")
            for part in ("training", "task"):
                block = config.get(part)
                explicit_times = block.get("explicit_times") if isinstance(block, dict) else None
                if isinstance(explicit_times, dict) and explicit_times.get(which) is not None:
                    return _int_list(explicit_times[which])
        return list(indices)

    def _data_frequency(self, dataset_name: str | None) -> str | None:
        """The data frequency: ``config.task.output_timestep``, else ``dataset.frequency``, else ``config.data``."""
        config = self._section("config")
        dataset = self._section("dataset")
        nested = dataset.get(dataset_name) if dataset_name is not None else None
        sources = (config.get("task"), nested if isinstance(nested, dict) else dataset, config.get("data"))
        for source, key in zip(sources, ("output_timestep", "frequency", "frequency"), strict=True):
            value = source.get(key) if isinstance(source, dict) else None
            if value is not None:
                return value if isinstance(value, str) else None
        return None

    def _default_n_step_input(self) -> int | None:
        """Number of input steps when no input indices are recorded: ``config.task`` or ``config.training``."""
        return _as_int(self._config_value("multistep_input"))

    def _default_n_step_output(self, n_input: int | None) -> int | None:
        """Number of output steps when no output indices are recorded.

        ``multistep_output`` from the config; when the config has none the model
        has a single output step (what inference assumes) as soon as the input
        side is known.
        """
        n = _as_int(self._config_value("multistep_output"))
        if n is None and n_input is not None:
            return 1
        return n


class TaskReaderV1_0(TaskReaderV0_2):
    """Version reader for the component version ``1.0`` (metadata format ``3.0``): the same task layout."""


# --------------------------------------------------------------------------
# Name axis: behaviours
# --------------------------------------------------------------------------


class ForecasterBehaviour(TaskAdapter):
    """Forecaster: offsets, rollout shift and advance map follow from the timestep and step counts.

    The values are the ones inference computes: input offsets
    ``-s * timestep``, output offsets ``(s + 1) * timestep``, the rollout
    shift is the last output offset, and ``advance_map`` maps outputs and
    inputs of one step onto the inputs of the next.
    """

    def _extend(self, fields: dict[str, Any], timesteps: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
        extra = super()._extend(fields, timesteps, dataset_name)
        lagged = fields["lagged"]
        delta = _timedelta(fields["timestep"])
        n, m = fields["n_step_input"], fields["n_step_output"]
        if delta is not None and m is not None:
            output_offsets = [(s + 1) * delta for s in range(m)]
            extra.update(output_offsets=output_offsets, rollout_shift=output_offsets[-1] if output_offsets else None)
        if lagged is not None and n is not None and m is not None:
            extra.update(
                input_offsets=list(lagged),
                advance_map={
                    "outin": [(m - i - 1, n - i - 1) for i in range(min(n, m))],
                    "inin": [(i, i - m) for i in range(m, n)],
                },
            )
        return extra


class OffsetForecasterBehaviour(ForecasterBehaviour):
    """Offset forecaster: training stores the offsets, ``rollout_shift`` and ``advance_map``.

    The relative date indices are positions in the input and output offsets
    lists.  Stored values replace the derived ones; keys that are missing keep
    the forecaster derivation.
    """

    def _extend(self, fields: dict[str, Any], timesteps: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
        extra = super()._extend(fields, timesteps, dataset_name)

        input_offsets = _timedelta_list(timesteps.get("input_offsets"))
        output_offsets = _timedelta_list(timesteps.get("output_offsets"))
        if input_offsets:
            extra["input_offsets"] = input_offsets
            extra["lagged"] = list(input_offsets)
        if output_offsets:
            extra["output_offsets"] = output_offsets
            # Without a stored shift the rollout advances by the last stored output offset.
            extra["rollout_shift"] = output_offsets[-1]
        rollout_shift = _timedelta(timesteps.get("rollout_shift"))
        if rollout_shift is not None:
            extra["rollout_shift"] = rollout_shift
        advance_map = timesteps.get("advance_map")
        if isinstance(advance_map, dict):
            extra["advance_map"] = {
                str(key): [(int(pair[0]), int(pair[1])) for pair in pairs]
                for key, pairs in advance_map.items()
                if isinstance(pairs, list)
            }
        return extra


class TemporalDownscalerBehaviour(TaskAdapter):
    """Temporal downscaler: inputs at ``0`` and one (input) timestep, looking ahead.

    ``timestep`` is the timestep between the two inputs.  When
    ``config.task`` has ``input_timestep`` and ``output_timestep`` the offsets
    are the ones training defines: ``input_offsets`` and ``lagged`` are
    ``[0, input_timestep]``, the output offsets are ``output_timestep * (i + 1)``
    for ``i`` below ``input_timestep / output_timestep - 1``, with ``0`` first
    when ``output_left_boundary`` and ``input_timestep`` last when
    ``output_right_boundary``, and ``rollout_shift`` is ``input_timestep``.

    Older files (time interpolators) store only explicit times, counted in
    timesteps of the data (``config.task.output_timestep``, else the data
    frequency).  Then ``lagged`` is the input times in that unit, so it looks
    ahead instead of backwards (``sorted(s * frequency)``), ``input_offsets``
    are the same values, the output offsets are the target explicit times in
    that unit, and ``rollout_shift`` is the last input offset.  Without a data
    frequency the input offsets are ``[0, timestep]`` and the rollout shift is
    the timestep.
    """

    @staticmethod
    def _training_offsets(
        task: dict[str, object],
    ) -> tuple[list[datetime.timedelta], list[datetime.timedelta]] | None:
        """Input and output offsets as training defines them, or ``None`` when ``config.task`` cannot give them."""
        input_delta = _timedelta(task.get("input_timestep"))
        output_delta = _timedelta(task.get("output_timestep"))
        zero = datetime.timedelta(0)
        if input_delta is None or output_delta is None or output_delta <= zero or input_delta % output_delta:
            return None
        outputs = [output_delta * (i + 1) for i in range(input_delta // output_delta - 1)]
        if task.get("output_left_boundary"):
            outputs.insert(0, zero)
        if task.get("output_right_boundary"):
            outputs.append(input_delta)
        return [zero, input_delta], outputs

    def _extend(self, fields: dict[str, Any], timesteps: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
        extra = super()._extend(fields, timesteps, dataset_name)
        task = self._section("config").get("task")
        training = self._training_offsets(cast("dict[str, object]", task) if isinstance(task, dict) else {})
        if training is not None:
            input_offsets, output_offsets = training
            extra.update(
                input_offsets=input_offsets,
                lagged=list(input_offsets),
                output_offsets=output_offsets,
                rollout_shift=input_offsets[-1],
            )
            return extra

        delta = _timedelta(fields["timestep"])
        if delta is not None:
            extra["input_offsets"] = [datetime.timedelta(0), delta]
            extra["rollout_shift"] = delta
        frequency = _timedelta(self._data_frequency(dataset_name))
        if frequency is not None:
            input_times = cast("list[int]", fields["input_explicit_times"])
            target_times = cast("list[int]", fields["target_explicit_times"])
            lagged = sorted(s * frequency for s in input_times)
            extra["lagged"] = lagged
            extra["output_offsets"] = [s * frequency for s in target_times]
            if lagged:
                extra["input_offsets"] = list(lagged)
                extra["rollout_shift"] = lagged[-1]
        return extra


class TimelessBehaviour(TaskAdapter):
    """Timeless task: input and output are one snapshot at ``t=0`` and the timestep is ``0H``."""

    def _default_n_step_output(self, n_input: int | None) -> int | None:
        del n_input  # A timeless task always has one output step.
        return 1

    def _default_n_step_input(self) -> int | None:
        return 1

    def _config_timestep(self) -> Any:
        return "0H"

    def _extend(self, fields: dict[str, Any], timesteps: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
        extra = super()._extend(fields, timesteps, dataset_name)
        zero = [datetime.timedelta(0)]
        extra.update(input_offsets=list(zero), output_offsets=list(zero))
        return extra


class AutoencoderBehaviour(TimelessBehaviour):
    """Autoencoder: the timeless task training writes under this name."""


#: The task behaviours by task name; ``None`` is the generic task (no behaviour of its own).
TASK_BEHAVIOURS: dict[str | None, type[TaskAdapter] | None] = {
    None: None,
    "forecaster": ForecasterBehaviour,
    "offset-forecaster": OffsetForecasterBehaviour,
    "temporal-downscaler": TemporalDownscalerBehaviour,
    "timeless": TimelessBehaviour,
    "autoencoder": AutoencoderBehaviour,
}

#: Task names read as another task name: older checkpoints record temporal downscalers
#: under ``time-interpolator``.  The recorded name is kept by :func:`task_name`.
TASK_NAME_ALIASES: dict[str, str] = {"time-interpolator": "temporal-downscaler"}


# --------------------------------------------------------------------------
# Both axes together
# --------------------------------------------------------------------------


def register_task_version(
    version: str,
    reader: type[TaskAdapter],
    *,
    names: Iterable[str | None] | None = None,
) -> dict[str | None, type[TaskAdapter]]:
    """Register the task adapters of one component version: *reader* combined with each task name.

    For every task name (``None`` is the generic adapter) a class
    ``<Name>V<version>`` is built from the name's behaviour, *reader* and
    :class:`TaskAdapter`, and registered under ``("task", version, name)``.
    A change made in *reader* therefore reaches every named task.

    Parameters
    ----------
    version : str
        Component version the reader reads (e.g. ``"1.1"``).
    reader : type[TaskAdapter]
        Version reader, usually a subclass of the nearest older reader.
    names : Iterable[str | None] | None, optional
        Task names to register; every name of :data:`TASK_BEHAVIOURS` when omitted.

    Returns
    -------
    dict[str | None, type[TaskAdapter]]
        The registered classes by task name.

    Examples
    --------
    >>> class TaskReaderV1_1(TaskReaderV1_0):
    ...     def _timesteps(self, dataset_name): ...
    >>> classes = register_task_version("1.1", TaskReaderV1_1)
    >>> classes["forecaster"].version
    '1.1'
    """
    registered: dict[str | None, type[TaskAdapter]] = {}
    for name in TASK_BEHAVIOURS if names is None else names:
        behaviour = TASK_BEHAVIOURS[name]
        label = "".join(part.title() for part in name.split("-")) if name else "Task"
        bases = (reader, TaskAdapter) if behaviour is None else (behaviour, reader, TaskAdapter)
        adapter_cls = cast(
            "type[TaskAdapter]",
            type(
                f"{label}V{version.replace('.', '_')}",
                bases,
                {
                    "__module__": __name__,
                    "__doc__": f"Task adapter for {name or 'the generic task'}, version {version}.",
                },
            ),
        )
        registered[name] = ComponentRegistry.register("task", version, name=name)(adapter_cls)
    return registered


def build_task_adapter(raw: Mapping[str, object], version: str) -> TaskAdapter:
    """Build the task adapter of a checkpoint: the version reader of *version* and the task's behaviour.

    The task is chosen with :func:`effective_task_name`, read through
    :data:`TASK_NAME_ALIASES`; a name without an adapter of its own uses the
    generic one, with a warning.

    Parameters
    ----------
    raw : Mapping[str, Any]
        The raw metadata dictionary of the checkpoint.
    version : str
        The task component version (stamped or inferred).
    """
    name = effective_task_name(raw)
    name = TASK_NAME_ALIASES.get(name, name)
    adapter_cls = cast("type[TaskAdapter]", ComponentRegistry.resolve("task", version, name=name))
    return adapter_cls(raw)


# Component version 0.2 (format 2.0) and 1.0 (format 3.0).
_V0_2 = register_task_version("0.2", TaskReaderV0_2)
_V1_0 = register_task_version("1.0", TaskReaderV1_0)

TaskV0_2 = _V0_2[None]
ForecasterV0_2 = _V0_2["forecaster"]
OffsetForecasterV0_2 = _V0_2["offset-forecaster"]
TemporalDownscalerV0_2 = _V0_2["temporal-downscaler"]
TimelessV0_2 = _V0_2["timeless"]
AutoencoderV0_2 = _V0_2["autoencoder"]

TaskV1_0 = _V1_0[None]
ForecasterV1_0 = _V1_0["forecaster"]
OffsetForecasterV1_0 = _V1_0["offset-forecaster"]
TemporalDownscalerV1_0 = _V1_0["temporal-downscaler"]
TimelessV1_0 = _V1_0["timeless"]
AutoencoderV1_0 = _V1_0["autoencoder"]
