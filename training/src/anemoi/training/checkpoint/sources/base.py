# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Abstract base class for checkpoint sources.

Checkpoint sources are responsible for acquiring checkpoint data from
various locations (local filesystem, S3, HTTP, etc.) and populating
the pipeline context with the loaded data and detected format.

Source selection is **explicit**: the user picks which source class
runs by listing its fully-qualified name under ``_target_`` in the
Hydra checkpoint config. There is no scheme-based dispatcher that
maps ``s3://`` URLs to ``S3Source`` automatically — that choice is
left to whoever writes the config, so a future caller could (for
instance) use ``LocalSource`` against a FUSE-mounted S3 path without
the pipeline trying to interpret the scheme. Each source's
``supports(...)`` classmethod exists for opt-in validation, not for
automatic dispatch.

Concretely, if a user configures ``LocalSource`` with an ``s3://``
URI (without a FUSE mount), the URI is interpreted as a literal
local path: ``Path("s3://bucket/key").resolve()`` does not exist on
disk, so ``LocalSource.process(...)`` raises
:class:`~anemoi.training.checkpoint.exceptions.CheckpointNotFoundError`.
Callers that want scheme-based routing should perform their own
dispatch (e.g., by inspecting ``urlparse(uri).scheme``) before
selecting a ``_target_``.

:meth:`CheckpointSource.process` acquires the checkpoint and loads it. A source
that reads a local file also implements :meth:`CheckpointSource.resolve`, which
makes the checkpoint reachable as a local file and publishes its path on the
context; its ``process`` is resolve followed by the ``torch.load``. A resume runs
resolve only, because ``Trainer.fit(ckpt_path=)`` performs the load itself, so
only a source that implements ``resolve`` can resume a run (see
:meth:`CheckpointSource.supports_resume`). A remote source downloads to a
temporary file, loads it and deletes it straight away.

Example
-------
>>> class LocalSource(CheckpointSource):
...     def resolve(self, context: CheckpointContext) -> Path:
...         context.checkpoint_path = Path(context.checkpoint_path).expanduser().resolve()
...         return context.checkpoint_path
...
...     def process(self, context: CheckpointContext) -> CheckpointContext:
...         path = self.resolve(context)
...         self._load_from_path(context, path)
...         return context
"""

from __future__ import annotations

import logging
import pickle
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

import torch

from anemoi.training.checkpoint.base import PipelineStage

if TYPE_CHECKING:
    from pathlib import Path

    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


def source_class_from_config(source_cfg: Any) -> type[CheckpointSource] | None:
    """Resolve a ``training.checkpoint.source`` block to its source class.

    Reads ``_target_`` and imports it with :func:`hydra.utils.get_class`, so a
    subclass (``class EcmwfRunSource(RunIdSource)``) is recognised through
    inheritance rather than by what its name ends in. Returns ``None`` for a
    missing, empty or unimportable target, or one that is not a
    :class:`CheckpointSource`; the pipeline build (or, on a resume,
    ``build_resume_source``) reports those with a ``CheckpointConfigError``
    naming the target, so callers only need a neutral answer here.

    Parameters
    ----------
    source_cfg : Any
        The source configuration (a mapping with ``_target_``), or ``None``.

    Returns
    -------
    type[CheckpointSource] or None
        The configured source class, when it resolves.
    """
    if source_cfg is None:
        return None
    from omegaconf import OmegaConf

    target = (
        OmegaConf.select(source_cfg, "_target_", default="")
        if OmegaConf.is_config(source_cfg)
        else source_cfg.get("_target_", "")
    ) or ""
    if not target:
        return None

    from hydra.utils import get_class

    try:
        cls = get_class(target)
    except (ImportError, ValueError):
        return None
    return cls if isinstance(cls, type) and issubclass(cls, CheckpointSource) else None


class CheckpointSource(PipelineStage):
    """Abstract base class for all checkpoint sources.

    Checkpoint sources form the acquisition layer of the checkpoint
    pipeline. They are responsible for obtaining checkpoint data from
    a specific source type (local file, cloud storage, HTTP endpoint,
    etc.) and populating the context for downstream stages.

    Subclasses implement ``process`` (acquire, then load). A source that
    can make the checkpoint reachable as a local file for a resume also
    implements ``resolve`` (publish the local path, do not load).
    The ``_load_from_path`` and ``_load_and_populate`` convenience methods
    standardise the load and how raw checkpoint data is attached to the
    context with format detection.

    Parameters
    ----------
    None

    Examples
    --------
    >>> class S3Source(CheckpointSource):
    ...     def __init__(self, url: str):
    ...         self.url = url
    ...
    ...     def process(self, context: CheckpointContext) -> CheckpointContext:
    ...         path = self._download_to_temp(self.url)
    ...         try:
    ...             self._load_from_path(context, path)
    ...         finally:
    ...             path.unlink(missing_ok=True)
    ...         return context
    """

    @classmethod
    def run_identity(cls, source_cfg: Any) -> tuple[str | None, str | None]:
        """The ``(run_id, fork_run_id)`` this source configuration expresses.

        Most sources carry no run identity: an explicit file or a remote object
        says nothing about which experiment-tracker run the job belongs to.
        :class:`~anemoi.training.checkpoint.sources.run.RunIdSource` overrides this.

        Parameters
        ----------
        source_cfg : Any
            The source configuration block.

        Returns
        -------
        tuple[str | None, str | None]
            ``(run_id, fork_run_id)``; at most one is set. ``(None, None)`` here.
        """
        del source_cfg
        return None, None

    @classmethod
    def with_run_lineage(
        cls,
        source_cfg: Any,
        parent_run_server2server: str | None,
        fork_run_server2server: str | None,
    ) -> Any:
        """Merge runtime server-to-server lineage into the source configuration.

        The lineage ids are logger-derived at run time and cannot be written in
        the static config. Only a source that resolves a run accepts them;
        everything else returns its configuration unchanged, so instantiation
        never fails on an unknown keyword.

        Parameters
        ----------
        source_cfg : Any
            The source configuration block.
        parent_run_server2server : str or None
            Runtime server-to-server resume lineage id.
        fork_run_server2server : str or None
            Runtime server-to-server fork lineage id.

        Returns
        -------
        Any
            The configuration to instantiate; unchanged here.
        """
        del parent_run_server2server, fork_run_server2server
        return source_cfg

    def resolve(self, context: CheckpointContext) -> Path | None:
        """Make the checkpoint reachable as a local file and publish its path, without loading it.

        Sets ``context.checkpoint_path`` to the resolved file and returns it. A
        source that downloads does not implement this: its download is deleted
        straight after loading, so there is no file to hand Lightning. Returns
        ``None`` when there is nothing to resolve on this rank (an error deferred
        to rank 0), in which case ``process`` leaves the context untouched.

        This is the step a resume runs on its own: ``Trainer.fit(ckpt_path=)`` loads
        the file, so the pipeline only has to produce it.

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context. May carry ``checkpoint_path`` or source
            configuration in ``config``.

        Returns
        -------
        Path or None
            The local checkpoint file, or ``None`` when deferred.

        Raises
        ------
        NotImplementedError
            If the source only implements ``process``; such a source cannot be
            resumed from, because it has no local file to hand Lightning.
        """
        msg = (
            f"{type(self).__name__} does not implement resolve(); a resume needs a source that can "
            "hand Trainer.fit(ckpt_path=) a local checkpoint file"
        )
        raise NotImplementedError(msg)

    @classmethod
    def supports_resume(cls) -> bool:
        """Whether a resume can use this source.

        A resume hands ``Trainer.fit(ckpt_path=)`` a local file, which only a
        source that overrides :meth:`resolve` provides (``LocalSource``,
        ``RunIdSource``). The builder checks this at build time.

        Returns
        -------
        bool
            ``True`` if the class overrides :meth:`resolve`.
        """
        return cls.resolve is not CheckpointSource.resolve

    @abstractmethod
    def process(self, context: CheckpointContext) -> CheckpointContext:
        """Acquire checkpoint data from source and populate context.

        Implementations should:
        1. Call ``resolve`` to obtain the local checkpoint file
        2. Call ``_load_from_path`` to load it onto the context
        3. Add source-specific metadata for tracking

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context. May contain ``checkpoint_path``
            or source configuration in ``config``.

        Returns
        -------
        CheckpointContext
            Context with ``checkpoint_data`` and ``checkpoint_format``
            populated.

        Raises
        ------
        CheckpointSourceError
            If the source cannot be reached or data cannot be fetched
        CheckpointNotFoundError
            If the checkpoint does not exist at the source location
        CheckpointLoadError
            If the fetched data cannot be parsed as a checkpoint
        """

    def _load_from_path(self, context: CheckpointContext, path: Path) -> None:
        """Load the checkpoint file at ``path`` onto the context.

        Loads with ``weights_only=False`` (Anemoi checkpoints carry non-tensor
        metadata such as ``hyper_parameters``) and ``map_location="cpu"``, in a
        worker thread, then delegates to :meth:`_load_and_populate`.

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context (mutated in place)
        path : Path
            The local checkpoint file

        Raises
        ------
        CheckpointLoadError
            If the file cannot be loaded by PyTorch
        """
        from anemoi.training.checkpoint.exceptions import CheckpointLoadError

        try:
            raw_data = torch.load(path, weights_only=False, map_location="cpu")
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as e:
            raise CheckpointLoadError(path, e) from e

        self._load_and_populate(context, raw_data)

    def _load_and_populate(
        self,
        context: CheckpointContext,
        raw_data: dict[str, Any],
    ) -> None:
        """Populate context **in-place** with loaded checkpoint data and detected format.

        This convenience method standardises how checkpoint sources
        attach raw data to the context. It detects the checkpoint format
        from the loaded data and sets both ``checkpoint_data`` and
        ``checkpoint_format`` on the context. The context is mutated in
        place; nothing is returned.

        Parameters
        ----------
        context : CheckpointContext
            Current pipeline context (mutated in place)
        raw_data : dict
            Raw loaded checkpoint data dictionary
        """
        context.checkpoint_data = raw_data

        from anemoi.training.checkpoint.formats import detect_format_from_data

        context.checkpoint_format = detect_format_from_data(raw_data)
        LOGGER.debug(
            "Detected checkpoint format '%s' from data keys",
            context.checkpoint_format,
        )
