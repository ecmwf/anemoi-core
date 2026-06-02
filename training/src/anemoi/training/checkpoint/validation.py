# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pipeline-level validation for checkpoint operations.

Two consumers shape the public surface of this module:

1. :class:`CheckpointPipelineValidator` — called by
   :meth:`anemoi.training.checkpoint.pipeline.CheckpointPipeline._perform_pre_execution_validation`
   to perform lightweight environment and configuration checks before the
   pipeline runs. Each method returns a dict with the keys
   ``status`` (``"ok" | "warning" | "error"``), ``issues``, ``warnings`` and
   optionally ``info``. The pipeline logs these and records the status in
   ``context.metadata`` under ``validation_environment_status`` and
   ``validation_config_status``.

2. :func:`validate_pipeline_health` — post-execution sanity check that
   inspects a finished :class:`CheckpointContext` and either returns
   ``True`` or raises :class:`CheckpointValidationError` with the issues
   collected in ``validation_errors``. The API mirrors
   :func:`anemoi.training.checkpoint.utils.validate_checkpoint`.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import TYPE_CHECKING
from typing import Any

from .exceptions import CheckpointValidationError

if TYPE_CHECKING:
    from .base import CheckpointContext

LOGGER = logging.getLogger(__name__)

_MIN_PYTHON = (3, 11)
_MIN_TORCH = (2, 2)


def _status_from_buckets(issues: list[str], warnings: list[str]) -> str:
    """Reduce ``(issues, warnings)`` to a single status label."""
    if issues:
        return "error"
    if warnings:
        return "warning"
    return "ok"


def _parse_version(version: str) -> tuple[int, ...]:
    """Best-effort parse of a dotted version string into an int tuple.

    Stops at the first non-numeric component to tolerate suffixes like
    ``"2.2.0+cu121"`` or ``"3.11.10rc1"``.
    """
    parts: list[int] = []
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


class CheckpointPipelineValidator:
    """Pre-execution health checks for the checkpoint pipeline.

    Methods are static and side-effect-free: they collect findings into a
    dict and return it. The caller (the pipeline) decides whether a
    ``"warning"`` or ``"error"`` status should be acted on.
    """

    @staticmethod
    def validate_environment_setup() -> dict[str, Any]:
        """Check the runtime environment is sane for checkpoint operations.

        Returns
        -------
        dict
            Mapping with keys ``status``, ``issues``, ``warnings``, ``info``.
        """
        issues: list[str] = []
        warnings: list[str] = []
        info: list[str] = []

        py = sys.version_info
        if (py.major, py.minor) < _MIN_PYTHON:
            issues.append(
                f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ required, found {py.major}.{py.minor}.{py.micro}",
            )
        else:
            info.append(f"Python {py.major}.{py.minor}.{py.micro}")

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            issues.append("PyTorch is not installed; checkpoint loading requires torch")
        else:
            try:
                import torch
            except ImportError as e:
                issues.append(f"PyTorch import failed: {e}")
            else:
                torch_version = _parse_version(torch.__version__)
                if torch_version and torch_version < _MIN_TORCH:
                    issues.append(
                        f"torch>={_MIN_TORCH[0]}.{_MIN_TORCH[1]} required, found {torch.__version__}",
                    )
                else:
                    info.append(f"torch {torch.__version__}")

                if torch.cuda.is_available():
                    info.append(f"CUDA available ({torch.cuda.device_count()} device(s))")
                else:
                    info.append("CUDA not available (CPU only)")

        # Optional dependencies — absence is informational, never fatal.
        if importlib.util.find_spec("pytorch_lightning") is None:
            info.append("PyTorch Lightning not installed (optional)")
        else:
            info.append("PyTorch Lightning available")

        return {
            "status": _status_from_buckets(issues, warnings),
            "issues": issues,
            "warnings": warnings,
            "info": info,
        }

    @staticmethod
    def validate_configuration(config: Any) -> dict[str, Any]:
        """Shape-check the Hydra config passed to the pipeline.

        Parameters
        ----------
        config : DictConfig or mapping-like
            The user-supplied configuration. May be ``None``.

        Returns
        -------
        dict
            Mapping with keys ``status``, ``issues``, ``warnings``.
        """
        issues: list[str] = []
        warnings: list[str] = []

        if config is None:
            issues.append("Configuration is None; expected a DictConfig")
            return {"status": "error", "issues": issues, "warnings": warnings}

        training = _safe_attr(config, "training")
        if training is None:
            warnings.append("Configuration has no 'training' section")
        else:
            checkpoint_cfg = _safe_attr(training, "checkpoint")
            if checkpoint_cfg is not None:
                _check_checkpoint_subblocks(checkpoint_cfg, warnings)

        return {
            "status": _status_from_buckets(issues, warnings),
            "issues": issues,
            "warnings": warnings,
        }


def _safe_attr(obj: Any, name: str) -> Any:
    """Read ``obj.name`` or ``obj[name]`` without raising on absence."""
    if obj is None:
        return None
    if hasattr(obj, name):
        try:
            return getattr(obj, name)
        except AttributeError:
            return None
    try:
        return obj[name]  # type: ignore[index]
    except (KeyError, TypeError):
        return None


def _check_checkpoint_subblocks(
    checkpoint_cfg: Any,
    warnings: list[str],
) -> None:
    """Verify ``training.checkpoint`` has at least one of source/loading."""
    source = _safe_attr(checkpoint_cfg, "source")
    loading = _safe_attr(checkpoint_cfg, "loading")

    if source is None and loading is None:
        warnings.append(
            "training.checkpoint has neither 'source' nor 'loading' block; the pipeline will be a no-op",
        )
        return

    if source is not None and _safe_attr(source, "_target_") is None:
        warnings.append("training.checkpoint.source has no '_target_' field")

    if loading is not None and _safe_attr(loading, "_target_") is None:
        warnings.append("training.checkpoint.loading has no '_target_' field")


def validate_pipeline_health(
    context: CheckpointContext,
    *,
    raise_on_error: bool = True,
) -> bool:
    """Check that a finished checkpoint pipeline left the context in a sane state.

    The function inspects ``context.metadata`` (for stage execution markers
    and the pre-execution validation status) and the context's structural
    fields (model/optimizer/scheduler/format coherence).

    Parameters
    ----------
    context : CheckpointContext
        The context returned by ``CheckpointPipeline.execute()``.
    raise_on_error : bool, optional
        When ``True`` (the default), any collected issue raises
        :class:`CheckpointValidationError`. When ``False``, the function
        returns ``False`` instead.

    Returns
    -------
    bool
        ``True`` if the context looks healthy. ``False`` only when
        ``raise_on_error=False`` and at least one issue was found.

    Raises
    ------
    CheckpointValidationError
        If ``raise_on_error=True`` and any issue was found. The exception's
        ``.validation_errors`` attribute lists each problem as a string.
    """
    issues: list[str] = []

    _check_stage_completion(context, issues)
    _check_source_loaded_weights(context, issues)
    _check_structural_invariants(context, issues)
    _check_pre_execution_status(context, issues)

    if not issues:
        LOGGER.debug("Pipeline health validation passed")
        return True

    if raise_on_error:
        msg = "Pipeline health check failed"
        raise CheckpointValidationError(msg, issues, {"num_issues": len(issues)})
    LOGGER.warning("Pipeline health validation found %d issue(s)", len(issues))
    return False


def _check_stage_completion(context: CheckpointContext, issues: list[str]) -> None:
    """Every ``stage_{i}_*`` entry must be a completion marker."""
    stage_entries = [(k, v) for k, v in context.metadata.items() if k.startswith("stage_")]

    if not stage_entries:
        # A pipeline with zero stages is allowed; the pre-execution
        # validation still runs and records validation_performed=True.
        # Flag only when the metadata is completely empty, which would
        # indicate the pipeline never executed.
        if not context.metadata:
            issues.append("Context metadata is empty; pipeline did not execute")
        return

    for key, value in stage_entries:
        if not isinstance(value, str):
            issues.append(f"Stage entry '{key}' has non-string value: {value!r}")
            continue
        if "failed" in value.lower():
            issues.append(f"Stage '{key}' did not complete: {value}")


def _check_source_loaded_weights(context: CheckpointContext, issues: list[str]) -> None:
    """If a Source stage ran, the model must report loaded weights."""
    source_ran = any("source" in key.lower() for key in context.metadata if key.startswith("stage_"))
    if not source_ran:
        return

    model = context.model
    if model is None:
        issues.append("A Source stage executed but context.model is None")
        return

    if not getattr(model, "weights_initialized", False):
        issues.append(
            "A Source stage executed but model.weights_initialized is False; loading strategy may have been skipped",
        )


def _check_structural_invariants(context: CheckpointContext, issues: list[str]) -> None:
    """Catch mutually-incoherent combinations of context fields."""
    if context.optimizer is not None and context.model is None:
        issues.append("Optimizer present but model is None")

    if context.scheduler is not None and context.optimizer is None:
        issues.append("Scheduler present but optimizer is None")

    if context.pl_module is not None and context.checkpoint_format not in (None, "lightning"):
        issues.append(
            f"pl_module set but checkpoint_format is {context.checkpoint_format!r}, expected 'lightning'",
        )


def _check_pre_execution_status(context: CheckpointContext, issues: list[str]) -> None:
    """Flag any error reported by the pre-execution validation hook."""
    env_status = context.metadata.get("validation_environment_status")
    if env_status == "error":
        issues.append("Pre-execution environment validation reported errors")

    cfg_status = context.metadata.get("validation_config_status")
    if cfg_status == "error":
        issues.append("Pre-execution configuration validation reported errors")
