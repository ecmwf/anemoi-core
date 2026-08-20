# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Migration from V0 (legacy, no ``metadata_inference``) to V1.

This migration synthesises the ``metadata_inference`` block that V1 requires
from the raw ``data_indices``, ``config``, and ``dataset`` fields stored in a
V0 checkpoint.
"""

from typing import Any

from ..migration import MetadataMigrator
from ..versions.v0 import MetadataV0
from ..versions.v1 import MetadataV1


@MetadataMigrator.register_migration("0.0", "1.0")
def migrate_v0_to_v1(old: MetadataV0) -> MetadataV1:
    """Migrate a V0 (legacy) metadata instance to V1.

    Synthesises the ``metadata_inference`` block required by V1 from the
    raw ``data_indices``, ``config``, and ``dataset`` fields of the V0
    instance.  All other top-level fields are carried forward verbatim.

    Parameters
    ----------
    old : MetadataV0
        Validated V0 instance (legacy checkpoint without ``metadata_inference``).

    Returns
    -------
    MetadataV1
        Validated V1 instance with a synthesised ``metadata_inference`` block.
    """
    # ------------------------------------------------------------------
    # Extract raw fields from the V0 instance
    # ------------------------------------------------------------------
    data_indices: dict[str, Any] = old.data_indices  # type: ignore[reportAttributeAccessIssue]
    config: dict[str, Any] = old.config  # type: ignore[reportAttributeAccessIssue]

    # Resolve the effective dataset section (handles nested "data" sub-key).
    dataset_section: dict[str, Any] = old._ds  # type: ignore[reportAttributeAccessIssue]

    # Variable list.
    variables: list[str] = dataset_section.get("variables", [])

    # data_input_full[i] = dataset index of the i-th input tensor slot.
    data_input_full: list[int] = data_indices.get("data", {}).get("input", {}).get("full", [])
    data_output_full: list[int] = data_indices.get("data", {}).get("output", {}).get("full", [])

    # ------------------------------------------------------------------
    # Build input variable index mapping: variable_name -> tensor position
    # ------------------------------------------------------------------
    input_var_indices: dict[str, int] = {}
    for pos, dataset_idx in enumerate(data_input_full):
        if dataset_idx < len(variables):
            input_var_indices[variables[dataset_idx]] = pos

    # ------------------------------------------------------------------
    # Build output variable index mapping: variable_name -> tensor position
    # ------------------------------------------------------------------
    output_var_indices: dict[str, int] = {}
    output_pos_to_var: dict[int, str] = {}
    for pos, dataset_idx in enumerate(data_output_full):
        if dataset_idx < len(variables):
            name = variables[dataset_idx]
            output_var_indices[name] = pos
            output_pos_to_var[pos] = name

    # ------------------------------------------------------------------
    # Variable types
    # ------------------------------------------------------------------
    config_data: dict[str, Any] = config.get("data", {})
    forcing_names: list[str] = config_data.get("forcing") or []
    diagnostic_names: list[str] = config_data.get("diagnostic") or []

    # Prognostic: model output prognostic indices → variable names.
    model_output: dict[str, Any] = data_indices.get("model", {}).get("output", {})
    model_output_full: list[int] = model_output.get("full", [])
    model_output_prognostic: list[int] = model_output.get("prognostic", [])

    prognostic_names: list[str] = []
    for model_pos in model_output_prognostic:
        if model_pos < len(model_output_full):
            data_out_pos = model_output_full[model_pos]
            if data_out_pos in output_pos_to_var:
                prognostic_names.append(output_pos_to_var[data_out_pos])

    # Target = all output variables.
    target_names: list[str] = [variables[idx] for idx in data_output_full if idx < len(variables)]

    # ------------------------------------------------------------------
    # Timestep and multi-step info
    # ------------------------------------------------------------------
    timestep: str = config_data.get("timestep", "6h")
    multistep_input: int = config.get("training", {}).get("multistep_input", 1)

    input_indices: list[int] = list(range(-(multistep_input - 1), 1))
    output_indices: list[int] = [1]
    training_indices: list[int] = list(range(-(multistep_input - 1), 2))

    # ------------------------------------------------------------------
    # Grid size from dataset.shape
    # ------------------------------------------------------------------
    shape: list[int] = dataset_section.get("shape", [])
    grid: int | None = shape[-1] if len(shape) >= 4 else (shape[1] if len(shape) == 3 else None)

    # ------------------------------------------------------------------
    # Assemble the metadata_inference block
    # ------------------------------------------------------------------
    dataset_config: dict[str, Any] = {
        "data_indices": {
            "input": input_var_indices,
            "output": output_var_indices,
        },
        "variable_types": {
            "forcing": forcing_names,
            "diagnostic": diagnostic_names,
            "prognostic": prognostic_names,
            "target": target_names,
        },
        "timesteps": {
            "timestep": timestep,
            "input_relative_date_indices": input_indices,
            "output_relative_date_indices": output_indices,
            "relative_date_indices_training": training_indices,
        },
        "shapes": {
            "variables": len(data_input_full),
            "input_timesteps": multistep_input,
            "ensemble": 1,
            "grid": grid,
        },
    }

    # Carry forward run_id from the V0 extra fields (or uuid as fallback).
    extras: dict[str, Any] = dict(old.model_extra or {})
    run_id: str = str(extras.get("run_id", extras.get("uuid", "legacy")))

    metadata_inference: dict[str, Any] = {
        "seed": 0,
        "run_id": run_id,
        "task": None,
        "dataset_names": ["data"],
        "datasets": {"data": dataset_config},
    }

    # ------------------------------------------------------------------
    # Build the V1 dict, carrying forward all permissive sections
    # ------------------------------------------------------------------
    # Use the original (non-nested) dataset dict for V1's permissive section.
    # V0's `provenance_training` maps to V1's `provenance` field.
    # V0's `data_indices` is intentionally NOT carried forward (superseded by
    # the synthesised `metadata_inference` block above).
    v1_dict: dict[str, Any] = {
        "schema_version": "1.0",
        "original_schema_version": "0.0",
        "metadata_inference": metadata_inference,
        "config": dict(old.config),  # type: ignore[reportAttributeAccessIssue]
        "dataset": dict(old.dataset),  # type: ignore[reportAttributeAccessIssue]
        "provenance": dict(old.provenance_training),  # type: ignore[reportAttributeAccessIssue]
    }

    # Carry forward any extra top-level keys from V0 (e.g. run_id, uuid,
    # timestamp, version, seed, supporting_arrays_paths, …).
    for key, value in extras.items():
        v1_dict.setdefault(key, value)

    return MetadataV1.model_validate(v1_dict)
