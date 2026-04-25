#!/usr/bin/env python
"""Inspect Anemoi checkpoint metadata for variable role splits.

Usage:
  python check_checkpoint_roles.py /path/to/last.ckpt
  python check_checkpoint_roles.py /path/to/inference-last.ckpt --dataset data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat

import torch
from anemoi.utils.checkpoints import load_metadata


def _load_metadata_from_checkpoint(path: Path) -> dict:
    """Load metadata from either a Lightning or inference checkpoint."""
    metadata = {}

    try:
        metadata = load_metadata(path)
    except Exception as exc:
        print(f"WARNING: load_metadata failed for {path}: {exc}")

    if metadata:
        return metadata

    raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        hyper_parameters = raw.get("hyper_parameters", {})
        if isinstance(hyper_parameters, dict):
            metadata = hyper_parameters.get("metadata", {}) or {}
    else:
        metadata = getattr(raw, "metadata", {}) or {}

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to last.ckpt or inference-last.ckpt")
    parser.add_argument("--dataset", default="data", help="Dataset name to inspect (default: data)")
    args = parser.parse_args()

    metadata = _load_metadata_from_checkpoint(args.checkpoint)
    if not metadata:
        raise SystemExit(f"No metadata found in {args.checkpoint}")

    metadata_inference = metadata.get("metadata_inference", {})
    dataset_meta = metadata_inference.get(args.dataset, {}) if isinstance(metadata_inference, dict) else {}
    variable_types = dataset_meta.get("variable_types", {}) if isinstance(dataset_meta, dict) else {}
    data_indices = dataset_meta.get("data_indices", {}) if isinstance(dataset_meta, dict) else {}

    print(f"checkpoint: {args.checkpoint}")
    print(f"dataset: {args.dataset}")
    print("run_id:", metadata.get("run_id"))
    print("metadata_inference.run_id:", metadata_inference.get("run_id") if isinstance(metadata_inference, dict) else None)
    print("task:", metadata_inference.get("task") if isinstance(metadata_inference, dict) else None)
    print()
    print("variable_types:")
    print(json.dumps(variable_types, indent=2, sort_keys=True))
    print()
    print("model input variables:")
    print(pformat(sorted((data_indices.get("input") or {}).keys())))
    print()
    print("model output variables:")
    print(pformat(sorted((data_indices.get("output") or {}).keys())))
    print()

    if "refc" in variable_types.get("diagnostic", []):
        print("RESULT: refc is diagnostic/output-only in this checkpoint metadata.")
    elif "refc" in variable_types.get("prognostic", []):
        print("RESULT: refc is prognostic/input-state in this checkpoint metadata.")
    else:
        print("RESULT: refc was not found in diagnostic or prognostic lists.")


if __name__ == "__main__":
    main()
