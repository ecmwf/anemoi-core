# (C) Copyright 2026 Anemoi contributors.

"""Run a query-aware inference checkpoint on physical-valued input fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anemoi.utils.checkpoints import load_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output-coordinates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata, supporting_arrays = load_metadata(
        str(args.checkpoint),
        supporting_arrays=True,
    )
    model.metadata = metadata
    model.supporting_arrays = supporting_arrays
    model.eval()

    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
    with args.query.open(encoding="utf-8") as handle:
        query = json.load(handle)
    if args.output_coordinates is not None:
        query["output_coordinates"] = np.load(args.output_coordinates)

    with torch.inference_mode():
        prediction = model(inputs, query)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prediction.detach().cpu().numpy())


if __name__ == "__main__":
    main()
