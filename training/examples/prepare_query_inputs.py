# (C) Copyright 2026 Anemoi contributors.

"""Create the physical-valued input mapping consumed by QueryForecaster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "VALUES_NPY", "FIELDS_JSON"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = {}
    for name, values_path, fields_path in args.source:
        values = torch.from_numpy(np.load(values_path)).float()
        with Path(fields_path).open(encoding="utf-8") as handle:
            fields = json.load(handle)
        if values.ndim != 2 or values.shape[1] != len(fields):
            msg = f"{name}: values must have shape (native_nodes, len(fields))."
            raise ValueError(msg)
        inputs[name] = {"values": values, "fields": fields}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, args.output)


if __name__ == "__main__":
    main()
