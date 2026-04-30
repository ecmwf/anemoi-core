#!/usr/bin/env python
"""Summarize validation loss tables exported by ExportValidationLossTable."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _mean(rows: list[float]) -> float:
    return sum(rows) / len(rows) if rows else float("nan")


def _write_summary(path: Path, key_names: list[str], values: dict[tuple[str, ...], list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*key_names, "mean_loss", "num_rows"])
        for key, losses in sorted(values.items()):
            writer.writerow([*key, _mean(losses), len(losses)])


def summarize(detail_csv: Path, output_dir: Path) -> None:
    by_variable: dict[tuple[str, ...], list[float]] = defaultdict(list)
    by_sample: dict[tuple[str, ...], list[float]] = defaultdict(list)
    by_sample_variable: dict[tuple[str, ...], list[float]] = defaultdict(list)
    by_lead_variable: dict[tuple[str, ...], list[float]] = defaultdict(list)

    with detail_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss = float(row["loss"])
            scope = row["scope"]
            if scope == "configured_loss_sample_lead_variable":
                by_variable[(row["variable"],)].append(loss)
                by_sample_variable[(row["sample_global_index"], row["variable"])].append(loss)
                by_lead_variable[(row["rollout_step"], row["lead_index"], row["variable"])].append(loss)
            elif scope == "configured_loss_sample_all_variables":
                by_sample[(row["sample_global_index"],)].append(loss)

    _write_summary(output_dir / "summary_by_variable.csv", ["variable"], by_variable)
    _write_summary(output_dir / "summary_by_sample.csv", ["sample_global_index"], by_sample)
    _write_summary(output_dir / "summary_by_sample_variable.csv", ["sample_global_index", "variable"], by_sample_variable)
    _write_summary(output_dir / "summary_by_lead_variable.csv", ["rollout_step", "lead_index", "variable"], by_lead_variable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path, help="validation_loss_detail_epochNNN.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary CSVs. Defaults to a summaries directory beside the detail CSV.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.detail_csv.parent / "summaries"
    summarize(args.detail_csv, output_dir)
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
