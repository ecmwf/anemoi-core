#!/usr/bin/env python
"""Summarize validation loss tables exported by ExportValidationLossTable."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _mean(rows: list[float]) -> float:
    return sum(rows) / len(rows) if rows else float("nan")


def _float_or_nan(row: dict[str, str], key: str, fallback: str | None = None) -> float:
    value = row.get(key, "")
    if value == "" and fallback is not None:
        value = row.get(fallback, "")
    return float(value) if value != "" else float("nan")


def _metrics_from_row(row: dict[str, str]) -> dict[str, float]:
    return {
        "scaled_loss": _float_or_nan(row, "scaled_loss", fallback="loss"),
        "raw_unscaled_loss": _float_or_nan(row, "raw_unscaled_loss"),
        "loss_contribution_to_total": _float_or_nan(row, "loss_contribution_to_total", fallback="loss"),
    }


def _write_summary(path: Path, key_names: list[str], values: dict[tuple[str, ...], list[dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value_names = ["mean_scaled_loss", "mean_raw_unscaled_loss", "mean_loss_contribution_to_total"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*key_names, "mean_loss", *value_names, "num_rows"])
        for key, metrics in sorted(values.items()):
            mean_scaled = _mean([item["scaled_loss"] for item in metrics])
            mean_raw = _mean([item["raw_unscaled_loss"] for item in metrics])
            mean_contribution = _mean([item["loss_contribution_to_total"] for item in metrics])
            writer.writerow([*key, mean_scaled, mean_scaled, mean_raw, mean_contribution, len(metrics)])


def _metadata_from_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "sample_dataset_index": row.get("sample_dataset_index", ""),
        "sample_worker_id": row.get("sample_worker_id", ""),
        "sample_worker_position": row.get("sample_worker_position", ""),
        "input_times": row.get("input_times", ""),
        "target_times": row.get("target_times", ""),
    }


def _write_sample_variable_totals(
    path: Path,
    sample_variable_rows: dict[tuple[str, str], list[dict[str, float]]],
    sample_metadata: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_global_index",
                "variable",
                "sample_dataset_index",
                "sample_worker_id",
                "sample_worker_position",
                "input_times",
                "target_times",
                "mean_scaled_loss",
                "mean_raw_unscaled_loss",
                "sum_loss_contribution_to_total",
                "num_rows",
            ],
        )
        for key, metrics in sorted(sample_variable_rows.items()):
            sample_global_index, variable = key
            metadata = sample_metadata.get(sample_global_index, {})
            writer.writerow(
                [
                    sample_global_index,
                    variable,
                    metadata.get("sample_dataset_index", ""),
                    metadata.get("sample_worker_id", ""),
                    metadata.get("sample_worker_position", ""),
                    metadata.get("input_times", ""),
                    metadata.get("target_times", ""),
                    _mean([item["scaled_loss"] for item in metrics]),
                    _mean([item["raw_unscaled_loss"] for item in metrics]),
                    sum(item["loss_contribution_to_total"] for item in metrics),
                    len(metrics),
                ],
            )


def _write_variable_totals(
    path: Path,
    sample_variable_rows: dict[tuple[str, str], list[dict[str, float]]],
) -> None:
    variable_totals: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    for (_, variable), metrics in sample_variable_rows.items():
        variable_totals[(variable,)].append(
            {
                "scaled_loss": _mean([item["scaled_loss"] for item in metrics]),
                "raw_unscaled_loss": _mean([item["raw_unscaled_loss"] for item in metrics]),
                "loss_contribution_to_total": sum(item["loss_contribution_to_total"] for item in metrics),
            },
        )
    _write_summary(path, ["variable"], variable_totals)


def summarize(detail_csv: Path, output_dir: Path) -> None:
    by_sample: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    by_sample_variable: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    by_lead_variable: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    sample_metadata: dict[str, dict[str, str]] = {}

    with detail_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scope = row["scope"]
            if scope == "configured_loss_sample_lead_variable":
                metrics = _metrics_from_row(row)
                by_sample_variable[(row["sample_global_index"], row["variable"])].append(metrics)
                by_lead_variable[(row["rollout_step"], row["lead_index"], row["variable"])].append(metrics)
            elif scope == "configured_loss_sample_all_variables":
                by_sample[(row["sample_global_index"],)].append(_metrics_from_row(row))
                sample_metadata[row["sample_global_index"]] = _metadata_from_row(row)

    _write_variable_totals(output_dir / "summary_by_variable.csv", by_sample_variable)
    _write_summary(output_dir / "summary_by_sample.csv", ["sample_global_index"], by_sample)
    _write_sample_variable_totals(output_dir / "summary_by_sample_variable.csv", by_sample_variable, sample_metadata)
    _write_summary(output_dir / "summary_by_lead_variable.csv", ["rollout_step", "lead_index", "variable"], by_lead_variable)

    sample_summary = output_dir / "summary_by_sample.csv"
    rows = []
    with sample_summary.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata = sample_metadata.get(row["sample_global_index"], {})
            rows.append(row | metadata)
    if rows:
        with sample_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


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
