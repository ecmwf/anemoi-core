#!/usr/bin/env python
"""Plot validation loss summaries exported from Anemoi loss tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary file: {path}")
    return pd.read_csv(path)


def _resolve_paths(input_path: Path, detail_csv: Path | None, output_dir: Path | None) -> tuple[Path, Path | None, Path]:
    if input_path.is_file():
        detail_csv = input_path
        summaries_dir = input_path.parent / "summaries"
    else:
        summaries_dir = input_path
    plots_dir = output_dir or summaries_dir / "plots"
    return summaries_dir, detail_csv, plots_dir


def _total_context(summaries_dir: Path, detail_csv: Path | None) -> str:
    epoch_summary = summaries_dir.parent / "validation_loss_epoch_summary.csv"
    parts: list[str] = []

    if epoch_summary.exists():
        data = pd.read_csv(epoch_summary)
        if not data.empty and "weighted_mean_lightning_val_step_total" in data:
            value = float(data["weighted_mean_lightning_val_step_total"].iloc[-1])
            parts.append(f"epoch val loss={value:.6g}")

    if detail_csv is not None and detail_csv.exists():
        detail = pd.read_csv(detail_csv)
        if "scope" in detail and "loss" in detail:
            batch_total = detail.loc[detail["scope"] == "lightning_val_step_total", "loss"]
            rollout_total = detail.loc[
                detail["scope"] == "configured_loss_rollout_all_samples_all_variables", "loss"
            ]
            if len(batch_total):
                parts.append(f"mean batch loss={batch_total.mean():.6g}")
            if len(rollout_total):
                parts.append(f"mean rollout loss={rollout_total.mean():.6g}")

    return " | ".join(parts) if parts else "total loss unavailable"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def plot_variable_ranking(by_variable: pd.DataFrame, plots_dir: Path, context: str, top_variables: int) -> None:
    data = by_variable.sort_values("mean_loss", ascending=False).head(top_variables).iloc[::-1]
    fig_height = max(5.0, 0.28 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(data["variable"], data["mean_loss"], color="#2f6f9f")
    ax.set_xlabel("Mean configured loss")
    ax.set_ylabel("Variable")
    ax.set_title(f"Top Variable Loss Contributions\n{context}")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, plots_dir / "variable_loss_ranking.png")


def plot_sample_distribution(by_sample: pd.DataFrame, plots_dir: Path, context: str) -> None:
    data = _clean_numeric(by_sample, ["sample_global_index", "mean_loss"]).sort_values("sample_global_index")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(data["sample_global_index"], data["mean_loss"], lw=1.2, color="#2f6f9f")
    axes[0].set_xlabel("Sample global index")
    axes[0].set_ylabel("Mean configured loss")
    axes[0].set_title(f"Sample Loss Sequence\n{context}")
    axes[0].grid(alpha=0.25)

    axes[1].hist(data["mean_loss"], bins=50, color="#7f9c57", edgecolor="white")
    axes[1].set_xlabel("Mean configured loss")
    axes[1].set_ylabel("Number of samples")
    axes[1].set_title("Sample Loss Distribution")
    axes[1].grid(axis="y", alpha=0.25)

    _save(fig, plots_dir / "sample_loss_distribution.png")


def plot_lead_variable_heatmap(
    by_lead_variable: pd.DataFrame,
    by_variable: pd.DataFrame,
    plots_dir: Path,
    context: str,
    top_variables: int,
) -> None:
    data = _clean_numeric(by_lead_variable, ["rollout_step", "lead_index", "mean_loss"])
    top_names = (
        by_variable.sort_values("mean_loss", ascending=False)
        .head(top_variables)["variable"]
        .astype(str)
        .tolist()
    )
    data = data[data["variable"].isin(top_names)].copy()
    data["lead"] = "r" + data["rollout_step"].astype(int).astype(str) + "_out" + data["lead_index"].astype(int).astype(str)
    pivot = data.pivot_table(index="lead", columns="variable", values="mean_loss", aggfunc="mean")
    pivot = pivot.reindex(columns=top_names)

    fig_width = max(10.0, 0.35 * len(pivot.columns) + 3.0)
    fig_height = max(4.5, 0.45 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Rollout/output lead")
    ax.set_title(f"Lead by Variable Loss Heatmap\n{context}")
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label("Mean configured loss")
    _save(fig, plots_dir / "lead_variable_loss_heatmap.png")


def plot_top_sample_variable_heatmap(
    by_sample: pd.DataFrame,
    by_sample_variable: pd.DataFrame,
    by_variable: pd.DataFrame,
    plots_dir: Path,
    context: str,
    top_samples: int,
    top_variables: int,
) -> None:
    sample_data = _clean_numeric(by_sample, ["sample_global_index", "mean_loss"])
    sample_names = (
        sample_data.sort_values("mean_loss", ascending=False)
        .head(top_samples)["sample_global_index"]
        .astype(int)
        .astype(str)
        .tolist()
    )
    variable_names = (
        by_variable.sort_values("mean_loss", ascending=False)
        .head(top_variables)["variable"]
        .astype(str)
        .tolist()
    )

    data = _clean_numeric(by_sample_variable, ["sample_global_index", "mean_loss"])
    data["sample_global_index"] = data["sample_global_index"].astype("Int64").astype(str)
    data = data[data["sample_global_index"].isin(sample_names) & data["variable"].isin(variable_names)]
    pivot = data.pivot_table(index="sample_global_index", columns="variable", values="mean_loss", aggfunc="mean")
    pivot = pivot.reindex(index=sample_names, columns=variable_names)

    fig_width = max(10.0, 0.35 * len(pivot.columns) + 3.0)
    fig_height = max(6.0, 0.22 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Top-loss sample global index")
    ax.set_title(f"Top Sample by Variable Loss Heatmap\n{context}")
    cbar = fig.colorbar(image, ax=ax, shrink=0.9)
    cbar.set_label("Mean configured loss")
    _save(fig, plots_dir / "top_sample_variable_loss_heatmap.png")


def plot_focus_variable(
    focus_variable: str,
    by_sample_variable: pd.DataFrame,
    by_lead_variable: pd.DataFrame,
    plots_dir: Path,
    context: str,
) -> None:
    sample_data = _clean_numeric(by_sample_variable, ["sample_global_index", "mean_loss"])
    sample_data = sample_data[sample_data["variable"] == focus_variable].sort_values("sample_global_index")

    lead_data = _clean_numeric(by_lead_variable, ["rollout_step", "lead_index", "mean_loss"])
    lead_data = lead_data[lead_data["variable"] == focus_variable].copy()
    lead_data["lead"] = "r" + lead_data["rollout_step"].astype(int).astype(str) + "_out" + lead_data["lead_index"].astype(int).astype(str)

    if sample_data.empty and lead_data.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    if not sample_data.empty:
        axes[0].plot(sample_data["sample_global_index"], sample_data["mean_loss"], color="#b24b3a", lw=1.1)
        axes[0].set_xlabel("Sample global index")
        axes[0].set_ylabel(f"{focus_variable} mean loss")
        axes[0].grid(alpha=0.25)
    axes[0].set_title(f"{focus_variable} Loss by Sample\n{context}")

    if not lead_data.empty:
        axes[1].bar(lead_data["lead"], lead_data["mean_loss"], color="#b24b3a")
        axes[1].set_xlabel("Rollout/output lead")
        axes[1].set_ylabel(f"{focus_variable} mean loss")
        axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title(f"{focus_variable} Loss by Lead")

    _save(fig, plots_dir / f"{focus_variable}_loss_focus.png")


def plot_overview(
    by_variable: pd.DataFrame,
    by_sample: pd.DataFrame,
    by_lead_variable: pd.DataFrame,
    plots_dir: Path,
    context: str,
    top_variables: int,
) -> None:
    top_var = by_variable.sort_values("mean_loss", ascending=False).head(top_variables)
    sample_data = _clean_numeric(by_sample, ["mean_loss"])
    lead_data = _clean_numeric(by_lead_variable, ["rollout_step", "lead_index", "mean_loss"]).copy()
    lead_total = lead_data.groupby(["rollout_step", "lead_index"], as_index=False)["mean_loss"].mean()
    lead_total["lead"] = (
        "r" + lead_total["rollout_step"].astype(int).astype(str) + "_out" + lead_total["lead_index"].astype(int).astype(str)
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(top_var["variable"], top_var["mean_loss"], color="#2f6f9f")
    axes[0, 0].set_title("Top Variable Mean Loss")
    axes[0, 0].set_ylabel("Mean configured loss")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].hist(sample_data["mean_loss"], bins=50, color="#7f9c57", edgecolor="white")
    axes[0, 1].set_title("Sample Loss Distribution")
    axes[0, 1].set_xlabel("Mean configured loss")
    axes[0, 1].set_ylabel("Number of samples")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].plot(lead_total["lead"], lead_total["mean_loss"], marker="o", color="#b24b3a")
    axes[1, 0].set_title("Mean Loss by Lead")
    axes[1, 0].set_xlabel("Rollout/output lead")
    axes[1, 0].set_ylabel("Mean configured loss")
    axes[1, 0].grid(alpha=0.25)

    text = (
        f"{context}\n\n"
        f"variables: {len(by_variable)}\n"
        f"samples: {len(by_sample)}\n"
        f"lead-variable rows: {len(by_lead_variable)}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.95, text, va="top", ha="left", fontsize=12)
    fig.suptitle("Validation Loss Review", fontsize=16, y=1.02)
    _save(fig, plots_dir / "validation_loss_overview.png")


def plot_summaries(
    summaries_dir: Path,
    detail_csv: Path | None,
    plots_dir: Path,
    top_variables: int,
    top_samples: int,
    focus_variable: str,
) -> None:
    by_variable = _read_csv(summaries_dir / "summary_by_variable.csv")
    by_sample = _read_csv(summaries_dir / "summary_by_sample.csv")
    by_sample_variable = _read_csv(summaries_dir / "summary_by_sample_variable.csv")
    by_lead_variable = _read_csv(summaries_dir / "summary_by_lead_variable.csv")
    context = _total_context(summaries_dir, detail_csv)

    plot_overview(by_variable, by_sample, by_lead_variable, plots_dir, context, min(top_variables, 16))
    plot_variable_ranking(by_variable, plots_dir, context, top_variables)
    plot_sample_distribution(by_sample, plots_dir, context)
    plot_lead_variable_heatmap(by_lead_variable, by_variable, plots_dir, context, top_variables)
    plot_top_sample_variable_heatmap(
        by_sample,
        by_sample_variable,
        by_variable,
        plots_dir,
        context,
        top_samples,
        min(top_variables, 24),
    )
    plot_focus_variable(focus_variable, by_sample_variable, by_lead_variable, plots_dir, context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Either validation_loss_detail_epochNNN.csv or the summaries directory.",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Optional detail CSV, used to add total-loss context when input is a summaries directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output plot directory. Defaults to summaries/plots.")
    parser.add_argument("--top-variables", type=int, default=30, help="Number of highest-loss variables to show.")
    parser.add_argument("--top-samples", type=int, default=50, help="Number of highest-loss samples for sample-variable heatmap.")
    parser.add_argument("--focus-variable", default="refc", help="Variable for the focused diagnostic plot.")
    args = parser.parse_args()

    summaries_dir, detail_csv, plots_dir = _resolve_paths(args.input, args.detail_csv, args.output_dir)
    plot_summaries(
        summaries_dir=summaries_dir,
        detail_csv=detail_csv,
        plots_dir=plots_dir,
        top_variables=args.top_variables,
        top_samples=args.top_samples,
        focus_variable=args.focus_variable,
    )
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
