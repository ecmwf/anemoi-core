#!/usr/bin/env python3
"""Plot epoch losses from Anemoi training logs.

Works with old and new metric names, e.g.:
- train_mse_loss_epoch
- train_multi_dataset_loss_epoch
- val_multi_dataset_loss_epoch
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+):")
LOSS_RE = re.compile(r"(?P<key>[A-Za-z0-9_]*loss_epoch)=(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_log(path: str) -> dict[str, dict[int, float]]:
    by_metric: dict[str, dict[int, float]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            m_epoch = EPOCH_RE.search(line)
            if not m_epoch:
                continue
            epoch = int(m_epoch.group("epoch"))
            for m_loss in LOSS_RE.finditer(line):
                key = m_loss.group("key")
                val = float(m_loss.group("val"))
                # Keep the last value seen for the epoch.
                by_metric[key][epoch] = val

    return by_metric


def choose_default_keys(all_keys: list[str]) -> list[str]:
    preferred = [
        "train_multi_dataset_loss_epoch",
        "val_multi_dataset_loss_epoch",
        "train_mse_loss_epoch",
        "val_mse_loss_epoch",
    ]
    selected = [k for k in preferred if k in all_keys]
    if selected:
        return selected
    # Fallback: plot any epoch-loss keys.
    return sorted(all_keys)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Anemoi epoch losses from a log file.")
    p.add_argument("log_file", help="Path to log file (e.g. dd.err)")
    p.add_argument("output_png", nargs="?", default="loss_epoch.png", help="Output figure path")
    p.add_argument(
        "--keys",
        default="",
        help="Comma-separated metric keys to plot. Default: auto-detect",
    )
    args = p.parse_args()

    metrics = parse_log(args.log_file)
    if not metrics:
        raise SystemExit("No '*loss_epoch=' metrics found in log.")

    all_keys = sorted(metrics.keys())
    if args.keys.strip():
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        missing = [k for k in keys if k not in metrics]
        if missing:
            raise SystemExit(f"Requested keys not found: {missing}. Available: {all_keys}")
    else:
        keys = choose_default_keys(all_keys)

    plt.figure(figsize=(8, 4.8))
    for key in keys:
        series = metrics[key]
        if not series:
            continue
        epochs = sorted(series.keys())
        values = [series[e] for e in epochs]
        plt.plot(epochs, values, marker="o", linewidth=1.2, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150)

    print("Available keys:", ", ".join(all_keys))
    print("Plotted keys:", ", ".join(keys))
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
