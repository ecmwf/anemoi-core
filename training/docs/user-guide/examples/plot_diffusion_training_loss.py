#!/usr/bin/env python3
"""Plot diffusion training loss from Anemoi logs.

This is intended for the current diffusion training logs, which commonly expose
`train_multi_dataset_loss_step=...` in the progress-bar output rather than only
epoch-level metrics.

The script can also plot any matching validation/train metrics found in the
same log.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
STEP_LOSS_RE = re.compile(
    r"(?P<key>(?:train|val)_[A-Za-z0-9_]*(?:loss|mse)_(?:step|epoch))="
    r"(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+):")


def parse_log(path: str) -> dict[str, list[tuple[int, float]]]:
    """Parse diffusion-relevant loss metrics from a log file.

    Returns a dict keyed by metric name, where each value is a list of
    `(sample_index, value)` pairs in log order.
    """
    by_metric: dict[str, list[tuple[int, float]]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)

            epoch = None
            m_epoch = EPOCH_RE.search(line)
            if m_epoch:
                epoch = int(m_epoch.group("epoch"))

            for m_loss in STEP_LOSS_RE.finditer(line):
                key = m_loss.group("key")
                val = float(m_loss.group("val"))
                x = epoch if key.endswith("_epoch") and epoch is not None else counters[key]
                by_metric[key].append((x, val))
                counters[key] += 1

    return by_metric


def choose_default_keys(all_keys: list[str]) -> list[str]:
    preferred = [
        "train_multi_dataset_loss_step",
        "train_multi_dataset_loss_epoch",
        "val_multi_dataset_loss_epoch",
        "val_mse_epoch",
        "train_mse_loss_epoch",
        "val_mse_loss_epoch",
    ]
    selected = [k for k in preferred if k in all_keys]
    if selected:
        return selected
    return sorted(all_keys)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot diffusion training loss from an Anemoi log.")
    p.add_argument("log_file", help="Path to the log file (for example dd.output)")
    p.add_argument("output_png", nargs="?", default="diffusion_training_loss.png", help="Output figure path")
    p.add_argument("--keys", default="", help="Comma-separated metric keys to plot. Default: auto-detect.")
    p.add_argument("--title", default="Diffusion Training Loss", help="Plot title")
    args = p.parse_args()

    metrics = parse_log(args.log_file)
    if not metrics:
        raise SystemExit("No diffusion-style '*loss_step', '*loss_epoch', or '*mse_epoch' metrics found in log.")

    all_keys = sorted(metrics.keys())
    if args.keys.strip():
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        missing = [k for k in keys if k not in metrics]
        if missing:
            raise SystemExit(f"Requested keys not found: {missing}. Available: {all_keys}")
    else:
        keys = choose_default_keys(all_keys)

    plt.figure(figsize=(8.5, 5.0))
    for key in keys:
        series = metrics[key]
        if not series:
            continue
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        marker = "o" if key.endswith("_epoch") else None
        plt.plot(xs, ys, linewidth=1.2, marker=marker, label=key)

    plt.xlabel("Step index" if any(k.endswith("_step") for k in keys) else "Epoch")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150)

    print("Available keys:", ", ".join(all_keys))
    print("Plotted keys:", ", ".join(keys))
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
