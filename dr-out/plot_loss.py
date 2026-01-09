#!/usr/bin/env python3
import re
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise SystemExit("Usage: plot_loss.py <log_file> [output_png]")

    log_path = sys.argv[1]
    output_png = sys.argv[2] if len(sys.argv) == 3 else "loss.png"

    pattern = re.compile(r"Epoch\\s+(?P<epoch>\\d+):.*?train_mse_loss_epoch=(?P<loss>[0-9.]+)")

    # Keep the last loss per epoch (dedupe repeated ranks)
    epoch_to_loss = OrderedDict()

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            loss = float(m.group("loss"))
            epoch_to_loss[epoch] = loss

    if not epoch_to_loss:
        raise SystemExit("No train_mse_loss_epoch entries found.")

    epochs = list(epoch_to_loss.keys())
    losses = list(epoch_to_loss.values())

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, losses, marker="o", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("train_mse_loss_epoch")
    plt.title("Training Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
