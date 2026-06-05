#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich", "fire"]
# ///

import fire


def main(initial_count: float, epochs: int = 6):
    import rich.table

    table = rich.table.Table("Epoch", "it/s")
    for epoch in range(0, epochs + 1):
        count = initial_count / (epoch + 1)
        table.add_row(str(epoch), f"{count:.2f}")
    rich.print(table)


if __name__ == "__main__":
    fire.Fire(main)
