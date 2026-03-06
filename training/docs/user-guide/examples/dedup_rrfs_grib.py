#!/usr/bin/env python3
"""Deduplicate RRFS GRIB2 files by GRIB inventory key.

Keeps the first record for each key in a file, where key defaults to:
  (date_token, variable, level, forecast_type)
which matches the useful parts of:
  wgrib2 <file> -s
lines like:
  590:458746568:d=2024050209:VGRD:850 mb:anl:

This script requires `wgrib2` in PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run_cmd(args: list[str], stdin_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_inventory_lines(stdout: str) -> list[tuple[int, tuple[str, str, str, str]]]:
    """Return list of (record_index, dedup_key)."""
    result: list[tuple[int, tuple[str, str, str, str]]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(":")
        # Expect at least:
        # idx, offset, date_token, var, level, fcst_type, ...
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        key = (parts[2], parts[3], parts[4], parts[5])
        result.append((idx, key))
    return result


def dedup_indices(inv: list[tuple[int, tuple[str, str, str, str]]]) -> tuple[list[int], int]:
    seen: set[tuple[str, str, str, str]] = set()
    keep: list[int] = []
    dup_count = 0
    for idx, key in inv:
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        keep.append(idx)
    return keep, dup_count


def make_output_path(input_path: Path, output_dir: Path, suffix: str) -> Path:
    stem = input_path.name
    if stem.endswith(".grib2"):
        out_name = stem[:-6] + suffix + ".grib2"
    else:
        out_name = stem + suffix + ".grib2"
    return output_dir / out_name


def collect_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = sorted(p for p in input_path.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files match pattern '{pattern}' in {input_path}")
    return files


def process_file(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> tuple[int, int]:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {dst}")

    inv = run_cmd(["wgrib2", str(src), "-s"])
    if inv.returncode != 0:
        raise RuntimeError(f"wgrib2 -s failed for {src}:\n{inv.stderr}")

    entries = parse_inventory_lines(inv.stdout)
    if not entries:
        raise RuntimeError(f"No inventory records parsed from {src}")

    keep, dup_count = dedup_indices(entries)
    if dry_run:
        return len(entries), dup_count

    if dst.exists() and overwrite:
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Feed selected record numbers to wgrib2 stdin.
    idx_text = "\n".join(str(i) for i in keep) + "\n"
    out = run_cmd(["wgrib2", str(src), "-i", "-grib", str(dst)], stdin_text=idx_text)
    if out.returncode != 0:
        raise RuntimeError(f"wgrib2 -i -grib failed for {src} -> {dst}:\n{out.stderr}")

    return len(entries), dup_count


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate RRFS GRIB2 records by key (date,var,level,forecast-type) "
            "and write cleaned GRIB2 files."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input GRIB2 file or directory.",
    )
    parser.add_argument(
        "--pattern",
        default="rrfs.v*.grib2",
        help="Glob pattern when input is a directory (default: rrfs.v*.grib2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to '<input_dir>/dedup' for directory input, "
        "or '<input_parent>/dedup' for single file.",
    )
    parser.add_argument(
        "--suffix",
        default=".dedup",
        help="Suffix added before .grib2 in output filenames (default: .dedup).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report duplicate counts only; do not write output files.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    try:
        files = collect_files(args.input, args.pattern)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        out_dir = args.output_dir
    elif args.input.is_dir():
        out_dir = args.input / "dedup"
    else:
        out_dir = args.input.parent / "dedup"

    total_records = 0
    total_dups = 0
    processed = 0

    for src in files:
        dst = make_output_path(src, out_dir, args.suffix)
        try:
            n_records, n_dups = process_file(
                src=src,
                dst=dst,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"ERROR processing {src}: {e}", file=sys.stderr)
            return 1

        processed += 1
        total_records += n_records
        total_dups += n_dups
        action = "would write" if args.dry_run else "wrote"
        print(
            f"{src.name}: records={n_records}, duplicates_removed={n_dups}, "
            f"{action}={dst}"
        )

    print(
        f"Done. files={processed}, records={total_records}, "
        f"duplicates_removed={total_dups}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

