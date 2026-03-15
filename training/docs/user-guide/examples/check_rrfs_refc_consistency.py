#!/usr/bin/env python3
"""Check that RRFS REFC GRIB key encoding is consistent across files.

This script finds the REFC message index in each GRIB2 file using `wgrib2 -s`,
then inspects the exact ecCodes keys for that message with `grib_ls`.
It verifies the tuple:
  (discipline, parameterCategory, parameterNumber, typeOfLevel, level)
is identical for all files (defaults to RRFS values observed on Ursa).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RefcKeys:
    discipline: str
    parameter_category: str
    parameter_number: str
    type_of_level: str
    level: str


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _collect_files(inp: Path, pattern: str) -> list[Path]:
    if inp.is_file():
        return [inp]
    if not inp.exists():
        raise FileNotFoundError(f"Input path does not exist: {inp}")
    files = sorted(p for p in inp.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files match {pattern} under {inp}")
    return files


def _refc_message_index(path: Path, wgrib2_bin: str) -> int | None:
    cp = _run([wgrib2_bin, str(path), "-s"])
    if cp.returncode != 0:
        raise RuntimeError(f"wgrib2 failed on {path}: {cp.stderr.strip()}")
    for line in cp.stdout.splitlines():
        # Format: idx:offset:d=...:REFC:...
        if ":REFC:" in line:
            m = re.match(r"^(\d+):", line)
            if m:
                return int(m.group(1))
    return None


def _read_refc_keys(path: Path, count_idx: int, grib_ls_bin: str) -> RefcKeys:
    cp = _run(
        [
            grib_ls_bin,
            "-w",
            f"count={count_idx}",
            "-p",
            "discipline,parameterCategory,parameterNumber,typeOfLevel,level",
            str(path),
        ]
    )
    if cp.returncode != 0:
        raise RuntimeError(f"grib_ls failed on {path}: {cp.stderr.strip()}")

    # Parse the row printed under the header.
    values_line = None
    for line in cp.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(str(path)):
            continue
        if line.startswith("discipline"):
            continue
        if "of" in line and "messages in" in line:
            continue
        values_line = line
        break

    if not values_line:
        raise RuntimeError(f"Could not parse grib_ls output for {path}")

    parts = values_line.split()
    if len(parts) < 5:
        raise RuntimeError(f"Unexpected grib_ls row for {path}: {values_line}")

    return RefcKeys(
        discipline=parts[0],
        parameter_category=parts[1],
        parameter_number=parts[2],
        type_of_level=parts[3],
        level=parts[4],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check REFC key consistency across RRFS GRIB files.")
    p.add_argument("input", type=Path, help="GRIB2 file or directory")
    p.add_argument("--pattern", default="rrfs.v*.grib2", help="Glob pattern if input is directory")
    p.add_argument("--wgrib2", default="wgrib2", help="Path to wgrib2")
    p.add_argument("--grib-ls", default="grib_ls", help="Path to grib_ls")
    p.add_argument("--expect-discipline", default="0")
    p.add_argument("--expect-category", default="16")
    p.add_argument("--expect-number", default="5")
    p.add_argument("--expect-typeoflevel", default="atmosphereSingleLayer")
    p.add_argument("--expect-level", default="0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = _collect_files(args.input, args.pattern)

    expected = RefcKeys(
        discipline=args.expect_discipline,
        parameter_category=args.expect_category,
        parameter_number=args.expect_number,
        type_of_level=args.expect_typeoflevel,
        level=args.expect_level,
    )

    missing_refc: list[Path] = []
    mismatches: list[tuple[Path, RefcKeys]] = []

    for f in files:
        idx = _refc_message_index(f, args.wgrib2)
        if idx is None:
            missing_refc.append(f)
            continue
        keys = _read_refc_keys(f, idx, args.grib_ls)
        if keys != expected:
            mismatches.append((f, keys))

    if missing_refc:
        print(f"ERROR: REFC missing in {len(missing_refc)} file(s).", file=sys.stderr)
        for p in missing_refc[:20]:
            print(f"  - {p}", file=sys.stderr)
        return 2

    if mismatches:
        print(f"ERROR: REFC key mismatch in {len(mismatches)} file(s).", file=sys.stderr)
        print(f"Expected: {expected}", file=sys.stderr)
        for p, k in mismatches[:20]:
            print(f"  - {p}: {k}", file=sys.stderr)
        return 3

    print(f"REFC consistency OK across {len(files)} file(s): {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

