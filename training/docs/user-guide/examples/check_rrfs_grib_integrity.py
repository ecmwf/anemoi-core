#!/usr/bin/env python3
"""Check RRFS valid-time GRIB files for required variable/level completeness.

This script scans each GRIB2 file with `wgrib2 -s`, verifies required keys,
and reports:
  - missing fields
  - duplicated fields
  - which valid times are incomplete

Default required keys are aligned with the RRFS recipe used in this repo.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Key = Tuple[str, str, str]  # (var, level, fcst)


@dataclass
class FileReport:
    file: Path
    valid_time: str
    missing: List[Key]
    duplicated: List[Tuple[Key, int]]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.duplicated


def run_cmd(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def default_expected_keys() -> List[Key]:
    pl_vars = ["UGRD", "VGRD", "TMP", "SPFH", "HGT", "CLMR", "ICMR", "RWMR", "SNMR", "GRLE"]
    pl_levels = ["925 mb", "850 mb", "500 mb", "200 mb"]
    out: List[Key] = []
    for v in pl_vars:
        for lev in pl_levels:
            out.append((v, lev, "anl"))
    out.extend(
        [
            ("PRES", "surface", "anl"),
            ("TMP", "surface", "anl"),
            ("SMREF", "surface", "anl"),
            ("SDSWRF", "surface", "anl"),
            ("LSM", "surface", "anl"),
            ("OROG", "surface", "anl"),
        ]
    )
    return out


def parse_inventory(stdout: str) -> Dict[Key, int]:
    counts: Dict[Key, int] = defaultdict(int)
    for line in stdout.splitlines():
        parts = line.strip().split(":")
        # Typical format:
        # idx:offset:d=YYYYMMDDHH:VAR:LEVEL:anl:
        if len(parts) < 6:
            continue
        var = parts[3].strip().upper()
        level = parts[4].strip().lower()
        fcst = parts[5].strip().lower()
        counts[(var, level, fcst)] += 1
    return counts


def parse_valid_time_from_name(path: Path) -> str:
    m = re.search(r"rrfs\.v(\d{10})\.grib2$", path.name)
    if m:
        return m.group(1)
    return path.name


def collect_files(path: Path, pattern: str) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    files = sorted(p for p in path.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files match '{pattern}' under {path}")
    return files


def check_file(file: Path, expected: List[Key], wgrib2_bin: str) -> FileReport:
    cp = run_cmd([wgrib2_bin, str(file), "-s"])
    if cp.returncode != 0:
        raise RuntimeError(f"wgrib2 failed for {file}:\n{cp.stderr}")
    counts = parse_inventory(cp.stdout)

    missing: List[Key] = []
    duplicated: List[Tuple[Key, int]] = []
    for k in expected:
        # normalize expected to inventory normalization
        ek = (k[0].upper(), k[1].lower(), k[2].lower())
        n = counts.get(ek, 0)
        if n == 0:
            missing.append(k)
        elif n > 1:
            duplicated.append((k, n))

    return FileReport(
        file=file,
        valid_time=parse_valid_time_from_name(file),
        missing=missing,
        duplicated=duplicated,
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check RRFS GRIB2 files for missing/duplicate required fields."
    )
    p.add_argument("input", type=Path, help="Input GRIB2 file or directory")
    p.add_argument(
        "--pattern",
        default="rrfs.v*.grib2",
        help="Glob pattern if input is a directory (default: rrfs.v*.grib2)",
    )
    p.add_argument(
        "--wgrib2",
        default="wgrib2",
        help="Path to wgrib2 executable (default: wgrib2 from PATH)",
    )
    p.add_argument(
        "--show-ok",
        action="store_true",
        help="Print OK files too (default only prints problematic files)",
    )
    return p.parse_args(list(argv))


def fmt_key(k: Key) -> str:
    return f"{k[0]} | {k[1]} | {k[2]}"


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    expected = default_expected_keys()

    try:
        files = collect_files(args.input, args.pattern)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    bad: List[FileReport] = []
    for f in files:
        try:
            rep = check_file(f, expected, args.wgrib2)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        if args.show_ok or not rep.ok:
            print(f"\n[{rep.valid_time}] {rep.file}")
            if rep.ok:
                print("  OK")
            else:
                if rep.missing:
                    print(f"  missing: {len(rep.missing)}")
                    for k in rep.missing:
                        print(f"    - {fmt_key(k)}")
                if rep.duplicated:
                    print(f"  duplicated: {len(rep.duplicated)}")
                    for k, n in rep.duplicated:
                        print(f"    - {fmt_key(k)} (count={n})")

        if not rep.ok:
            bad.append(rep)

    print("\n=== Summary ===")
    print(f"files_checked: {len(files)}")
    print(f"files_incomplete: {len(bad)}")
    if bad:
        times = ", ".join(r.valid_time for r in bad)
        print(f"incomplete_times: {times}")
        return 3
    print("All files complete and non-duplicated for required keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

