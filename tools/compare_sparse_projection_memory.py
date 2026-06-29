#!/usr/bin/env python3
# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Compare memory use of old COO and new CSR sparse projection construction.

This script is intentionally standalone: it does not import Anemoi modules.
Each variant runs in a fresh subprocess so peak RSS measurements are isolated.

Examples
--------
    python tools/compare_sparse_projection_memory.py --rows 200000 --cols 200000 --edges-per-row 16
    python tools/compare_sparse_projection_memory.py --rows 200000 --edges-per-row 16 --include-file
    python tools/compare_sparse_projection_memory.py --device cuda --move-to-device
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz

GRAPH_VARIANTS = ("old_graph_coo", "old_graph_coo_then_csr", "new_graph_csr")
FILE_VARIANTS = ("old_file_coo", "old_file_coo_then_csr", "new_file_csr")


def _max_rss_mb() -> float:
    """Return process peak RSS in MiB."""
    # Linux reports KiB; macOS reports bytes. This environment is Linux, but
    # keeping the conversion portable makes the script less surprising.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024**2
    return rss / 1024


def _current_rss_mb() -> float | None:
    """Return current RSS in MiB on Linux, or None when /proc is unavailable."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def _cuda_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"cuda_allocated_mb": None, "cuda_peak_allocated_mb": None}
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def _tensor_storage_mb(matrix: torch.Tensor) -> float:
    """Return sparse tensor index/value storage size in MiB."""
    if matrix.layout == torch.sparse_coo:
        matrix = matrix.coalesce()
        bytes_ = matrix.indices().numel() * matrix.indices().element_size()
        bytes_ += matrix.values().numel() * matrix.values().element_size()
        return bytes_ / 1024**2

    if matrix.layout == torch.sparse_csr:
        bytes_ = matrix.crow_indices().numel() * matrix.crow_indices().element_size()
        bytes_ += matrix.col_indices().numel() * matrix.col_indices().element_size()
        bytes_ += matrix.values().numel() * matrix.values().element_size()
        return bytes_ / 1024**2

    msg = f"Unsupported sparse layout: {matrix.layout}"
    raise TypeError(msg)


def _make_graph_inputs(
    rows: int,
    cols: int,
    edges_per_row: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create PyG-like edge_index and edge weights."""
    rng = np.random.default_rng(seed)
    row_index = np.repeat(np.arange(rows, dtype=np.int64), edges_per_row)
    col_index = rng.integers(0, cols, size=rows * edges_per_row, dtype=np.int64)
    weights = np.full(rows * edges_per_row, 1.0 / edges_per_row, dtype=np.float32)

    # PyG convention: edge_index[0] = source/column, edge_index[1] = target/row.
    edge_index = torch.from_numpy(np.stack([col_index, row_index]))
    edge_weight = torch.from_numpy(weights)
    return edge_index.to(device), edge_weight.to(device)


def _make_scipy_matrix(rows: int, cols: int, edges_per_row: int, seed: int):
    rng = np.random.default_rng(seed)
    row_index = np.repeat(np.arange(rows, dtype=np.int64), edges_per_row)
    col_index = rng.integers(0, cols, size=rows * edges_per_row, dtype=np.int64)
    weights = np.full(rows * edges_per_row, 1.0 / edges_per_row, dtype=np.float32)
    matrix = coo_matrix((weights, (row_index, col_index)), shape=(rows, cols), dtype=np.float32).tocsr()
    matrix.sum_duplicates()
    return matrix


def _old_from_graph(edge_index: torch.Tensor, weights: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    row_index = edge_index[1].long()
    col_index = edge_index[0].long()
    edge_index_for_coo = torch.stack([row_index, col_index])
    return torch.sparse_coo_tensor(edge_index_for_coo, weights, (rows, cols), device=edge_index.device).coalesce()


def _new_from_graph(edge_index: torch.Tensor, weights: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    row_index = edge_index[1]
    col_index = edge_index[0]
    matrix = coo_matrix(
        (
            weights.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy(),
            (
                row_index.detach().cpu().contiguous().numpy(),
                col_index.detach().cpu().contiguous().numpy(),
            ),
        ),
        shape=(rows, cols),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr),
        torch.from_numpy(matrix.indices),
        torch.from_numpy(matrix.data),
        size=matrix.shape,
    )


def _old_from_file(path: Path) -> torch.Tensor:
    matrix = load_npz(path)
    edge_index = torch.tensor(np.vstack(matrix.nonzero()), dtype=torch.long)
    weights = torch.tensor(matrix.data, dtype=torch.float32)
    rows, cols = matrix.shape
    return torch.sparse_coo_tensor(edge_index, weights, (rows, cols)).coalesce()


def _new_from_file(path: Path) -> torch.Tensor:
    matrix = load_npz(path).astype(np.float32, copy=False).tocsr()
    matrix.sum_duplicates()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr),
        torch.from_numpy(matrix.indices),
        torch.from_numpy(matrix.data),
        size=matrix.shape,
    )


def _worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA requested but torch.cuda.is_available() is False."
        raise RuntimeError(msg)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    input_max_rss_mb: float
    if args.variant in GRAPH_VARIANTS:
        edge_index, weights = _make_graph_inputs(args.rows, args.cols, args.edges_per_row, device, args.seed)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        input_max_rss_mb = _max_rss_mb()
        input_current_rss_mb = _current_rss_mb()

        if args.variant == "old_graph_coo":
            projection = _old_from_graph(edge_index, weights, args.rows, args.cols)
        elif args.variant == "old_graph_coo_then_csr":
            projection = _old_from_graph(edge_index, weights, args.rows, args.cols).to_sparse_csr()
        elif args.variant == "new_graph_csr":
            projection = _new_from_graph(edge_index, weights, args.rows, args.cols)
        else:
            raise AssertionError(args.variant)

    elif args.variant in FILE_VARIANTS:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "projection.npz"
            save_npz(path, _make_scipy_matrix(args.rows, args.cols, args.edges_per_row, args.seed))
            gc.collect()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            input_max_rss_mb = _max_rss_mb()
            input_current_rss_mb = _current_rss_mb()

            if args.variant == "old_file_coo":
                projection = _old_from_file(path)
            elif args.variant == "old_file_coo_then_csr":
                projection = _old_from_file(path).to_sparse_csr()
            elif args.variant == "new_file_csr":
                projection = _new_from_file(path)
            else:
                raise AssertionError(args.variant)
    else:
        raise AssertionError(args.variant)

    if args.move_to_device:
        projection = projection.to(device)

    # Touch metadata so the result cannot be optimized away in unusual runtimes.
    nnz = projection._nnz()
    final_max_rss_mb = _max_rss_mb()
    result: dict[str, Any] = {
        "variant": args.variant,
        "rows": args.rows,
        "cols": args.cols,
        "edges_per_row": args.edges_per_row,
        "nnz": nnz,
        "layout": str(projection.layout).replace("torch.", ""),
        "tensor_device": str(projection.device),
        "input_current_rss_mb": input_current_rss_mb,
        "input_peak_rss_mb": input_max_rss_mb,
        "final_current_rss_mb": _current_rss_mb(),
        "final_peak_rss_mb": final_max_rss_mb,
        "construction_peak_delta_mb": max(0.0, final_max_rss_mb - input_max_rss_mb),
        "sparse_tensor_storage_mb": _tensor_storage_mb(projection),
    }
    result.update(_cuda_stats(device))
    print(json.dumps(result, sort_keys=True))


def _run_worker(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    command = [
        sys.executable,
        __file__,
        "--worker",
        "--variant",
        variant,
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--edges-per-row",
        str(args.edges_per_row),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.move_to_device:
        command.append("--move-to-device")

    env = {**os.environ, "PYTHONWARNINGS": "ignore:Sparse CSR tensor support is in beta state"}
    completed = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
    return json.loads(completed.stdout)


def _format_mb(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.1f}"


def _print_table(results: list[dict[str, Any]]) -> None:
    columns = [
        ("variant", "variant"),
        ("layout", "layout"),
        ("tensor_device", "device"),
        ("nnz", "nnz"),
        ("sparse_tensor_storage_mb", "tensor MB"),
        ("construction_peak_delta_mb", "peak delta MB"),
        ("final_current_rss_mb", "final RSS MB"),
        ("cuda_peak_allocated_mb", "CUDA peak MB"),
    ]
    widths = {header: len(header) for _, header in columns}
    rows = []
    for result in results:
        row = {}
        for key, header in columns:
            value = result[key]
            if key == "nnz":
                text = f"{value:,}"
            elif key.endswith("_mb"):
                text = _format_mb(value)
            else:
                text = str(value)
            row[header] = text
            widths[header] = max(widths[header], len(text))
        rows.append(row)

    print(" ".join(header.ljust(widths[header]) for _, header in columns))
    print(" ".join("-" * widths[header] for _, header in columns))
    for row in rows:
        print(" ".join(row[header].ljust(widths[header]) for _, header in columns))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="Projection matrix rows.")
    parser.add_argument("--cols", type=int, default=None, help="Projection matrix columns. Defaults to --rows.")
    parser.add_argument("--edges-per-row", type=int, default=16, help="Average non-zero entries per row.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Input graph tensor device.")
    parser.add_argument(
        "--move-to-device",
        action="store_true",
        help="Move the final projection matrix to --device, simulating get_edges(device=...).",
    )
    parser.add_argument("--include-file", action="store_true", help="Also compare .npz file-backed construction.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=GRAPH_VARIANTS + FILE_VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.cols = args.rows if args.cols is None else args.cols
    return args


def main() -> None:
    args = _parse_args()
    if args.worker:
        _worker(args)
        return

    variants = list(GRAPH_VARIANTS)
    if args.include_file:
        variants.extend(FILE_VARIANTS)

    print(
        f"rows={args.rows:,} cols={args.cols:,} edges_per_row={args.edges_per_row} "
        f"device={args.device} move_to_device={args.move_to_device}"
    )
    results = [_run_worker(args, variant) for variant in variants]
    _print_table(results)
    print(
        "\npeak delta MB is process peak RSS after inputs/files are prepared minus peak RSS at that baseline. "
        "It is useful for relative comparisons, not an exact allocator-level profile."
    )


if __name__ == "__main__":
    main()
