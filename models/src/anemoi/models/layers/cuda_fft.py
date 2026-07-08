# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""CUDA FFT extension bindings."""

import os
import platform
import site
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

LonsPerLat = tuple[int, ...]


def _device_index(x: torch.Tensor) -> int:
    device_index = x.device.index
    if device_index is None:
        return torch.cuda.current_device()
    return device_index


def _env_paths(name: str) -> list[Path]:
    return [Path(path) for path in os.environ.get(name, "").split(os.pathsep) if path]


def _nvidia_module_arch() -> str:
    machine = platform.machine()
    return "Linux_aarch64" if machine in ("aarch64", "arm64") else "Linux_x86_64"


def _nvhpc_version_root() -> Path | None:
    if nvhpc_root := os.environ.get("NVHPC"):
        nvhpc_path = Path(nvhpc_root)
        return nvhpc_path / _nvidia_module_arch() / nvhpc_path.name
    return None


def _usable_cuda_home(path: Path) -> bool:
    return (path / "bin" / "nvcc").is_file() and (path / "include" / "cuda.h").is_file()


def _candidate_cuda_homes() -> list[Path]:
    candidates: list[Path] = []
    if cuda_home := os.environ.get("CUDA_HOME"):
        candidates.append(Path(cuda_home))
    if nvhpc_version_root := _nvhpc_version_root():
        candidates.append(nvhpc_version_root / "cuda" / "13.1")
    return candidates


def _configure_cuda_home() -> None:
    for cuda_home in _candidate_cuda_homes():
        if not _usable_cuda_home(cuda_home):
            continue

        os.environ["CUDA_HOME"] = str(cuda_home)
        import torch.utils.cpp_extension

        # torch caches CUDA_HOME when cpp_extension is imported.
        torch.utils.cpp_extension.CUDA_HOME = str(cuda_home)
        return


def _candidate_cufft_roots() -> list[Path]:
    roots: list[Path] = []
    # Prefer the module path, then fall back to NVIDIA wheels in the venv.
    if cufft_root := os.environ.get("ANEMOI_CUFFT_ROOT"):
        roots.append(Path(cufft_root))

    if nvhpc_version_root := _nvhpc_version_root():
        roots.append(nvhpc_version_root / "math_libs")

    for site_packages in site.getsitepackages():
        roots.append(Path(site_packages) / "nvidia" / "cufft")

    return roots


def _find_library(lib_dir: Path) -> Path | None:
    for name in ("libcufft.so", "libcufft.so.12", "libcufft.so.11"):
        candidate = lib_dir / name
        if candidate.is_file():
            return candidate
    return None


def _cufft_paths_from_root(root: Path) -> tuple[Path, Path] | None:
    include_dirs = [
        root / "include",
        root / "targets" / "x86_64-linux" / "include",
        root / "13.1" / "targets" / "x86_64-linux" / "include",
    ]
    lib_dirs = [
        root / "lib64",
        root / "lib",
        root / "targets" / "x86_64-linux" / "lib",
        root / "13.1" / "targets" / "x86_64-linux" / "lib",
    ]

    for include_dir in include_dirs:
        if not (include_dir / "cufft.h").is_file():
            continue
        for lib_dir in lib_dirs:
            if lib_file := _find_library(lib_dir):
                return include_dir, lib_file
    return None


@lru_cache(maxsize=1)
def _find_cufft_paths() -> tuple[Path, Path] | None:
    include_dir = os.environ.get("ANEMOI_CUFFT_INCLUDE_DIR")
    library_dir = os.environ.get("ANEMOI_CUFFT_LIBRARY_DIR")
    if include_dir is not None and library_dir is not None:
        include_path = Path(include_dir)
        if (include_path / "cufft.h").is_file() and (lib_file := _find_library(Path(library_dir))):
            return include_path, lib_file

    for root in _candidate_cufft_roots():
        if paths := _cufft_paths_from_root(root):
            return paths

    include_dirs = [path for path in _env_paths("CPATH") if (path / "cufft.h").is_file()]
    library_files = [lib_file for lib_dir in _env_paths("LD_LIBRARY_PATH") if (lib_file := _find_library(lib_dir))]
    if include_dirs and library_files:
        return include_dirs[0], library_files[0]

    return None


@lru_cache(maxsize=1)
def _load_cuda_fft_extension():
    _configure_cuda_home()

    source_dir = Path(__file__).resolve().parent / "cuda"
    verbose = os.environ.get("ANEMOI_CUDA_FFT_VERBOSE", os.environ.get("ANEMOI_cuda_rfft_VERBOSE", "0")) == "1"
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3"]
    extra_include_paths: list[str] = []
    extra_ldflags: list[str] = []

    if cufft_paths := _find_cufft_paths():
        include_dir, lib_file = cufft_paths
        extra_cflags.append("-DANEMOI_FFT_ENABLE_CUDA")
        extra_cuda_cflags.append("-DANEMOI_FFT_ENABLE_CUDA")
        extra_include_paths.append(str(include_dir))
        extra_ldflags.extend([str(lib_file), f"-Wl,-rpath,{lib_file.parent}"])

    return load(
        name="anemoi_cuda_fft",
        sources=[
            str(source_dir / "fft.cpp"),
            str(source_dir / "fft.cu"),
        ],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        verbose=verbose,
    )


def _offsets(lons_per_lat: LonsPerLat) -> tuple[int, ...]:
    offsets = []
    offset = 0
    for nlon in lons_per_lat:
        offsets.append(offset)
        offset += nlon
    return tuple(offsets)


@lru_cache(maxsize=None)
def _metadata(lons_per_lat: LonsPerLat, device_index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    # These tensors are the same for every call on a given grid and device.
    device = torch.device("cuda", device_index)
    return (
        torch.tensor(_offsets(lons_per_lat), device=device, dtype=torch.int32),
        torch.tensor(lons_per_lat, device=device, dtype=torch.int32),
        max(lons_per_lat),
    )


class _CUDARFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lons_per_lat: LonsPerLat) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("RFFT CUDA extension requires a CUDA tensor")
        if x.dtype not in (torch.float32, torch.float64):
            raise RuntimeError("RFFT CUDA extension supports float32 and float64")

        lons_per_lat = tuple(int(nlon) for nlon in lons_per_lat)
        grid_points = sum(lons_per_lat)
        if x.shape[-1] != grid_points:
            raise ValueError(f"Expected last dimension to be {grid_points}, got {x.shape[-1]}")

        x_flat = x.contiguous().reshape(-1, grid_points)
        offsets, lons, max_nlon = _metadata(lons_per_lat, _device_index(x))
        out = _load_cuda_fft_extension().forward(x_flat, offsets, lons, max_nlon)

        ctx.input_shape = tuple(x.shape)
        ctx.grid_points = grid_points
        ctx.max_nlon = max_nlon
        ctx.save_for_backward(offsets, lons)
        return out.reshape(*x.shape[:-1], len(lons_per_lat), max_nlon // 2 + 1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        offsets, lons = ctx.saved_tensors
        nlat = lons.numel()
        grad_flat = grad_output.contiguous().reshape(-1, nlat, ctx.max_nlon // 2 + 1)
        grad_x = _load_cuda_fft_extension().backward(
            grad_flat,
            offsets,
            lons,
            ctx.max_nlon,
            ctx.grid_points,
        )
        return grad_x.reshape(ctx.input_shape), None, None, None


class _CUDAIRFFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lons_per_lat: LonsPerLat) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("IRFFT CUDA extension requires a CUDA tensor")
        if x.dtype not in (torch.complex64, torch.complex128):
            raise RuntimeError("IRFFT CUDA extension supports complex64 and complex128")

        lons_per_lat = tuple(int(nlon) for nlon in lons_per_lat)
        nlat = len(lons_per_lat)
        if x.shape[-2] != nlat:
            raise ValueError(f"Expected latitude dimension to be {nlat}, got {x.shape[-2]}")

        grid_points = sum(lons_per_lat)
        nmodes = int(x.shape[-1])
        x_flat = x.contiguous().reshape(-1, nlat, nmodes)
        offsets, lons, max_nlon = _metadata(lons_per_lat, _device_index(x))
        out = _load_cuda_fft_extension().irfft_forward(x_flat, offsets, lons, max_nlon, grid_points)

        ctx.input_shape = tuple(x.shape)
        ctx.grid_points = grid_points
        ctx.max_nlon = max_nlon
        ctx.nmodes = nmodes
        ctx.save_for_backward(offsets, lons)
        return out.reshape(*x.shape[:-2], grid_points)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        offsets, lons = ctx.saved_tensors
        grad_flat = grad_output.contiguous().reshape(-1, ctx.grid_points)
        grad_x = _load_cuda_fft_extension().irfft_backward(
            grad_flat,
            offsets,
            lons,
            ctx.max_nlon,
            ctx.nmodes,
        )
        return grad_x.reshape(ctx.input_shape), None, None


def cuda_rfft(
    x: torch.Tensor,
    lons_per_lat: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Compute ``rfft(norm="forward")`` on a flattened reduced grid."""

    lons_per_lat = tuple(lons_per_lat)
    return _CUDARFFT.apply(x, lons_per_lat)


def cuda_irfft(
    x: torch.Tensor,
    lons_per_lat: list[int] | tuple[int, ...],
) -> torch.Tensor:
    """Compute ``irfft(norm="forward")`` on a reduced grid."""

    lons_per_lat = tuple(lons_per_lat)
    return _CUDAIRFFT.apply(x, lons_per_lat)
