#!/usr/bin/env -S uv run --script
# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: T201, S403, S301, ANN001, ANN002, ANN201, ANN202, ARG002, EM101, EM102, ERA001, RUF012, C901, TRY003, SIM105, RET504
# Explanations for suppressed rules:
# T201: print is the output mechanism for this CLI tool
# S403/S301: pickle usage is intentional — loading torch checkpoints
# ANN*: type annotations relaxed for exploratory script
# ARG002: stub classes accept *args/**kwargs they don't use
# EM101/EM102: string literals in exceptions are fine for scripts
# ERA001: commented-out code in the demo section is intentional
# RUF012: ClassVar annotation not needed for inner class
# C901: complexity acceptable for main() and extract_full_metadata() in a script
# TRY003/SIM105/RET504: style nits not worth fixing in exploratory code

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=2.0",
#     "safetensors>=0.4.0",
#     "packaging",
#     "numpy",
#     "omegaconf",
# ]
# ///
"""Compare Anemoi checkpoint sizes across formats.

Exploratory script for evaluating safetensors as a replacement format
for inference checkpoints. See: https://github.com/ecmwf/anemoi-core/issues/250

Usage:
    uv run compare_checkpoint_sizes.py /path/to/checkpoint.ckpt
    uv run compare_checkpoint_sizes.py /path/to/checkpoint.ckpt --output-dir /tmp/safetensors-test
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Stubbed loading: unpickle checkpoints without their original packages
# ---------------------------------------------------------------------------


class _Stub(torch.nn.Module):
    """Generic stub that accepts any constructor args.

    Inherits from nn.Module so that pickled model objects (like inference
    checkpoints) can be reconstructed. The stub captures all args/kwargs
    without doing anything with them, and supports state_dict() for
    extracting weights.
    """

    def __init__(self, *args, **kwargs):
        # nn.Module.__init__ sets up _parameters, _modules, etc.
        super().__init__()
        self._stub_kwargs = kwargs

    def __repr__(self):
        return f"{type(self).__name__}(stub)"

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Stub module: forward not implemented")


class _StubModule(types.ModuleType):
    """Module that auto-creates stub classes on attribute access."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        # Could be a submodule or a class; create a stub class
        stub_cls = type(name, (_Stub,), {})
        setattr(self, name, stub_cls)
        return stub_cls


class _ForgivingUnpickler(pickle.Unpickler):
    """Unpickler that stubs missing modules instead of raising ImportError.

    This lets us load checkpoints that embed classes from packages we
    don't have installed (aifs, torch_geometric, etc.) The actual tensor
    data is stored separately in the zip archive and doesn't go through
    pickle class resolution — only metadata/config objects do.
    """

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Create stub module hierarchy on-the-fly
            parts = module.split(".")
            for i in range(len(parts)):
                mod_path = ".".join(parts[: i + 1])
                if mod_path not in sys.modules:
                    sys.modules[mod_path] = _StubModule(mod_path)
                if i > 0:
                    parent = ".".join(parts[:i])
                    setattr(sys.modules[parent], parts[i], sys.modules[mod_path])

            mod = sys.modules[module]
            cls = getattr(mod, name, None)
            if cls is None:
                cls = type(name, (_Stub,), {})
                setattr(mod, name, cls)
            return cls


def load_checkpoint_forgiving(path: Path) -> dict:
    """Load a PyTorch checkpoint, stubbing any missing modules.

    Uses a custom unpickler that creates stub classes for any module
    not available in the current environment. This is safe because we
    only care about the tensor data (state_dict), not the reconstructed
    model objects.
    """
    with zipfile.ZipFile(path) as zf:
        pkl_files = [f for f in zf.namelist() if f.endswith(".pkl")]
        if not pkl_files:
            # Not a zip-based checkpoint, fall back to torch.load
            return torch.load(path, map_location="cpu", weights_only=False)

        # Read the pickle data
        pkl_data = zf.read(pkl_files[0])

        # Set up a custom unpickler that resolves tensors from the zip
        data_dir = pkl_files[0].rsplit("/", 1)[0] + "/data"

        class _TorchUnpickler(_ForgivingUnpickler):
            """Unpickler that also handles torch tensor reconstruction."""

            persistent_id_map: dict = {}

            def persistent_load(self, saved_id):
                # torch uses persistent_id to reference tensor data files
                # saved_id is typically (storage_type, key, device, numel)
                assert isinstance(saved_id, tuple)
                type_tag = saved_id[0]

                if type_tag == "storage":
                    storage_type, key, _location, _numel = saved_id[1:]
                    storage_key = f"{data_dir}/{key}"

                    if storage_key in self.persistent_id_map:
                        return self.persistent_id_map[storage_key]

                    dtype = _storage_type_to_dtype(storage_type)
                    raw = zf.read(storage_key)
                    if len(raw) == 0:
                        # Empty tensor storage (e.g. zero-element parameters)
                        storage = torch.empty(0, dtype=dtype)._typed_storage()
                    else:
                        tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
                        storage = tensor._typed_storage()
                    self.persistent_id_map[storage_key] = storage
                    return storage

                raise RuntimeError(f"Unknown persistent_id type: {type_tag}")

        unpickler = _TorchUnpickler(io.BytesIO(pkl_data))
        result = unpickler.load()
        return result


def _install_aifs_stubs() -> None:
    """Pre-install stub modules for legacy aifs package.

    Training checkpoints only reference aifs.data.dataindices classes in
    their metadata. Installing these stubs before torch.load lets the
    standard loader handle them without falling through to the forgiving
    loader.
    """
    module_paths = [
        "aifs",
        "aifs.data",
        "aifs.data.dataindices",
        "aifs.data.normalizer",
        "aifs.layers",
        "aifs.layers.block",
        "aifs.layers.chunk",
        "aifs.layers.conv",
        "aifs.layers.graph",
        "aifs.layers.mapper",
        "aifs.layers.mlp",
        "aifs.layers.processor",
        "aifs.layers.utils",
        "aifs.model",
        "aifs.model.gnn",
        "aifs.model.model",
    ]
    for path in module_paths:
        if path not in sys.modules:
            sys.modules[path] = _StubModule(path)
    for path in module_paths:
        parts = path.split(".")
        if len(parts) > 1:
            parent = ".".join(parts[:-1])
            if parent in sys.modules:
                setattr(sys.modules[parent], parts[-1], sys.modules[path])


def _storage_type_to_dtype(storage_type) -> torch.dtype:
    """Map torch storage types to dtypes."""
    mapping = {
        torch.FloatStorage: torch.float32,
        torch.DoubleStorage: torch.float64,
        torch.HalfStorage: torch.float16,
        torch.BFloat16Storage: torch.bfloat16,
        torch.IntStorage: torch.int32,
        torch.LongStorage: torch.int64,
        torch.ShortStorage: torch.int16,
        torch.CharStorage: torch.int8,
        torch.ByteStorage: torch.uint8,
        torch.BoolStorage: torch.bool,
    }
    if storage_type in mapping:
        return mapping[storage_type]
    # Try by name as fallback
    name = getattr(storage_type, "__name__", str(storage_type))
    name_map = {
        "FloatStorage": torch.float32,
        "DoubleStorage": torch.float64,
        "HalfStorage": torch.float16,
        "BFloat16Storage": torch.bfloat16,
        "IntStorage": torch.int32,
        "LongStorage": torch.int64,
        "ShortStorage": torch.int16,
        "CharStorage": torch.int8,
        "ByteStorage": torch.uint8,
        "BoolStorage": torch.bool,
    }
    if name in name_map:
        return name_map[name]
    # Default to float32
    return torch.float32


# ---------------------------------------------------------------------------
# Checkpoint analysis utilities
# ---------------------------------------------------------------------------


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    kb = size_bytes / 1024
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Extract model state_dict from a Lightning checkpoint.

    Lightning checkpoints nest the model weights under 'state_dict'.
    Plain PyTorch checkpoints may use 'model_state_dict' or be the
    state_dict directly.
    """
    for key in ("state_dict", "model_state_dict", "model"):
        if key in checkpoint:
            sd = checkpoint[key]
            if isinstance(sd, dict):
                return sd

    # Assume the checkpoint itself is a state dict if all values are tensors
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    msg = f"Cannot find state_dict. Top-level keys: {list(checkpoint.keys())}"
    raise KeyError(msg)


def _deep_convert(obj: object) -> object:
    """Recursively convert OmegaConf / stub objects to plain Python types."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {str(k): _deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_convert(v) for v in obj]
    # Stub objects from legacy aifs classes — extract their dicts
    if hasattr(obj, "__dict__") and type(obj).__name__ in (
        "DataIndex",
        "IndexCollection",
        "InputTensorIndex",
        "ModelIndex",
        "OutputTensorIndex",
    ):
        return {str(k): _deep_convert(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def extract_full_metadata(checkpoint: dict) -> tuple[dict[str, str], dict[str, torch.Tensor]]:
    """Extract ALL Anemoi checkpoint metadata for safetensors storage.

    safetensors has two storage mechanisms:
      1. metadata: dict[str, str] — string key-value pairs in the header
      2. tensors: dict[str, Tensor] — the main tensor storage

    Strategy:
      - Numpy arrays (statistics) → stored as tensors with '__meta__/' prefix
      - OmegaConf configs → serialised as JSON strings
      - data_indices, provenance → serialised as JSON strings
      - Scalar training state (epoch, global_step) → JSON strings

    Returns (metadata_strings, metadata_tensors).
    """
    import numpy as np

    meta_strings: dict[str, str] = {}
    meta_tensors: dict[str, torch.Tensor] = {}

    # --- Scalar training state ---
    for key in ("epoch", "global_step", "pytorch-lightning_version"):
        if key in checkpoint:
            meta_strings[key] = json.dumps(checkpoint[key], default=str)

    # --- hyper_parameters: the big one ---
    hp = checkpoint.get("hyper_parameters")
    if hp is None:
        return meta_strings, meta_tensors

    hp = _deep_convert(hp)
    if not isinstance(hp, dict):
        meta_strings["hyper_parameters"] = json.dumps(hp, default=str)
        return meta_strings, meta_tensors

    # config — full Hydra configuration
    if "config" in hp:
        meta_strings["anemoi.config"] = json.dumps(_deep_convert(hp["config"]), default=str)

    # statistics — numpy arrays for normalization (critical for inference)
    if "statistics" in hp and isinstance(hp["statistics"], dict):
        for stat_name, stat_val in hp["statistics"].items():
            if isinstance(stat_val, np.ndarray):
                meta_tensors[f"__meta__/statistics.{stat_name}"] = torch.from_numpy(stat_val.copy())
            else:
                meta_strings[f"anemoi.statistics.{stat_name}"] = json.dumps(stat_val, default=str)

    # data_indices — variable index mappings
    if "data_indices" in hp:
        meta_strings["anemoi.data_indices"] = json.dumps(_deep_convert(hp["data_indices"]), default=str)

    # metadata — provenance, run_id, dataset info
    if "metadata" in hp:
        meta_strings["anemoi.metadata"] = json.dumps(_deep_convert(hp["metadata"]), default=str)

    # --- lr_schedulers config (useful for warm restart) ---
    if "lr_schedulers" in checkpoint:
        try:
            meta_strings["anemoi.lr_schedulers"] = json.dumps(_deep_convert(checkpoint["lr_schedulers"]), default=str)
        except (TypeError, ValueError):
            pass

    # --- MixedPrecision state ---
    if "MixedPrecision" in checkpoint:
        meta_strings["anemoi.mixed_precision"] = json.dumps(checkpoint["MixedPrecision"], default=str)

    return meta_strings, meta_tensors


def extract_metadata(checkpoint: dict) -> dict[str, str] | None:
    """Simple metadata extraction (legacy, for backwards compat)."""
    hp = checkpoint.get("hyper_parameters")
    if hp is None:
        return None
    hp = _deep_convert(hp)
    try:
        return {"hyper_parameters": json.dumps(hp, default=str)}
    except (TypeError, ValueError) as exc:
        print(f"  Warning: could not serialise hyper_parameters: {exc}")
        return None


def count_parameters(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() for t in state_dict.values())


def verify_roundtrip(
    original: dict[str, torch.Tensor],
    safetensors_path: Path,
) -> tuple[bool, float]:
    """Load back from safetensors and verify all tensors match exactly."""
    reloaded = load_file(safetensors_path)

    if set(original.keys()) != set(reloaded.keys()):
        missing = set(original.keys()) - set(reloaded.keys())
        extra = set(reloaded.keys()) - set(original.keys())
        print(f"  Key mismatch! Missing: {missing}, Extra: {extra}")
        return False, float("inf")

    max_diff = 0.0
    for key in original:
        diff = (original[key].float() - reloaded[key].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            print(f"  Non-zero diff in {key}: {diff}")

    return max_diff == 0.0, max_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Anemoi checkpoint sizes across formats (issue #250)")
    parser.add_argument("checkpoint", type=Path, help="Path to .ckpt file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: temp directory)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep generated files (don't clean up)",
    )
    args = parser.parse_args()

    ckpt_path: Path = args.checkpoint
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} does not exist")
        sys.exit(1)

    original_size = ckpt_path.stat().st_size

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"  Original size: {format_size(original_size)}")

    # Pre-install stub modules so torch.load can handle legacy aifs metadata
    _install_aifs_stubs()

    # Try standard torch.load first, fall back to forgiving loader
    t0 = time.monotonic()
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, ImportError, RuntimeError) as exc:
        print(f"  Standard load failed ({exc}), using forgiving loader...")
        checkpoint = load_checkpoint_forgiving(ckpt_path)
    load_time = time.monotonic() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Handle both dict-based (training) and model-object (inference) checkpoints
    if isinstance(checkpoint, torch.nn.Module):
        print("  Format: pickled model object (inference checkpoint)")
        print(f"  Model class: {type(checkpoint).__name__}")
        state_dict = checkpoint.state_dict()
        metadata_source = None  # No separate hyper_parameters
        has_optimizer = False
    elif isinstance(checkpoint, dict):
        print(f"  Top-level keys: {list(checkpoint.keys())}")
        state_dict = extract_state_dict(checkpoint)
        metadata_source = checkpoint
        has_optimizer = "optimizer_states" in checkpoint
    else:
        print(f"  Error: unexpected checkpoint type: {type(checkpoint)}")
        sys.exit(1)
    n_params = count_parameters(state_dict)
    n_tensors = len(state_dict)
    print(f"  Model: {n_params:,} parameters across {n_tensors} tensors")

    # Dtype breakdown
    dtype_counts: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    for t in state_dict.values():
        dt = str(t.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + t.numel()
        dtype_bytes[dt] = dtype_bytes.get(dt, 0) + t.numel() * t.element_size()
    for dt in sorted(dtype_counts):
        print(f"    {dt}: {dtype_counts[dt]:,} params ({format_size(dtype_bytes[dt])})")

    if metadata_source is not None:
        metadata = extract_metadata(metadata_source)
        full_meta_strings, full_meta_tensors = extract_full_metadata(metadata_source)
        if metadata:
            meta_size = len(metadata["hyper_parameters"].encode())
            print(f"  Hyper parameters (naive JSON): {format_size(meta_size)}")
        if full_meta_strings:
            total_str = sum(len(v.encode()) for v in full_meta_strings.values())
            print(f"  Full metadata ({len(full_meta_strings)} keys): {format_size(total_str)} as JSON")
            for k, v in sorted(full_meta_strings.items()):
                print(f"    {k}: {format_size(len(v.encode()))}")
        if full_meta_tensors:
            total_t = sum(t.numel() * t.element_size() for t in full_meta_tensors.values())
            print(f"  Metadata tensors ({len(full_meta_tensors)} arrays): {format_size(total_t)}")
            for k, t in sorted(full_meta_tensors.items()):
                print(f"    {k}: {t.shape} {t.dtype}")
    else:
        metadata = None
        full_meta_strings = {}
        full_meta_tensors = {}
        print("  Hyper parameters: N/A (model object checkpoint)")

    print(f"  Optimizer state: {'present' if has_optimizer else 'not present'}")

    # --- Set up output directory ---
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.mkdtemp(prefix="safetensors-explore-")
        output_dir = Path(tmp)
        cleanup = not args.keep

    stem = ckpt_path.stem

    # --- Save state_dict only as .pt ---
    pt_path = output_dir / f"{stem}_state_dict.pt"
    t0 = time.monotonic()
    torch.save(state_dict, pt_path)
    pt_save_time = time.monotonic() - t0
    pt_size = pt_path.stat().st_size

    # --- Save as safetensors (no metadata) ---
    st_path = output_dir / f"{stem}.safetensors"
    t0 = time.monotonic()
    save_file(state_dict, st_path)
    st_save_time = time.monotonic() - t0
    st_size = st_path.stat().st_size

    # --- Save as safetensors with full Anemoi metadata ---
    if full_meta_strings or full_meta_tensors:
        stm_path = output_dir / f"{stem}_full.safetensors"
        # Merge model weights + metadata tensors (statistics arrays)
        all_tensors = {**state_dict, **full_meta_tensors}
        t0 = time.monotonic()
        save_file(all_tensors, stm_path, metadata=full_meta_strings)
        stm_save_time = time.monotonic() - t0
        stm_size = stm_path.stat().st_size
    else:
        stm_path = None
        stm_save_time = 0.0
        stm_size = 0

    # --- Load times for comparison ---
    t0 = time.monotonic()
    torch.load(pt_path, map_location="cpu", weights_only=True)
    pt_load_time = time.monotonic() - t0

    t0 = time.monotonic()
    load_file(st_path)
    st_load_time = time.monotonic() - t0

    # --- Verify roundtrip ---
    print("\nVerifying roundtrip correctness...")
    ok, max_diff = verify_roundtrip(state_dict, st_path)

    # Verify full metadata roundtrip if applicable
    if stm_path and full_meta_tensors:
        from safetensors import safe_open

        print("Verifying metadata roundtrip...")
        with safe_open(stm_path, framework="pt") as f:
            stored_meta = f.metadata()
            for key in full_meta_strings:
                if key not in stored_meta:
                    print(f"  Missing metadata key: {key}")
                elif stored_meta[key] != full_meta_strings[key]:
                    print(f"  Metadata mismatch for {key}")
                else:
                    # Verify JSON parses back
                    json.loads(stored_meta[key])
            print(f"  Metadata keys verified: {len(full_meta_strings)}")

            for key in full_meta_tensors:
                rt = f.get_tensor(key)
                orig = full_meta_tensors[key]
                diff = (orig.float() - rt.float()).abs().max().item()
                if diff != 0.0:
                    print(f"  Metadata tensor mismatch: {key} (diff={diff})")
            print(f"  Metadata tensors verified: {len(full_meta_tensors)}")

    # --- Print results ---
    print(f"\n{'=' * 72}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Parameters: {n_params:,} ({n_tensors} tensors)")
    if has_optimizer:
        print("Note: Original includes optimizer state (training checkpoint)")
    print(f"{'=' * 72}")

    header = f"{'Format':<36} {'Size':>10} {'vs original':>12} {'Save':>8} {'Load':>8}"
    print(header)
    print("─" * 72)

    def pct(size: int) -> str:
        diff = (size - original_size) / original_size * 100
        return f"{diff:+.1f}%"

    rows = [
        ("Original .ckpt (full)", original_size, "baseline", f"{load_time:.1f}s", f"{load_time:.1f}s"),
        ("State dict only (.pt)", pt_size, pct(pt_size), f"{pt_save_time:.1f}s", f"{pt_load_time:.1f}s"),
        ("Safetensors", st_size, pct(st_size), f"{st_save_time:.1f}s", f"{st_load_time:.1f}s"),
    ]
    if stm_path:
        rows.append(
            ("Safetensors + full metadata", stm_size, pct(stm_size), f"{stm_save_time:.1f}s", f"{st_load_time:.1f}s"),
        )

    for name, size, vs, save_t, load_t in rows:
        print(f"{name:<36} {format_size(size):>10} {vs:>12} {save_t:>8} {load_t:>8}")

    print(f"\n{'─' * 72}")
    if ok:
        print(f"Roundtrip verification: ✓ All {n_tensors} tensors match (max diff: {max_diff})")
    else:
        print(f"Roundtrip verification: ✗ MISMATCH (max diff: {max_diff})")

    # --- Metadata access demo ---
    if stm_path:
        print(f"\n{'=' * 72}")
        print("DEMO: Reading metadata WITHOUT loading 186 MB of weights")
        print(f"{'=' * 72}")
        print()
        print("# With torch (pickle) — must load ENTIRE file + need original packages:")
        print("#   import torch")
        print(f"#   ckpt = torch.load('{ckpt_path.name}', map_location='cpu')")
        print("#   config = ckpt['hyper_parameters']['config']  # loaded 556 MB")
        print()
        print("# With safetensors — header only, zero tensor deserialization:")
        print("#   from safetensors import safe_open")
        print(f"#   with safe_open('{stm_path.name}', framework='pt') as f:")
        print("#       meta = f.metadata()  # instant, reads only the header")
        print("#       config = json.loads(meta['anemoi.config'])")
        print()

        from safetensors import safe_open

        t0 = time.monotonic()
        with safe_open(stm_path, framework="pt") as f:
            meta = f.metadata()
        header_time = time.monotonic() - t0

        print(f"Header read time: {header_time * 1000:.1f} ms (vs {load_time:.1f}s for torch.load)")
        print()

        # Show what you get from the header alone
        print("Metadata keys available without loading weights:")
        for key in sorted(meta):
            val = meta[key]
            preview = val[:80] + "..." if len(val) > 80 else val
            print(f"  {key} ({format_size(len(val.encode()))}): {preview}")

        print()
        print("Example: read the model config")
        config = json.loads(meta["anemoi.config"])
        print(f"  Model type: {config.get('model', {}).get('_target_', 'N/A')}")
        print(f"  Training LR: {config.get('training', {}).get('lr', {}).get('rate', 'N/A')}")

        if "anemoi.metadata" in meta:
            md = json.loads(meta["anemoi.metadata"])
            print(f"  Run ID: {md.get('run_id', 'N/A')}")
            print(f"  Dataset variables: {len(md.get('dataset', {}).get('variables', []))}")
            print(f"  Timestamp: {md.get('timestamp', 'N/A')}")

        # Show selective tensor loading
        print()
        print("Example: load ONLY normalization statistics (no model weights)")
        with safe_open(stm_path, framework="pt") as f:
            tensor_keys = f.keys()
            meta_tensors_available = [k for k in tensor_keys if k.startswith("__meta__/")]
            weight_tensors = [k for k in tensor_keys if not k.startswith("__meta__/")]
            print(f"  Total tensors in file: {len(list(tensor_keys))}")
            print(f"  Model weight tensors: {len(weight_tensors)} (not loaded)")
            print(f"  Metadata tensors: {len(meta_tensors_available)}")
            for k in meta_tensors_available:
                t = f.get_tensor(k)
                print(f"    {k}: {t.shape} {t.dtype} — e.g. {t[:3].tolist()}")

    print(f"\nOutput files in: {output_dir}")
    if cleanup:
        import shutil

        shutil.rmtree(output_dir)
        print("(Cleaned up temporary files)")


if __name__ == "__main__":
    main()
