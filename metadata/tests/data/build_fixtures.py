# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Rebuild the golden metadata fixtures in this directory.

Usage::

    uv run tests/data/build_fixtures.py --sources <dir>

``<dir>`` holds a checkout of anemoi-inference ``main`` as ``inf-main/`` (its
``tests/unit/checkpoints`` and ``tests/integration/*/metadata.json`` files are the
sources). See ``README.md`` for which fixture comes from which file and which are
synthetic. Fixtures are deterministic: running the script twice gives identical files.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Strings that must not appear in a fixture (paths, user and host names, e-mail).
FORBIDDEN = ("/home/", "/Users/", "/perm/", "/pfs/", "/scratch", "file://", "@", "ecm", "lustre")

#: Stable fake identifiers used by synthetic fixtures.
PARENT_UUID = "0dc0e962-102b-47dd-9f35-2e3dbd285b9a"
F1_RUN_ID = "11111111-2222-4333-8444-555555555555"

#: Keys that only add bulk to the nested ``specific`` / ``attrs`` chains of ``dataset``.
NESTED_DROP = {
    "variables",
    "variables_metadata",
    "statistics",
    "statistics_start_date",
    "statistics_end_date",
    "recipe",
    "history",
    "provenance_load",
    "origins",
    "order_by",
    "latest_write_timestamp",
    "total_size",
    "total_number_of_files",
    "uuid",
    "description",
    "licence",
    "attribution",
    "chunks",
    "missing_dates",
    "variables_with_nans",
    "remapping",
    "proj_string",
    "updated",
    "indices",
    "last_date",
}

TRAINING_KEEP = (
    "run_id",
    "seed",
    "multistep_input",
    "multistep_output",
    "precision",
    "rollout",
    "explicit_times",
    "target_forcing",
    "method",
    "model_task",
    "deterministic",
    "ensemble_size_per_device",
    "noise_initial_conditions",
)

#: Pre- and post-processing blocks of ``config.data`` (normaliser settings and the like).
DATA_PROCESSING_KEYS = ("processors", "normalizer", "imputer", "const_imputer", "conditional_nan")

#: ``config`` groups that are kept; every other group (model, graph, diagnostics, hardware, ...) is dropped.
CONFIG_KEEP = ("data", "training", "dataloader", "task", "datamodule")

DATALOADER_KEEP = ("batch_size", "training", "validation", "test", "dataset", "validation_rollout", "grid_indices")


def _scrub_str(value: str) -> str:
    """Replace absolute paths and URLs to local files by their last component."""
    if value.startswith(("/", "file://")):
        return ".../" + value.rstrip("/").rsplit("/", 1)[-1]
    return value


def scrub(obj: Any) -> Any:
    """Recursively scrub absolute paths from strings."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        return _scrub_str(obj)
    return obj


def trim_nested(node: Any) -> Any:
    """Drop bulky keys from the nested dataset description, keeping its structure."""
    if isinstance(node, dict):
        return {k: trim_nested(v) for k, v in node.items() if k not in NESTED_DROP}
    if isinstance(node, list):
        return [trim_nested(v) for v in node]
    return node


def trim_dataset(ds: dict) -> dict:
    """Trim one ``dataset.<name>`` block; top-level keys are kept, nested bulk is dropped."""
    out = {}
    for key, value in ds.items():
        if key in ("specific", "sources", "arguments"):
            out[key] = scrub(trim_nested(value))
        else:
            out[key] = scrub(value)
    return out


def trim_config(config: dict) -> dict:
    """Keep the config keys adapters can read; drop scalers, losses, optimiser, strategy..."""
    out: dict[str, Any] = {}
    for group in CONFIG_KEEP:
        if group not in config:
            continue
        value = copy.deepcopy(config[group])
        if group == "data":
            for key in DATA_PROCESSING_KEYS:
                value.pop(key, None)
            for ds_cfg in (value.get("datasets") or {}).values():
                for key in DATA_PROCESSING_KEYS:
                    ds_cfg.pop(key, None)
        elif group == "training":
            value = {k: value[k] for k in TRAINING_KEEP if k in value}
        elif group == "dataloader":
            value = {k: value[k] for k in DATALOADER_KEEP if k in value}
        out[group] = value
    return scrub(out)


def provenance_stub(source: dict | None, time: str, python: str) -> dict:
    """Small representative ``provenance_training`` (keys as written by gather_provenance_info)."""
    versions: dict[str, Any] = {}
    for name, value in ((source or {}).get("module_versions") or {}).items():
        if name in ("anemoi.datasets", "anemoi.models", "anemoi.training", "anemoi.utils", "numpy", "torch"):
            versions[name] = {"version": value["version"]} if isinstance(value, dict) else value
    for name, value in versions.items():
        # Drop local version suffixes ("+rocm...") that identify the build machine.
        if isinstance(value, dict):
            value["version"] = value["version"].split("+")[0]
        else:
            versions[name] = value.split("+")[0]
    if not versions:
        versions = {"anemoi.training": "0.0.0", "anemoi.models": "0.0.0", "anemoi.utils": "0.0.0"}
    return {
        "time": time,
        "python": python,
        "module_versions": versions,
        "distribution_names": {},
        "git_versions": {
            "anemoi.training": {"git": {"sha1": "0" * 40, "modified_files": 0, "untracked_files": 0}},
        },
    }


def trim_common(src: dict, flat_dataset: bool = False) -> dict:
    """Return a copy of a training-written metadata dict with bulk removed and paths scrubbed.

    ``flat_dataset`` is True for F0/F1 where ``dataset`` is one dataset's metadata, not keyed by name.
    """
    out = {}
    for key, value in src.items():
        if key == "config":
            out[key] = trim_config(value)
        elif key == "dataset":
            out[key] = trim_dataset(value) if flat_dataset else {name: trim_dataset(ds) for name, ds in value.items()}
        elif key == "provenance_training":
            out[key] = provenance_stub(value, value.get("time", ""), value.get("python", ""))
        elif key == "model" and "summary" in value:
            out[key] = {**value, "summary": "(torchinfo summary removed)"}
        else:
            out[key] = scrub(copy.deepcopy(value))
    return out


# ---------------------------------------------------------------------------
# Compact, diff-friendly JSON writer
# ---------------------------------------------------------------------------


def _inline(obj: Any) -> str:
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=False)


def _is_flat(obj: Any) -> bool:
    if isinstance(obj, dict):
        return all(not isinstance(v, (dict, list)) for v in obj.values())
    if isinstance(obj, list):
        return all(not isinstance(v, (dict, list)) for v in obj)
    return True


def dumps(obj: Any, level: int = 0, width: int = 110) -> str:
    """Dump JSON with one key per line, except small or flat containers which stay on one line."""
    text = _inline(obj)
    if not isinstance(obj, (dict, list)) or not obj:
        return text
    if level > 0 and (len(text) <= width or _is_flat(obj)):
        return text
    pad = " " * (level + 1)
    if isinstance(obj, dict):
        items = [f"{pad}{json.dumps(k)}: {dumps(v, level + 1, width)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
    items = [f"{pad}{dumps(v, level + 1, width)}" for v in obj]
    return "[\n" + ",\n".join(items) + "\n" + " " * level + "]"


def write(name: str, metadata: dict) -> None:
    """Write one fixture after checking that it is free of paths, names and e-mail addresses."""
    text = dumps(metadata) + "\n"
    for bad in FORBIDDEN:
        if bad in text:
            idx = text.index(bad)
            msg = f"{name}: forbidden string {bad!r} near: {text[max(0, idx - 60) : idx + 60]!r}"
            raise SystemExit(msg)
    json.loads(text)
    (HERE / name).write_text(text)
    print(f"{name:34s} {len(text) / 1024:6.1f} KB")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


#: Public or own checkpoints that real fixtures are read from (key -> file name inside ``--checkpoint-dir``).
CHECKPOINT_FILES = {
    "ens": "aifs-ens-crps-2.0.ckpt",
    "subs": "aifs-subs-crps-0.1.ckpt",
}

#: Fixtures built from one real F1 checkpoint (fixture -> checkpoint key).
F1_FROM_CHECKPOINT = {
    "f1_single_dataset.json": "subs",
    "f1_ensemble.json": "ens",
}

INF_UNIT = "inf-main/tests/unit/checkpoints"
INF_INTEGRATION = "inf-main/tests/integration"

#: Metadata JSON of a real anemoi-training main run (tests/integration test_metadata), relative to ``--sources``.
REAL_F6 = "real-f6/inference-last.ckpt.json"

#: Same for anemoi-training at commit 127c10618 (2026-10-06, before the task moved into metadata_inference).
REAL_F4 = "real-f4/inference-last.ckpt.json"

#: Fixture -> real source: ``("checkpoint", key)`` or ``("file", path relative to --sources)``.
REAL_SOURCES: dict[str, tuple[str, str]] = {
    **{name: ("checkpoint", key) for name, key in F1_FROM_CHECKPOINT.items()},
    "f1_null_diagnostic.json": ("file", f"{INF_UNIT}/ocean.json"),
    "f2_dev_scalar_index.json": ("file", f"{INF_UNIT}/multi-single.json"),
    "f3_single_dataset.json": ("file", f"{INF_INTEGRATION}/global-multi-steps-2in-2out/metadata.json"),
    "f3_multi_dataset.json": ("file", f"{INF_INTEGRATION}/era5-cerra/metadata.json"),
    "f3_temporal_downscaler.json": ("file", f"{INF_INTEGRATION}/temporal-downscaler/metadata.json"),
    "f4_task_top_level.json": ("file", REAL_F4),
    "f5_offsets.json": ("file", f"{INF_INTEGRATION}/rmi-lam/metadata.json"),
    "f6_main.json": ("file", REAL_F6),
}


def load_checkpoint_metadata(path: Path) -> dict:
    """Read the metadata JSON (``anemoi.json`` or ``ai-models.json``) from a checkpoint zip. Nothing else is read."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(("anemoi.json", "ai-models.json"))]
        if len(names) != 1:
            msg = f"{path.name}: expected one metadata entry, found {names}"
            raise ValueError(msg)
        return json.loads(archive.read(names[0]))


# ---------------------------------------------------------------------------
# Structure check: fixture key paths against the real source
# ---------------------------------------------------------------------------

#: Keys whose every occurrence, at any depth of ``dataset``, must survive trimming.
DEEP_KEYS = ("data_request", "constant_fields", "frequency")

#: Segments after which dropping ``NESTED_DROP`` keys is intended.
NESTED_SEGMENTS = ("specific", "sources", "arguments")


def key_paths(obj: Any, names: frozenset[str], pre: str = "", depth: int = 0, out: dict | None = None) -> dict:
    """Map key paths to value type names.

    Dataset names are replaced by ``<ds>``, dicts with more than 25 keys collapse to ``.{many}``, ``config`` is
    followed to depth 2 only, other keys to depth 6, and lists are described by their first element.
    """
    out = {} if out is None else out
    if isinstance(obj, dict):
        if len(obj) > 25 and depth > 0:
            out[pre + ".{many}"] = "dict"
            return out
        for key, value in obj.items():
            path = f"{pre}.{'<ds>' if key in names else key}" if pre else key
            if depth >= 6 or (pre.startswith("config") and depth >= 2):
                out[path] = type(value).__name__
            else:
                key_paths(value, names, path, depth + 1, out)
    elif isinstance(obj, list):
        out[pre + "[]"] = "list:" + (type(obj[0]).__name__ if obj else "")
        if obj and isinstance(obj[0], dict) and depth < 6:
            key_paths(obj[0], names, pre + "[0]", depth + 1, out)
    else:
        out[pre] = type(obj).__name__
    return out


def deep_paths(obj: Any, keys: tuple[str, ...], pre: str = "") -> set[str]:
    """Full paths (list indices collapsed to ``[]``) of every occurrence of ``keys`` at any depth."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{pre}.{key}"
            if key in keys:
                found.add(path)
            found |= deep_paths(value, keys, path)
    elif isinstance(obj, list):
        for value in obj:
            found |= deep_paths(value, keys, pre + "[]")
    return found


def is_intended_drop(path: str) -> bool:
    """True if trimming removes this source key path on purpose (bulk that no adapter reads)."""
    collapsed = path.endswith(".{many}")
    parts = [p.removesuffix("[0]").removesuffix("[]") for p in path.removesuffix(".{many}").split(".")]
    if parts[0] == "provenance_training":
        return True
    if parts[0] == "config" and len(parts) > 1:
        group = parts[1]
        if group not in CONFIG_KEEP:
            return True
        if collapsed and len(parts) == 2:
            return True  # a big group; its kept keys are compared one by one in check_fixture
        if len(parts) > 2:
            if group == "data":
                return parts[2] in DATA_PROCESSING_KEYS
            if group == "training":
                return parts[2] not in TRAINING_KEEP
            if group == "dataloader":
                return parts[2] not in DATALOADER_KEEP
    if parts[0] == "dataset" and any(p in NESTED_SEGMENTS for p in parts):
        # Inside the nested chains, bulk keys go; a dict that was above the 25-key collapse limit may now be below it.
        return collapsed or parts[-1] in NESTED_DROP
    return False


def check_fixture(source: dict, fixture: dict) -> list[str]:
    """Return the problems found when comparing the fixture with its real source (empty if none)."""
    names = frozenset((source.get("metadata_inference") or {}).get("dataset_names") or [])
    real, built = key_paths(source, names), key_paths(fixture, names)
    problems = [f"missing key path {p} ({t})" for p, t in real.items() if p not in built and not is_intended_drop(p)]
    problems += [
        f"type differs at {p}: source {t}, fixture {built[p]}"
        for p, t in real.items()
        if p in built and built[p] != t and not is_intended_drop(p)
    ]
    keep = {"data": None, "training": TRAINING_KEEP, "dataloader": DATALOADER_KEEP, "task": None, "datamodule": None}
    for group, keys in keep.items():
        src_group = (source.get("config") or {}).get(group)
        if isinstance(src_group, dict):
            fix_group = (fixture.get("config") or {}).get(group) or {}
            for k, v in src_group.items():
                if (keys is None and not (group == "data" and k in DATA_PROCESSING_KEYS)) or (keys and k in keys):
                    if k not in fix_group:
                        problems.append(f"missing config.{group}.{k}")
                    elif type(fix_group[k]) is not type(v):
                        problems.append(f"type differs at config.{group}.{k}")
    for key in ("dataset",):
        if key in source:
            lost = deep_paths(source[key], DEEP_KEYS) - deep_paths(fixture.get(key, {}), DEEP_KEYS)
            # A key below a bulk key that trimming removes (e.g. recipe.dates.frequency) goes with it.
            lost = {p for p in lost if not any(part in NESTED_DROP for part in p.split("."))}
            problems += [f"{key}: deep key {p} lost" for p in sorted(lost)]
    return problems


def run_check(sources: Path | None, checkpoints: dict[str, Path]) -> int:
    """Compare every fixture that has a real source; return the number of failing fixtures."""
    failures = 0
    for name, (kind, ref) in REAL_SOURCES.items():
        fixture_path = HERE / name
        if not fixture_path.is_file():
            print(f"FAIL {name}: fixture file missing")
            failures += 1
            continue
        try:
            if kind == "checkpoint":
                if ref not in checkpoints:
                    print(f"SKIP {name}: checkpoint {CHECKPOINT_FILES[ref]} not given")
                    continue
                source = load_checkpoint_metadata(checkpoints[ref])
            else:
                if sources is None or not (sources / ref).is_file():
                    print(f"SKIP {name}: source {ref} not found")
                    continue
                source = load(sources / ref)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            print(f"SKIP {name}: cannot open source ({error})")
            continue
        problems = check_fixture(source, load(fixture_path))
        print(f"{'FAIL' if problems else 'ok  '} {name}")
        for problem in problems:
            print(f"       {problem}")
        failures += bool(problems)
    return failures


# ---------------------------------------------------------------------------
# Index collections (replicates anemoi-models IndexCollection.todict for the metadata)
# ---------------------------------------------------------------------------


def index_collection(names: list[str], prognostic: list[str], forcing: list[str], diagnostic: list[str]) -> dict:
    """Build ``data_indices`` (data/model x input/output) for variables listed in dataset order."""
    n2i = {n: i for i, n in enumerate(names)}

    def idx(group: list[str], mapping: dict[str, int]) -> list[int]:
        return sorted(mapping[n] for n in names if n in group and n in mapping)

    def block(mapping: dict[str, int], full: list[str]) -> dict:
        return {
            "prognostic": idx(prognostic, mapping),
            "diagnostic": idx(diagnostic, mapping),
            "forcing": idx(forcing, mapping),
            "target": [],
            "full": sorted(mapping[n] for n in full if n in mapping),
            "name_to_index": dict(mapping),
        }

    model_in = {n: i for i, n in enumerate(x for x in names if x not in diagnostic)}
    model_out = {n: i for i, n in enumerate(x for x in names if x not in forcing)}
    return {
        "data": {
            "input": block(n2i, [n for n in names if n not in diagnostic]),
            "output": block(n2i, [n for n in names if n not in forcing]),
        },
        "model": {
            "input": block(model_in, list(model_in)),
            "output": block(model_out, list(model_out)),
        },
    }


def check_index_collection(md: dict) -> None:
    """Self-test: ``index_collection`` reproduces the real rmi-lam data_indices."""
    ds = "data"
    types = md["metadata_inference"][ds]["variable_types"]
    names = list(md["data_indices"][ds]["data"]["input"]["name_to_index"])
    built = index_collection(names, types["prognostic"], types["forcing"], types["diagnostic"])
    if built != md["data_indices"][ds]:
        msg = "index_collection does not reproduce the real data_indices; fix the helper"
        raise SystemExit(msg)


def metadata_inference_block(ic: dict, types: dict, timesteps: dict, extra: dict | None = None) -> dict:
    """Per-dataset block as written by datamodule and task."""
    block = dict(extra or {})
    block["data_indices"] = {
        "input": ic["model"]["input"]["name_to_index"],
        "output": ic["model"]["output"]["name_to_index"],
    }
    block["variable_types"] = types
    block["timesteps"] = timesteps
    return block


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def build_f0(sources: Path) -> None:
    simple = yaml.safe_load((sources / "inf-main/tests/unit/checkpoints/simple.yaml").read_text())
    # F0: no version, no metadata_inference. Real shape/grid lookups need the legacy table.
    f0 = copy.deepcopy(simple)
    f0["dataset"].pop("shape")
    # No real checkpoint has dataset.data_request; they all keep it deep in the dataset chain. What the oldest
    # ai-models checkpoints held is unknown, so use the shape of the real ones, cut down to the grid.
    data_request = f0["dataset"].pop("data_request")
    f0["dataset"]["specific"] = {"action": "zarr", "attrs": {"data_request": data_request}}
    write("f0_ai_models.json", f0)
    f0b = copy.deepcopy(f0)
    f0b["dataset"].pop("variables_metadata")
    write("f0_no_variables_metadata.json", f0b)


def f1_envelope(src: dict, seed: int, timestamp: str) -> dict:
    """Add the keys the F1 trainer wrote (f537d7f20 removed it), in the order of that writer."""
    return {
        "version": "1.0",
        "config": src["config"],
        "seed": seed,
        "run_id": F1_RUN_ID,
        "dataset": src["dataset"],
        "data_indices": src["data_indices"],
        "provenance_training": provenance_stub(None, timestamp.replace("+00:00", ""), "3.11.9"),
        "timestamp": timestamp,
    }


def build_f1(sources: Path, checkpoints: dict[str, Path]) -> None:
    for name, key in F1_FROM_CHECKPOINT.items():
        if key in checkpoints:
            write(name, trim_common(load_checkpoint_metadata(checkpoints[key]), flat_dataset=True))
        else:
            print(f"SKIPPED {name}: checkpoint {CHECKPOINT_FILES[key]} not given")
    base = sources / "inf-main/tests/unit/checkpoints"
    write(
        "f1_null_diagnostic.json",
        f1_envelope(load(base / "ocean.json"), 5678, "2025-06-02T09:00:00.000000+00:00"),
    )


def build_f2_f3(sources: Path) -> dict:
    unit = sources / "inf-main/tests/unit/checkpoints"
    integ = sources / "inf-main/tests/integration"

    # Real dev-branch output (2025-12-17) whose output_relative_date_indices is a bare int, not a list.
    write("f2_dev_scalar_index.json", trim_common(load(unit / "multi-single.json")))

    f3 = trim_common(load(integ / "global-multi-steps-2in-2out/metadata.json"))
    write("f3_single_dataset.json", f3)
    write("f3_multi_dataset.json", trim_common(load(integ / "era5-cerra/metadata.json")))
    write("f3_temporal_downscaler.json", trim_common(load(integ / "temporal-downscaler/metadata.json")))

    # F2 (f537d7f20): timesteps only hold relative_date_indices_training and timestep; no shapes, no task.
    f2 = copy.deepcopy(f3)
    mi = f2["metadata_inference"]
    mi.pop("task")
    for name in mi["dataset_names"]:
        mi[name].pop("shapes")
        for key in ("input_relative_date_indices", "output_relative_date_indices"):
            mi[name]["timesteps"].pop(key)
    write("f2_early.json", f2)
    return f3


def build_f4(sources: Path) -> None:
    """F4 (08f653665, 2026-06..10): task at top level, ``metadata_inference.task`` null. Real training run."""
    real_f4 = sources / REAL_F4
    if real_f4.is_file():
        write("f4_task_top_level.json", trim_common(load(real_f4)))
    else:
        print(f"SKIPPED f4_task_top_level.json: {REAL_F4} not found")


def build_f5(sources: Path) -> None:
    f5 = trim_common(load(sources / f"{INF_INTEGRATION}/rmi-lam/metadata.json"))
    check_index_collection(f5)
    write("f5_offsets.json", f5)


def build_f6_f8(sources: Path) -> None:
    real_f6 = sources / REAL_F6
    if not real_f6.is_file():
        print(f"SKIPPED f6_main.json, f6_warm_start.json, f8_richer_batch.json: {REAL_F6} not found")
        return

    # ---- F6: real training main run (see README) ------------------------------------------------
    f6 = trim_common(load(real_f6))
    check_index_collection(f6)
    write("f6_main.json", f6)

    # ---- F6 warm start: as checkpoint/lineage.py records a load (synthetic) ----------------------
    # record_checkpoint_lineage appends the entry and adds parent_uuid; both exist before uuid is assigned at save.
    warm: dict[str, Any] = {}
    for key, value in f6.items():
        if key == "uuid":
            warm["parent_uuid"] = PARENT_UUID
        warm[key] = value
    warm["checkpoint_lineage"] = [
        *f6["checkpoint_lineage"],
        {"loading_strategy": "warm_start", "parent_uuid": PARENT_UUID},
    ]
    write("f6_warm_start.json", warm)

    # ---- F8: richer-batch. One gridded dataset (from F6) and one synthetic tabular dataset -----
    f8 = copy.deepcopy(f6)
    f8["version"] = "3.0"
    f8["timestamp"] = "2026-11-23T00:00:00.000000+00:00"
    f8["provenance_training"]["time"] = "2026-11-23T00:00:00.000000"

    obs_names = ["2t", "2d", "msl", "cos_julian_day", "sin_julian_day", "tp"]
    obs_types = {
        "forcing": ["cos_julian_day", "sin_julian_day"],
        "target": [],
        "prognostic": ["2t", "2d", "msl"],
        "diagnostic": ["tp"],
    }
    obs_ic = index_collection(obs_names, obs_types["prognostic"], obs_types["forcing"], obs_types["diagnostic"])

    mi = f8["metadata_inference"]
    task = mi.pop("task")
    gridded = mi["data"]
    grid_size = f8["dataset"]["data"]["shape"][-1]
    # 3.0 no longer carries shapes (the model's fill_metadata is empty on the richer-batch branch)
    gridded.pop("shapes")
    mi["data"] = {"data_type": "gridded", "grid_size": grid_size, **gridded}
    mi["obs"] = metadata_inference_block(
        obs_ic,
        obs_types,
        copy.deepcopy(gridded["timesteps"]),
        {"data_type": "tabular", "grid_size": None},
    )
    mi["dataset_names"] = ["data", "obs"]
    mi["task"] = task

    f8["dataset"]["obs"] = {}
    f8["data_indices"]["obs"] = obs_ic
    f8["config"]["data"]["datasets"]["obs"] = {
        "forcing": obs_types["forcing"],
        "diagnostic": obs_types["diagnostic"],
        "target": None,
    }
    for part in ("training", "validation", "test"):
        f8["config"]["dataloader"][part]["datasets"]["obs"] = {
            "dataset_config": {"dataset": "example-observations", "frequency": "6h"},
            "start": f8["config"]["dataloader"][part]["datasets"]["data"].get("start"),
            "end": f8["config"]["dataloader"][part]["datasets"]["data"].get("end"),
        }
    f8["uuid"] = "6f1d1a64-3c0b-4f7e-9d63-2a7c9a1f4e10"
    write("f8_richer_batch.json", f8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="?", type=Path, help="directory containing inf-main/ (same as --sources)")
    parser.add_argument("--sources", dest="sources_opt", type=Path, help="directory containing inf-main/")
    parser.add_argument("--checkpoint-dir", type=Path, help=f"directory holding {', '.join(CHECKPOINT_FILES.values())}")
    for key, file_name in CHECKPOINT_FILES.items():
        parser.add_argument(f"--{key}", type=Path, help=f"path of {file_name} (overrides --checkpoint-dir)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare existing fixtures with their sources; write nothing",
    )
    args = parser.parse_args()

    sources = args.sources_opt or args.sources
    checkpoints: dict[str, Path] = {}
    for key, file_name in CHECKPOINT_FILES.items():
        explicit = getattr(args, key)
        default = args.checkpoint_dir / file_name if args.checkpoint_dir else None
        path = explicit or default
        if path is not None and path.is_file():
            checkpoints[key] = path
        elif explicit is not None:
            print(f"WARNING: checkpoint {explicit} not found, skipped")

    if args.check:
        failures = run_check(sources, checkpoints)
        print(f"{failures} fixture(s) failed the structure check" if failures else "structure check passed")
        return 1 if failures else 0

    if sources is None:
        parser.error("a sources directory is needed")
    build_f0(sources)
    build_f1(sources, checkpoints)
    build_f2_f3(sources)
    build_f4(sources)
    build_f5(sources)
    build_f6_f8(sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
