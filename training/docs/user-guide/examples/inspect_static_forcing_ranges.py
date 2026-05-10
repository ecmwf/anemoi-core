#!/usr/bin/env python
"""Inspect raw and normalized ranges for static/forcing variables.

This script composes an Anemoi training config, opens the configured dataset,
builds the same IndexCollection/InputNormalizer objects used by training, and
reports:

1. dataset statistics stored in the Zarr metadata
2. actual raw sample ranges over a selected time window
3. normalized sample ranges after the configured InputNormalizer

Typical usage:

  python training/docs/user-guide/examples/inspect_static_forcing_ranges.py ^
    --config-path training/docs/user-guide/examples ^
    --config-name anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-input-graphtransformer-base_1-finer-graph-v1-single-input-day20240505-pl925500-reduced-vars

You can override the sampled date window and variable list:

  python training/docs/user-guide/examples/inspect_static_forcing_ranges.py ^
    --config-path training/docs/user-guide/examples ^
    --config-name <config-name> ^
    --start 2024-05-05T00:00:00 --end 2024-05-05T23:00:00 ^
    --variables landcover orography swdown
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.training.data.dataset import NativeGridDataset


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _print_summary(title: str, stats: dict[str, float]) -> None:
    print(
        f"{title}: "
        f"min={_format_float(stats['min'])} "
        f"max={_format_float(stats['max'])} "
        f"mean={_format_float(stats['mean'])} "
        f"std={_format_float(stats['std'])}"
    )


def _compose_config(config_dir: Path, config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name without .yaml")
    parser.add_argument("--dataset-name", default="data", help="Dataset key to inspect")
    parser.add_argument("--start", default=None, help="Override start date")
    parser.add_argument("--end", default=None, help="Override end date")
    parser.add_argument("--frequency", default=None, help="Override data frequency")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["landcover", "orography", "swdown"],
        help="Variables to inspect",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_path)
    cfg = _compose_config(config_dir, args.config_name)

    dataset_name = args.dataset_name
    data_cfg = cfg.data.datasets[dataset_name]
    train_ds_cfg = cfg.dataloader.training.datasets[dataset_name]

    dataset_path = cfg.system.input.dataset
    start = args.start or train_ds_cfg.start
    end = args.end or train_ds_cfg.end
    frequency = args.frequency or train_ds_cfg.frequency
    drop = OmegaConf.to_container(train_ds_cfg.drop, resolve=True) if train_ds_cfg.get("drop") is not None else None

    print(f"CONFIG_NAME: {args.config_name}")
    print(f"DATASET_PATH: {dataset_path}")
    print(f"WINDOW: start={start} end={end} frequency={frequency}")
    print(f"DROP: {drop}")
    print(f"FORCING: {OmegaConf.to_container(data_cfg.forcing, resolve=True)}")
    print(f"DIAGNOSTIC: {OmegaConf.to_container(data_cfg.diagnostic, resolve=True)}")

    reader = NativeGridDataset(dataset=dataset_path, start=start, end=end, frequency=frequency, drop=drop)
    name_to_index = reader.name_to_index
    variables = [var for var in args.variables if var in name_to_index]
    missing = [var for var in args.variables if var not in name_to_index]
    if missing:
        print(f"MISSING_VARIABLES: {missing}")
    if not variables:
        raise SystemExit("None of the requested variables exist in the loaded dataset.")

    data_indices = IndexCollection(data_config=data_cfg, name_to_index=name_to_index)
    normalizer_cfg = data_cfg.processors.normalizer.config
    normalizer = InputNormalizer(config=normalizer_cfg, data_indices=data_indices, statistics=reader.statistics)

    sample = reader.get_sample(slice(None), None).float()
    normalized = normalizer.transform(sample.clone(), in_place=True)

    print(f"SAMPLE_SHAPE: {tuple(sample.shape)}  # [time, ensemble, grid, variables]")
    print(f"DATASET_VARIABLE_COUNT: {len(reader.variables)}")
    print(f"MODEL_INPUT_VARIABLE_COUNT: {len(data_indices.model.input)}")

    for var in variables:
        full_idx = name_to_index[var]
        input_idx = data_indices.data.input.name_to_index[var]
        method = normalizer.methods.get(var, normalizer.default)

        raw_values = sample[..., input_idx].cpu().numpy().reshape(-1)
        norm_values = normalized[..., input_idx].cpu().numpy().reshape(-1)
        stats = reader.statistics

        print("")
        print(f"[{var}]")
        print(f"  normalization_method: {method}")
        print(
            "  dataset_statistics: "
            f"minimum={_format_float(float(stats['minimum'][full_idx]))} "
            f"maximum={_format_float(float(stats['maximum'][full_idx]))} "
            f"mean={_format_float(float(stats['mean'][full_idx]))} "
            f"stdev={_format_float(float(stats['stdev'][full_idx]))}"
        )
        _print_summary("  raw_window_values", _summarize(raw_values))
        _print_summary("  normalized_window_values", _summarize(norm_values))

        unique_count = int(np.unique(raw_values).size)
        print(f"  raw_unique_count: {unique_count}")
        if unique_count <= 10:
            uniques = ", ".join(_format_float(float(v)) for v in np.unique(raw_values))
            print(f"  raw_unique_values: [{uniques}]")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
