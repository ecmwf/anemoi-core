# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
    resolve: bool = True,
) -> DictConfig:
    """Compose an Anemoi config from a YAML file using the repo's packaged defaults.

    The file may be given as an absolute or relative path (relative paths are resolved
    against the current working directory) and may contain a Hydra ``defaults:`` list.
    Group entries in that list (e.g. ``model: gnn``, ``data: zarr``, ``training: default``)
    are resolved against the packaged configs at ``pkg://anemoi.training/config`` via the
    auto-discovered ``AnemoiSearchPathPlugin``.

    Parameters
    ----------
    path : str | Path
        Path to the primary config YAML file.
    overrides : list[str] | None, optional
        Hydra-style command-line overrides (e.g. ``["model=gnn", "training.max_epochs=1"]``),
        by default None.
    resolve : bool, optional
        Whether to resolve OmegaConf interpolations (including ``${oc.env:...}``) in place
        before returning, by default True.

    Returns
    -------
    DictConfig
        The composed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    # initialize_config_dir requires an absolute directory; clear any prior global state
    # so the function is safe to call repeatedly or after another Hydra initialization.
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        cfg = compose(config_name=path.stem, overrides=overrides or [])

    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


def load_config2(base_config: Path, input_config: Path, *, resolve: bool = True) -> DictConfig:
    """Load a config file, resolving ``defaults:`` entries from *base_config* using only OmegaConf.

    Replicates the Hydra defaults-list mechanism without requiring Hydra at runtime.
    Each ``group: value`` entry in ``defaults:`` is resolved as
    ``{base_config}/{group}/{value}.yaml`` (or relative to the current file's
    directory for nested configs) and the loaded content is nested under ``group``.
    The special ``_self_`` entry controls where the primary file's own keys appear
    in the merge order; it defaults to the end (primary keys win) when absent.
    A ``null`` value for any group disables that default.

    Parameters
    ----------
    base_config : Path
        Directory containing the default config groups (the "config store").
    input_config : Path
        Primary YAML file to load.  May contain a ``defaults:`` list.
    resolve : bool, optional
        Whether to resolve OmegaConf interpolations before returning, by default True.

    Returns
    -------
    DictConfig
        The fully merged configuration, with the ``defaults`` key stripped.

    Raises
    ------
    FileNotFoundError
        If *input_config* or any referenced default file does not exist.
    """
    base_config = Path(base_config).expanduser().resolve()
    input_config = Path(input_config).expanduser().resolve()

    if not base_config.is_dir():
        msg = f"Base config directory not found: {base_config}"
        raise FileNotFoundError(msg)
    if not input_config.is_file():
        msg = f"Config file not found: {input_config}"
        raise FileNotFoundError(msg)

    cfg = _merge_config(base_config, input_config, current_dir=base_config)
    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


def _merge_config(base_config: Path, config_file: Path, current_dir: Path) -> DictConfig | ListConfig:
    """Recursively load *config_file* and merge its ``defaults:`` entries."""
    raw: DictConfig | ListConfig = OmegaConf.load(config_file)

    # Non-dict configs (e.g. a plain list like `[]`) have no defaults to process.
    if not isinstance(raw, DictConfig):
        return raw

    # Extract and remove defaults list before merging so it doesn't appear in output.
    defaults_node = raw.pop("defaults", OmegaConf.create([]))
    defaults: list[Any] = OmegaConf.to_container(defaults_node, resolve=False)  # type: ignore[assignment]

    before_self: list[DictConfig | ListConfig] = []
    after_self: list[DictConfig | ListConfig] = []
    self_seen = False

    for entry in defaults:
        if entry == "_self_":
            self_seen = True
            continue
        loaded = _load_default_entry(base_config, current_dir, entry)
        if loaded is not None:
            if self_seen:
                after_self.append(loaded)
            else:
                before_self.append(loaded)

    # Build ordered merge list.  Without _self_, primary keys come last (win).
    parts: list[DictConfig | ListConfig] = (
        [*before_self, raw, *after_self] if self_seen else [*before_self, *after_self, raw]
    )

    if not parts:
        return OmegaConf.create({})
    result = parts[0]
    for part in parts[1:]:
        result = OmegaConf.merge(result, part)
    return result


def _load_default_entry(base_config: Path, current_dir: Path, entry: Any) -> DictConfig | ListConfig | None:
    """Resolve and load a single entry from a ``defaults:`` list.

    Parameters
    ----------
    base_config : Path
        Root of the config store — used as the search root when a path relative to
        *current_dir* does not exist.
    current_dir : Path
        Directory of the config file that owns this defaults list.
    entry : Any
        A single defaults-list item: either ``_self_`` (str), a ``{group: value}`` dict,
        or a plain string path.

    Returns
    -------
    DictConfig | ListConfig | None
        The loaded (and recursively merged) config, wrapped under the group key when
        the entry is a ``{group: value}`` mapping, or ``None`` when disabled (``null``).
    """
    if isinstance(entry, dict):
        if len(entry) != 1:
            msg = f"Unsupported defaults entry (expected exactly one key): {entry}"
            raise ValueError(msg)
        group, value = next(iter(entry.items()))
        if value is None:
            return None

        # Resolve the config file: try current_dir first, fall back to base_config.
        candidate = current_dir / group / f"{value}.yaml"
        if not candidate.is_file():
            candidate = base_config / group / f"{value}.yaml"
        if not candidate.is_file():
            msg = f"Default config not found for '{group}: {value}' (tried {candidate})"
            raise FileNotFoundError(msg)

        sub_cfg = _merge_config(base_config, candidate, candidate.parent)
        return OmegaConf.create({group: OmegaConf.to_container(sub_cfg, resolve=False)})

    if isinstance(entry, str):
        # Plain path reference (no nesting).
        candidate = current_dir / f"{entry}.yaml"
        if not candidate.is_file():
            candidate = base_config / f"{entry}.yaml"
        if not candidate.is_file():
            msg = f"Default config not found for '{entry}' (tried {candidate})"
            raise FileNotFoundError(msg)
        return _merge_config(base_config, candidate, candidate.parent)

    msg = f"Unsupported defaults entry type {type(entry)}: {entry}"
    raise TypeError(msg)
