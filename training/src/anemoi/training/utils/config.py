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

from hydra import compose
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
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
