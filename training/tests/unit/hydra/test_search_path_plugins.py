# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import pytest
from hydra import initialize

# ConfigSearchPath is abstract; ConfigSearchPathImpl is its only concrete
# implementation and is stable across the Hydra 1.3.x floor we pin.
from hydra._internal.config_search_path_impl import ConfigSearchPathImpl
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra_plugins.anemoi_searchpath.anemoi_searchpath_plugin import AnemoiSearchPathPlugin


def test_anemoi_home_searchpath_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert AnemoiSearchPathPlugin.__name__ in [x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)]


def test_config_installed() -> None:
    with initialize(version_base=None):
        config_loader = GlobalHydra.instance().config_loader()
        assert "default" in config_loader.get_group_options("hydra/output")


def test_config_path_wins_and_home_env_paths_removed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # CWD that exists and has no nested 'config' subdir (so the cwd suffix is added)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    # A populated ANEMOI_CONFIG_PATH and a populated anemoi-home dir: under the OLD
    # behavior these would be prepended; under the new behavior they must be ignored.
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    monkeypatch.setenv("ANEMOI_CONFIG_PATH", str(env_dir))

    fake_home = tmp_path / "home"
    (fake_home / ".config" / "anemoi" / "training").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    search_path = ConfigSearchPathImpl()
    # Mimic the primary path that @hydra.main / --config-path installs.
    search_path.append(provider="main", path="pkg://anemoi.training/commands")

    AnemoiSearchPathPlugin().manipulate_search_path(search_path)

    providers = [entry.provider for entry in search_path.get_path()]

    # Home and env search paths are gone entirely.
    assert "anemoi-home-searchpath-plugin" not in providers
    assert "anemoi-env-searchpath-plugin" not in providers
    # CWD is appended AFTER the primary path, so --config-path keeps top priority.
    assert "anemoi-cwd-searchpath-plugin" in providers, "CWD was not appended to the search path"
    assert providers.count("anemoi-cwd-searchpath-plugin") == 1
    assert providers.index("main") < providers.index("anemoi-cwd-searchpath-plugin")
    # Packaged configs remain the lowest-priority fallback.
    assert providers[-1] == "anemoi-package-searchpath-plugin"


def _make_search_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, config_path: str | None = None) -> list[str]:
    """Helper: build a search path as the real code does and return the provider list.

    Parameters
    ----------
    config_path:
        Simulates ``--config-path``.  When *None* the entry point uses
        ``config_path=None`` (our fix) so no primary path is pre-populated.
    """
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)

    search_path = ConfigSearchPathImpl()
    if config_path is not None:
        # Mimic what Hydra does when the user passes --config-path.
        search_path.append(provider="main", path=config_path)

    AnemoiSearchPathPlugin().manipulate_search_path(search_path)
    return [entry.provider for entry in search_path.get_path()]


def test_cwd_beats_package_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Priority 2 > 3: CWD must come before the packaged defaults when no --config-path is given.

    With ``config_path=None`` in ``@hydra.main`` there is no "main" entry in the
    search path before the plugin runs, so the plugin is the sole authority on
    ordering.
    """
    providers = _make_search_path(monkeypatch, tmp_path, config_path=None)

    assert "anemoi-cwd-searchpath-plugin" in providers, "CWD was not added to the search path"
    assert "anemoi-package-searchpath-plugin" in providers, "Package fallback was not added"
    assert providers.index("anemoi-cwd-searchpath-plugin") < providers.index(
        "anemoi-package-searchpath-plugin"
    ), "CWD must come before package defaults (priority 2 > 3)"


def test_config_path_beats_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Priority 1 > 2: an explicit --config-path must come before CWD."""
    user_config = tmp_path / "my_configs"
    user_config.mkdir()

    providers = _make_search_path(monkeypatch, tmp_path, config_path=str(user_config))

    assert "main" in providers, "--config-path (main) was not in the search path"
    assert "anemoi-cwd-searchpath-plugin" in providers, "CWD was not added to the search path"
    assert providers.index("main") < providers.index(
        "anemoi-cwd-searchpath-plugin"
    ), "--config-path must come before CWD (priority 1 > 2)"


def test_config_path_beats_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Priority 1 > 3: an explicit --config-path must come before the packaged defaults."""
    user_config = tmp_path / "my_configs"
    user_config.mkdir()

    providers = _make_search_path(monkeypatch, tmp_path, config_path=str(user_config))

    assert "main" in providers, "--config-path (main) was not in the search path"
    assert "anemoi-package-searchpath-plugin" in providers, "Package fallback was not added"
    assert providers.index("main") < providers.index(
        "anemoi-package-searchpath-plugin"
    ), "--config-path must come before package defaults (priority 1 > 3)"


def test_full_priority_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End-to-end: config-path (1) > CWD (2) > package defaults (3)."""
    user_config = tmp_path / "my_configs"
    user_config.mkdir()

    providers = _make_search_path(monkeypatch, tmp_path, config_path=str(user_config))

    assert "main" in providers
    assert "anemoi-cwd-searchpath-plugin" in providers
    assert "anemoi-package-searchpath-plugin" in providers

    pos_config_path = providers.index("main")
    pos_cwd = providers.index("anemoi-cwd-searchpath-plugin")
    pos_package = providers.index("anemoi-package-searchpath-plugin")

    assert pos_config_path < pos_cwd < pos_package, (
        f"Expected config-path ({pos_config_path}) < CWD ({pos_cwd}) < package ({pos_package})"
    )
