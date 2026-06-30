# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import ast
import importlib.util
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


def _get_hydra_main_config_path(source_file: Path) -> str | None:
    """Parse *source_file* and return the ``config_path`` value passed to ``@hydra.main``, or None if absent."""
    tree = ast.parse(source_file.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            # Match `hydra.main(...)` or plain `main(...)`
            if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, (ast.Attribute, ast.Name))):
                continue
            func = decorator.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id
            if name != "main":
                continue
            for kw in decorator.keywords:
                if kw.arg == "config_path":
                    return None if isinstance(kw.value, ast.Constant) and kw.value.value is None else ast.unparse(kw.value)
    return "NOT_FOUND"


_ENTRY_POINTS = ["train/train.py", "train/evaluate.py", "train/profiler.py"]


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_entry_points_use_config_path_none(entry_point: str) -> None:
    """Regression test: every @hydra.main entry point must use config_path=None.

    With config_path="../config" the package is registered as provider=main
    *before* AnemoiSearchPathPlugin runs, giving the package higher priority
    than the CWD.  Setting config_path=None removes that pre-population so the
    plugin is the sole authority on search-path ordering.
    """
    src_root = Path(importlib.util.find_spec("anemoi.training").origin).parent
    source_file = src_root / entry_point
    assert source_file.exists(), f"Entry point not found: {source_file}"

    config_path_value = _get_hydra_main_config_path(source_file)
    assert config_path_value is None, (
        f"{entry_point}: @hydra.main must use config_path=None, "
        f"but found config_path={config_path_value!r}. "
        "A non-None config_path pre-populates the package as provider=main before "
        "the search-path plugin runs, causing the package to take precedence over "
        "the user's CWD overrides."
    )


def test_cwd_beats_package_without_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Priority 2 > 3: CWD must appear before packaged defaults when no --config-path is given.

    Simulates the real startup: no primary path pre-populated (config_path=None
    in @hydra.main), then the plugin runs.
    """
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    search_path = ConfigSearchPathImpl()
    # No provider=main entry — this is what config_path=None produces.
    AnemoiSearchPathPlugin().manipulate_search_path(search_path)

    providers = [entry.provider for entry in search_path.get_path()]

    assert "anemoi-cwd-searchpath-plugin" in providers, "CWD not added to search path"
    assert "anemoi-package-searchpath-plugin" in providers, "Package fallback not added"
    assert providers.index("anemoi-cwd-searchpath-plugin") < providers.index("anemoi-package-searchpath-plugin"), (
        "CWD must come before package defaults (priority 2 > 3)"
    )


def test_package_as_main_beats_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Documents the bug: when config_path='../config' is used, the package is registered
    as provider=main *before* the plugin runs, so it beats the CWD.

    This test asserts the broken behaviour to make clear why config_path=None is required.
    If this test ever starts *failing* (i.e. the plugin somehow fixes it internally),
    the config_path=None change in the entry points may no longer be necessary.
    """
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    search_path = ConfigSearchPathImpl()
    # Simulate @hydra.main(config_path="../config"): package is pre-populated as main.
    search_path.append(provider="main", path="pkg://anemoi.training/config")
    AnemoiSearchPathPlugin().manipulate_search_path(search_path)

    providers = [entry.provider for entry in search_path.get_path()]

    # The package (as "main") comes before CWD — this is the bug.
    assert providers.index("main") < providers.index("anemoi-cwd-searchpath-plugin"), (
        "Expected the bug: package registered as main should beat CWD. "
        "If this assertion fails the plugin itself now handles this case."
    )
