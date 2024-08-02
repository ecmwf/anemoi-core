from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from hydra_plugins.anemoi_searchpath.anemoi_searchpath_plugin import AnemoiEnvSearchPathPlugin
from hydra_plugins.anemoi_searchpath.anemoi_searchpath_plugin import AnemoiHomeSearchPathPlugin
from hydra_plugins.anemoi_searchpath.anemoi_searchpath_plugin import UserCWDSearchPathPlugin


def test_anemoi_home_searchpath_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert AnemoiHomeSearchPathPlugin.__name__ in [x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)]


def test_anemoi_env_searchpath_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert AnemoiEnvSearchPathPlugin.__name__ in [x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)]


def test_anemoi_cwd_searchpath_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking at all Plugins
    assert UserCWDSearchPathPlugin.__name__ in [x.__name__ for x in Plugins.instance().discover(SearchPathPlugin)]


def test_config_installed() -> None:
    with initialize(version_base=None):
        config_loader = GlobalHydra.instance().config_loader()
        assert "default" in config_loader.get_group_options("hydra/output")
