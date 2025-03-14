# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from hydra import initialize
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
