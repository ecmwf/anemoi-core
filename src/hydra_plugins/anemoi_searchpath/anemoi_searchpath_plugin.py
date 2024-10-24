# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

LOGGER = logging.getLogger(__name__)


class AnemoiSearchPathPlugin(SearchPathPlugin):
    """Prepend the Anemoi home directory to the hydra searchpath."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Prepend the Anemoi directories to the hydra searchpath.

        This builds the hierarchy of the search path by prepending the Anemoi
        directories to the search path. The hierarchy is as follows in decreasing priority:
        - Current working directory
        - Anemoi Config Env directory
        - Anemoi Home directory

        Parameters
        ----------
        search_path : ConfigSearchPath
            Hydra ConfigSearchPath object.

        """
        for suffix in ("", "config"):
            anemoi_home_path = Path(Path.home(), ".config", "anemoi", "training", suffix)
            if anemoi_home_path.exists() and not Path(anemoi_home_path, "config").exists():
                search_path.prepend(
                    provider="anemoi-home-searchpath-plugin",
                    path=str(anemoi_home_path),
                )
                LOGGER.info("Prepending Anemoi Home (%s) to the search path.", anemoi_home_path)
                LOGGER.debug("Search path is now: %s", search_path)

        for suffix in ("", "config"):
            env_anemoi_config_path = os.getenv("ANEMOI_CONFIG_PATH")
            if env_anemoi_config_path is None:
                continue
            anemoi_config_path = Path(env_anemoi_config_path, suffix)
            if anemoi_config_path.exists() and not Path(anemoi_config_path, "config").exists():
                search_path.prepend(
                    provider="anemoi-env-searchpath-plugin",
                    path=str(anemoi_config_path),
                )
                LOGGER.info("Prepending Anemoi Config Env (%s) to the search path.", anemoi_config_path)
                LOGGER.debug("Search path is now: %s", search_path)

        for suffix in ("", "config"):
            cwd_path = Path.cwd() / suffix
            if cwd_path.exists() and not Path(cwd_path, "config").exists():
                search_path.prepend(
                    provider="anemoi-cwd-searchpath-plugin",
                    path=str(cwd_path),
                )
                LOGGER.info("Prepending current user directory (%s) to the search path.", cwd_path)
                LOGGER.debug("Search path is now: %s", search_path)
        LOGGER.info("Search path is now: %s", search_path)
