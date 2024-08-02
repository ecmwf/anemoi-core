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
                LOGGER.info(f"Prepending Anemoi Home ({anemoi_home_path}) to the search path.")
                LOGGER.debug(f"Search path is now: {search_path}")

        for suffix in ("", "config"):
            env_anemoi_config_path = os.getenv("ANEMOI_CONFIG_PATH")
            if env_anemoi_config_path is None:
                return
            anemoi_config_path = Path(env_anemoi_config_path)
            if anemoi_config_path.exists() and not Path(anemoi_config_path, "config").exists():
                search_path.prepend(
                    provider="anemoi-env-searchpath-plugin",
                    path=str(anemoi_config_path),
                )
                LOGGER.info(f"Prepending Anemoi Config Env ({anemoi_config_path}) to the search path.")
                LOGGER.debug(f"Search path is now: {search_path}")

        for suffix in ("", "config"):
            cwd_path = Path.cwd() / suffix
            if cwd_path.exists() and not Path(cwd_path, "config").exists():
                search_path.prepend(
                    provider="anemoi-cwd-searchpath-plugin",
                    path=str(cwd_path),
                )
                LOGGER.info(f"Prepending current user directory ({cwd_path}) to the search path. ")
                LOGGER.debug(f"Search path is now: {search_path}")
        LOGGER.info(f"Search path is now: {search_path}")
