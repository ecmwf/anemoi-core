import os
from logging import getLogger
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

LOGGER = getLogger(__name__)


class AnemoiHomeSearchPathPlugin(SearchPathPlugin):
    """Prepend the Anemoi home directory to the hydra searchpath."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        anemoi_home_path = Path(Path.home(), ".config", "anemoi", "training", "config")
        if anemoi_home_path.exists():
            search_path.prepend(
                provider="anemoi-home-searchpath-plugin",
                path=str(anemoi_home_path),
            )
            LOGGER.info(f"Prepending {anemoi_home_path} to the search path.")
            LOGGER.debug(f"Search path is now: {search_path}")


class AnemoiEnvSearchPathPlugin(SearchPathPlugin):
    """Prepend the path env ANEMOI_CONFIG_PATH to hydra searchpath."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        anemoi_config_path = os.getenv("ANEMOI_CONFIG_PATH")
        if anemoi_config_path is not None and anemoi_config_path.exists():
            search_path.prepend(
                provider="anemoi-env-searchpath-plugin",
                path=str(anemoi_config_path),
            )
            LOGGER.info(f"Appending {anemoi_config_path} to the search path.")
            LOGGER.debug(f"Search path is now: {search_path}")


class UserCWDSearchPathPlugin(SearchPathPlugin):
    """Prepend the current working directory to the hydra search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        cwd_path = Path(Path.cwd(), "config")
        if cwd_path.exists():
            search_path.prepend(
                provider="anemoi-env-searchpath-plugin",
                path=str(cwd_path),
            )
            LOGGER.info(f"Appending {cwd_path} to the search path.")
            LOGGER.debug(f"Search path is now: {search_path}")
