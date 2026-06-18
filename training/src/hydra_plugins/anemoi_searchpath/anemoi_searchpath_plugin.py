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
    """Configure the Hydra search path for anemoi-training.

    Search paths, in decreasing priority:
    1. the path passed via ``--config-path`` (Hydra's primary ``@hydra.main`` path),
    2. the current working directory,
    3. the packaged default configs (``pkg://anemoi.training/config``).

    CWD and the packaged configs are both appended (not prepended), so any path
    already in the search path — including the ``--config-path`` installed by
    ``@hydra.main`` — keeps higher priority.
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Append the current working directory and the packaged configs.

        Parameters
        ----------
        search_path : ConfigSearchPath
            Hydra ConfigSearchPath object.

        """
        if os.environ.get("ANEMOI_CONFIG_PATH"):
            LOGGER.warning(
                "ANEMOI_CONFIG_PATH is set but is no longer read by anemoi-training. "
                "Use --config-path to specify a config directory.",
            )

        for suffix in ("", "config"):
            cwd_path = Path.cwd() / suffix
            if cwd_path.exists() and not Path(cwd_path, "config").exists():
                search_path.append(
                    provider="anemoi-cwd-searchpath-plugin",
                    path=str(cwd_path),
                )
                LOGGER.info("Appending current working directory (%s) to the search path.", cwd_path)
                LOGGER.debug("Search path is now: %s", search_path)

        # Lowest-priority fallback: the default configs shipped inside the package
        # (issue #570 / #656). This is how config.yaml's `defaults:` groups are found.
        search_path.append(
            provider="anemoi-package-searchpath-plugin",
            path="pkg://anemoi.training/config",
        )
        LOGGER.debug("Appended package config path (pkg://anemoi.training/config) to search path.")
        LOGGER.info("Search path is now: %s", search_path)
