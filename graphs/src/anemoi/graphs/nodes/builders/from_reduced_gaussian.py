# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
import os
import re
import tempfile
from functools import cached_property

import requests
import torch

from anemoi.graphs.nodes.builders.from_file import NPZFileNodes
from anemoi.utils.config import load_config

LOGGER = logging.getLogger(__name__)


class ReducedGaussianGridNodes(NPZFileNodes):
    """Nodes from a reduced gaussian grid.

    A gaussian grid is a latitude/longitude grid. The spacing of the latitudes is not regular. However, the spacing of
    the lines of latitude is symmetrical about the Equator. A grid is usually referred to by its 'number' N/O, which
    is the number of lines of latitude between a Pole and the Equator. The N code refers t othe original ECMWF reduced
    Gaussian grid, whereas the code O refers to the octahedral ECMWF reduced Gaussian grid.

    Attributes
    ----------
    grid : str
        The reduced gaussian grid, of shape {n,N,o,O}XXX with XXX latitude lines between the pole and
        equator.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attrs_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, grid: int, name: str) -> None:
        """Initialize the ReducedGaussianGridNodes builder."""
        assert re.fullmatch(
            r"^[oOnN]\d+$", grid
        ), f"{self.__class__.__name__}.grid must match the format [n|N|o|O]XXX with XXX latitude lines between the pole and equator."
        self.local_dir = tempfile.gettempdir().rstrip("/")
        self.file_name = f"grid-{grid.upper()}.npz"
        super().__init__(self.local_dir + "/" + self.file_name, name, lat_key="latitudes", lon_key="longitudes")

    def is_downloaded(self) -> bool:
        """Checks if the grid file is already downloaded."""
        return os.path.exists(self.npz_file)

    @cached_property
    def download_url(self) -> str:
        config = load_config(defaults={"graphs": {"named": {}}})
        return config["graphs"]["named"]["grids"]

    def download_file(self):
        """Downloads the grid file if it is not already downloaded."""
        url = self.download_url + "/" + self.file_name

        LOGGER.info(f"Downloading {self.file_name} grid from: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(self.npz_file, "wb") as f:
                f.write(response.content)
            LOGGER.info(f"File downloaded and saved to {self.local_dir}/.")
        else:
            raise FileNotFoundError(f"Failed to download file from {url}. HTTP status code: {response.status_code}")

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            Coordinates of the nodes, in radians.
        """
        if not self.is_downloaded():
            print(f"File {self.file_name} not found locally. Downloading...")
            self.download_file()

        return self.get_coordinates()
