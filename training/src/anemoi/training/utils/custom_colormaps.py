# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod

import matplotlib.cm as cm
from matplotlib.colors import Colormap
from matplotlib.colors import ListedColormap


class CustomColormap(ABC):
    """Abstract base class for custom colormaps."""

    def __init__(self, variables: list | None = None) -> None:
        """Initializes the custom colormap.

        Parameters
        ----------
        variables : list, optional
            A list of strings representing the variables for which the colormap is used.
        """
        super().__init__()
        if variables is None:
            variables = []
        self.variables = variables

    @abstractmethod
    def get_cmap(self) -> Colormap:
        """Returns the custom colormap."""
        ...


class MatplotlibColormap(CustomColormap):
    """Class for Matplotlib colormaps."""

    def __init__(self, name: str, variables: list | None = None) -> None:
        """Initializes the custom colormap with a Matplotlib colormap.

        Parameters
        ----------
        name : str
            The name of the Matplotlib colormap.
        variables : list, optional
            A list of strings representing the variables for which the colormap is used.
        """
        super().__init__(variables)
        self.name = name
        self.colormap = cm.get_cmap(self.name)

    def get_cmap(self) -> cm:
        return self.colormap


class MatplotlibColormapClevels(CustomColormap):
    """Class for Matplotlib colormaps with custom levels."""

    def __init__(self, clevels: list, variables: list | None = None) -> None:
        """Initializes the custom colormap with custom levels.

        Parameters
        ----------
        clevels : list
            The custom levels for the colormap.
        variables : list, optional
            A list of strings representing the variables for which the colormap is used.
        """
        super().__init__(variables)
        self.clevels = clevels
        self.colormap = ListedColormap(self.clevels)

    def get_cmap(self) -> ListedColormap:
        return self.colormap


class DistinctipyColormap(CustomColormap):
    def __init__(self, n_colors: int, variables: list | None = None, colorblind_type: str | None = None) -> None:
        """Initializes the custom colormap with distinctipy.

        Parameters
        ----------
        n_colors : int
            The number of colors in the colormap.
        variables : list, optional
            A list of strings representing the variables for which the colormap is used.
        colorblind_type : str, optional
            The type of colorblindness to simulate. If None, the default colorblindness from distinctipy is applied.
        """
        try:
            from distinctipy import distinctipy
        except ImportError:
            error_message = "distinctipy package is not available. Please install it to use DistinctipyColormapN."
            raise ImportError(error_message) from None
        super().__init__(variables)
        self.n_colors = n_colors
        self.colormap = distinctipy.get_colormap(distinctipy.get_colors(n_colors, colorblind_type=colorblind_type))

    def get_cmap(self) -> cm:
        return self.colormap
