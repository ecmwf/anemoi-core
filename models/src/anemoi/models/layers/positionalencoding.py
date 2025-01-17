from abc import ABC
from abc import abstractmethod

import torch
from torch import Tensor


class BasePositionalEncoding(ABC):
    """Configurable method calcuating positional encodings for latlons of a grid.

    To enable the positional encoding add the following to the model-config file and
    chose the corresponding positional-encoding-class:
    ```
    positional_encoding:
        _target_: anemoi.models.layers.positionalencoding.CosSinLatCosSinLon
        _convert_: all
    ```
    If the entry positional_encoding does not exist or is None, no positional encoding is used.

    """

    def __init__(self) -> None:
        """Initialise Function for calculating the positional encodings."""

    @abstractmethod
    def positional_encoding(self, latlons_hidden: Tensor) -> Tensor: ...


class LatCosSinLon(BasePositionalEncoding):
    """Lat, cos(lon), sin(lon) for grid points."""

    def positional_encoding(self, latlons_hidden: Tensor) -> Tensor:
        """Output lat, cos(lon), sin(lon) for grid points."""
        lat_coslon_sinlon_hidden = torch.cat(
            (
                latlons_hidden[:, 0].unsqueeze(-1),
                torch.cos(latlons_hidden[:, 1].unsqueeze(-1)),
                torch.sin(latlons_hidden[:, 1].unsqueeze(-1)),
            ),
            dim=-1,
        )
        return lat_coslon_sinlon_hidden


class CosSinLatCosSinLon(BasePositionalEncoding):
    """Cos(lat), sin(lat), cos(lon), sin(lon) for grid points."""

    def positional_encoding(self, latlons_hidden: Tensor) -> Tensor:
        """Output cos(lat), sin(lat), cos(lon), sin(lon) for grid points."""
        coslat_sinlat_coslon_sinlon_hidden = torch.cat(
            (
                torch.cos(latlons_hidden[:, 0].unsqueeze(-1)),
                torch.sin(latlons_hidden[:, 0].unsqueeze(-1)),
                torch.cos(latlons_hidden[:, 1].unsqueeze(-1)),
                torch.sin(latlons_hidden[:, 1].unsqueeze(-1)),
            ),
            dim=-1,
        )
        return coslat_sinlat_coslon_sinlon_hidden
