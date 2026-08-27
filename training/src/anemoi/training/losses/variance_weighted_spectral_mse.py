import logging

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.mse import MSELoss
from anemoi.utils.spectral import DCT2D

LOGGER = logging.getLogger(__name__)


class VarianceWeightedSpectralMSELoss(MSELoss):
    """Variance-weighted spectral MSE loss for use with diffusion models.
    This loss applies weights to the spectral MSE difference
    """

    name: str = "variance_weighted_spectral_mse"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        transform: DCT2D,
        variance: torch.Tensor,
        noise_weights: torch.Tensor | None = None,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Calculates the weighted MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        transform : anemoi.training.utils.spectral.DCT2D
            Instance of DCT2D used  TODO: generalize with mother class SpectralTransform class from models.layers
        variance : torch.Tensor
            Variance tensor, shape (lat*lon, n_outputs)
        noise_weights : torch.Tensor | None, optional
            Weights to apply to the MSE difference, by default None
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group: ProcessGroup, optional
            Distributed group to reduce over, by default None

        Returns
        -------
        torch.Tensor
            Weighted MSE loss
        """
        is_sharded = grid_shard_slice is not None
        out = self.calculate_difference(transform.forward(pred), transform.forward(target))

        out = out/variance
        
        if noise_weights is not None:
            out = out * noise_weights
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        sortie = self.reduce(out, squash, group=group if is_sharded else None)
        return sortie