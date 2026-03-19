# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pytorch_lightning.utilities.rank_zero import rank_zero_info

import numpy as np
import logging

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.mse import MSELoss

LOGGER = logging.getLogger(__name__)


class WeightedMSELoss(MSELoss):
    """Weighted MSE loss for use with diffusion models.

    This loss applies weights to the MSE difference
    """

    name: str = "weighted_mse"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
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
        weights : torch.Tensor | None, optional
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
        out = self.calculate_difference(pred, target)
        # self.plot_step(target, [0,1,2,3], ["u", "v", "t","tp"], 0, 0)
        if weights is not None:
            out = out * weights
        out = self.scale(out, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        # print("out shape dans loss : ", out.shape, out)
        return self.reduce(out, squash, group=group if is_sharded else None)


    def plot_step(self, y_denoise, idx_var, vars, denoising_step, sigma) -> None:
        """Write a step of the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        import earthkit.data as ekd
        import earthkit.plots as ekp

        print("plotting step ...")
        
        latitudes = np.load("/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/latitudes.npy")
        longitudes = np.load("/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/longitudes.npy")
        
        plotting_fields = []

        for i in range(len(vars)):
            idx = idx_var[i]
            variable = vars[i]
            plotting_fields.append(
                ekd.ArrayField(
                    y_denoise[0,0,:,idx].detach().cpu().numpy(),
                    {
                        "param": variable,
                        "shortName": variable,
                        "variable_name": variable,
                        "latitudes": latitudes,
                        "longitudes": longitudes,
                    },
                )
            )
        fig = ekp.quickplot(
            ekd.FieldList.from_fields(plotting_fields), mode="subplots", domain=None
        )

        title = f"sigma = {sigma: .2f}"

        fig.title(title)
        fname = f'/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/plot_step_during_inf/x=0_1_image_sdedit/target_{denoising_step}_sigma={sigma: .2f}.png'
        # mpl_fig = getattr(fig, "figure", None)
        
        # mpl_fig = getattr(fig, "figure", None)
        # if mpl_fig is None:
        #     mpl_fig = getattr(fig, "_fig", None)

        # if mpl_fig is not None:
        #     mpl_fig.suptitle(title, fontsize=14)
            
        fig.save(fname)
        del fig