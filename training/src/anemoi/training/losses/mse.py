# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import numpy as np
import pprint
import os 
rank = int(os.environ.get("RANK","0"))
import matplotlib.pyplot as plt 

import logging

import torch

from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class MSELoss(FunctionalLoss):
    """MSE loss."""

    name: str = "mse"

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            MSE loss
        """
        # print("on passe dans MSE  : ", torch.square(pred - target), "de shape :", torch.square(pred - target).shape)
        # vars = ["10u", "10v", "2t"]
        # idx_vars = [0,1,3]
        # if rank == 0 :
        #     print("plot step :  :  :")
        #     self.plot_step(torch.square(pred-target), idx_var=idx_vars, vars=vars)
        #     latitudes = np.load("/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/latitudes.npy")
        #     longitudes = np.load("/project/home/p200177/DE_371/avritj/experiments_anemoi/inference/longitudes.npy")
        #     plt.figure()
        #     plt.scatter(latitudes, longitudes, torch.square(pred-target)[0,0,:,0].detach().cpu().numpy())
        #     plt.savefig("savingfig")
        #     plt.close()
        return torch.square(pred - target)

    def plot_step(self, y_denoise, idx_var, vars) -> None:
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
        print("y denoise shape :", y_denoise.shape)
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

        fname = f'/home/users/u102751/code/anemoi/anemoi-env/MSE2.png'
        # mpl_fig = getattr(fig, "figure", None)
        
        # mpl_fig = getattr(fig, "figure", None)
        # if mpl_fig is None:
        #     mpl_fig = getattr(fig, "_fig", None)

        # if mpl_fig is not None:
        #     mpl_fig.suptitle(title, fontsize=14)
            
        fig.save(fname)
        del fig