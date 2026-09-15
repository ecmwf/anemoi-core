# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import einops
import torch
import torch.nn as nn

from anemoi.models.layers.feature_tokenizer import FeatureTokenizer
from anemoi.models.layers.set_transformer import PMA


class PlainInputTransform(nn.Module):
    """No-op input_transform: passes raw variable values straight through, only reshaping the
    timestep axis into the feature axis and appending node_attributes_data. Zero learnable
    parameters, so configuring no input_transform at all doesn't change a model's behavior.
    """

    def __init__(self, feature_names=None):
        super().__init__()

    def forward(self, x, node_attributes_data, feature_names=None):
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
        return torch.cat((x_vars, node_attributes_data), dim=-1)


class Embedder(nn.Module):
    """Turns a node's raw per-variable values into a single vector via attention pooling over
    the variables as a set, so the resulting encoding doesn't depend on a fixed variable count
    or order.
    """

    def __init__(self, feature_names, dim, d_model, nhead, mean=None, std=None):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(feature_names, dim, mean=mean, std=std)
        self.input_proj = nn.Linear(2 + 3 * dim, d_model)
        self.pma = PMA(d_model, nhead, num_seeds=1)
        self.output_dim = d_model

    def forward(self, x, node_attributes_data, feature_names=None):
        """x: (batch, time, ensemble, grid, vars) -> (batch ensemble grid, time d_model + attrs)

        Tokenizes and pools each (node, timestep) row independently, giving each row a
        frame_idx so it carries which timestep it came from, then concatenates timesteps back
        into the feature axis and appends node_attributes_data (static per-node features, with
        no time axis of their own).
        """
        batch, n_time, ensemble, grid, _n_vars = x.shape
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")
        frame_idx = torch.arange(n_time, device=x.device).repeat_interleave(ensemble * grid).repeat(batch)
        encodings = self.feature_tokenizer(x_vars, frame_idx, feature_names=feature_names)
        x_vars = self.pma(self.input_proj(encodings)).squeeze(1)
        x_vars = einops.rearrange(
            x_vars,
            "(batch time ensemble grid) d -> (batch ensemble grid) (time d)",
            batch=batch,
            time=n_time,
            ensemble=ensemble,
            grid=grid,
        )
        return torch.cat((x_vars, node_attributes_data), dim=-1)
