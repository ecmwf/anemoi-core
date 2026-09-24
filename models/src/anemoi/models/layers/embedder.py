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

from anemoi.models.layers.cls_pooling import ClsSelfAttentionPool
from anemoi.models.layers.feature_tokenizer import FeatureTokenizer
from anemoi.models.layers.feature_tokenizer import Tokenizer
from anemoi.models.layers.feature_tokenizer import sinusoidal_positional_encoding
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


class DeepSetEmbedder(nn.Module):
    """Turns a node's raw per-variable values into a single vector via Deep Sets pooling
    (Zaheer et al. 2017): each variable is tokenized and transformed independently (phi), then
    aggregated by mean - no cross-variable interaction during pooling, unlike attention-based
    pooling. Supports a reduced feature_names subset naturally, since phi doesn't depend on the
    total variable count and mean needs no padding/masking for a smaller set.
    """

    def __init__(self, feature_names, dim, d_model, mean=None, std=None):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(feature_names, dim, mean=mean, std=std)
        self.phi = nn.Sequential(
            nn.Linear(2 + 3 * dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # rho is residual - same pattern as ClsSelfAttentionPool's residual - since the mean
        # alone gives the gradient no direct path back to phi's output.
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.output_dim = d_model

    def forward(self, x, node_attributes_data, feature_names=None):
        """x: (batch, time, ensemble, grid, vars) -> (batch ensemble grid, time d_model + attrs)"""
        batch, n_time, ensemble, grid, _n_vars = x.shape
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")
        # no frame_idx (deliberate, as in HierarchicalEmbedder): held constant so it carries no
        # information, rather than modifying FeatureTokenizer to drop the column outright.
        frame_idx = torch.zeros(x_vars.shape[0], device=x.device)
        encodings = self.feature_tokenizer(x_vars, frame_idx, feature_names=feature_names)
        agg = self.phi(encodings).mean(dim=1)
        node_embedding = agg + self.rho(agg)
        node_embedding = einops.rearrange(
            node_embedding,
            "(batch time ensemble grid) d -> (batch ensemble grid) (time d)",
            batch=batch,
            time=n_time,
            ensemble=ensemble,
            grid=grid,
        )
        return torch.cat((node_embedding, node_attributes_data), dim=-1)


class HierarchicalEmbedder(nn.Module):
    """Pools a node's raw per-variable values in two stages: first the variables at each
    height/pressure level into one token per level, then the level tokens into one final
    embedding - instead of pooling every variable across every level at once. Lets the model
    treat "which physical variable" (level-independent) and "which level" (added only once
    the per-level summary already exists) as separate concerns.
    """

    def __init__(self, feature_names, hidden_dim, d_model, nhead, mean=None, std=None):
        super().__init__()
        self.feature_names = feature_names
        tokenizer = Tokenizer(feature_names)

        if mean is None:
            mean = torch.zeros(len(feature_names))
        if std is None:
            std = torch.ones(len(feature_names))
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.variable_embedding = nn.Embedding(tokenizer.num_variables, hidden_dim)
        self.register_buffer("variable_idx", torch.tensor(tokenizer.variable_idx))

        # group feature indices by level (0 = no-level, from parse_feature_name) - fixed at
        # construction time from feature_names, one group becomes one stage-1 token.
        groups: dict[float, list[int]] = {}
        for i, level in enumerate(tokenizer.levels):
            groups.setdefault(level, []).append(i)
        self.level_groups = [group for _level, group in sorted(groups.items())]
        group_levels = torch.tensor(sorted(groups.keys()), dtype=torch.float32)
        self.register_buffer("level_pe", sinusoidal_positional_encoding(group_levels, d_model))

        self.value_proj = nn.Linear(hidden_dim + 1, d_model)
        self.stage1_pool = ClsSelfAttentionPool(d_model, nhead)
        self.stage2_pool = ClsSelfAttentionPool(d_model, nhead)
        self.output_dim = d_model

    def forward(self, x, node_attributes_data, feature_names=None):
        """x: (batch, time, ensemble, grid, vars) -> (batch ensemble grid, time d_model + attrs)"""
        if feature_names is not None:
            msg = "HierarchicalEmbedder does not support a reduced feature_names subset yet."
            raise NotImplementedError(msg)

        batch, n_time, ensemble, grid, _n_vars = x.shape
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")
        n_rows = x_vars.shape[0]
        normalized = (x_vars - self.mean) / self.std

        level_tokens = []
        for group in self.level_groups:
            group_idx = torch.tensor(group, device=x.device)
            values = normalized[:, group_idx].unsqueeze(-1)
            variable_embedding = self.variable_embedding(self.variable_idx[group_idx])
            variable_embedding = variable_embedding.unsqueeze(0).expand(n_rows, -1, -1)
            tokens = self.value_proj(torch.cat([variable_embedding, values], dim=-1))
            level_tokens.append(self.stage1_pool(tokens))

        level_tokens = torch.stack(level_tokens, dim=1) + self.level_pe.unsqueeze(0)
        node_embedding = self.stage2_pool(level_tokens)

        node_embedding = einops.rearrange(
            node_embedding,
            "(batch time ensemble grid) d -> (batch ensemble grid) (time d)",
            batch=batch,
            time=n_time,
            ensemble=ensemble,
            grid=grid,
        )
        return torch.cat((node_embedding, node_attributes_data), dim=-1)
