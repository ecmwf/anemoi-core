# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import einops
import torch
import torch.nn as nn

from anemoi.models.layers.cls_pooling import ClsSelfAttentionPool
from anemoi.models.layers.set_transformer import PMA
from anemoi.models.utils.variables import parse_feature_name


class Tokenizer:
    """Turns raw feature names into indices: which physical variable, what pressure level, whether it has one."""

    def __init__(self, feature_names):
        self.feature_names = feature_names
        variables, levels, has_level = zip(*(parse_feature_name(name) for name in feature_names))

        unique_variables = sorted(set(variables))
        self.variable_to_idx = {variable: i for i, variable in enumerate(unique_variables)}
        self.num_variables = len(unique_variables)

        self.variable_idx = [self.variable_to_idx[v] for v in variables]
        self.levels = list(levels)
        self.has_level = list(has_level)


def sinusoidal_positional_encoding(positions, dim):
    positions = positions.float().unsqueeze(-1)
    i = torch.arange(dim // 2, dtype=torch.float32)
    freqs = torch.exp(-i * (math.log(10000.0) / (dim // 2)))
    angles = positions * freqs
    pe = torch.zeros(positions.shape[0], dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class FeatureTokenizer(nn.Module):
    """One vector per feature: physical-variable embedding + level positional encoding, alongside the raw value."""

    def __init__(self, feature_names, dim, mean=None, std=None):
        super().__init__()
        self.feature_names = feature_names
        tokenizer = Tokenizer(feature_names)
        self.feature_to_idx = {name: i for i, name in enumerate(feature_names)}

        if mean is None:
            mean = torch.zeros(len(feature_names))
        if std is None:
            std = torch.ones(len(feature_names))
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.variable_embedding = nn.Embedding(tokenizer.num_variables, dim)

        variable_idx = torch.tensor(tokenizer.variable_idx)
        self.register_buffer("variable_idx", variable_idx)

        levels = torch.tensor(tokenizer.levels, dtype=torch.float32)
        self.register_buffer("level_pe", sinusoidal_positional_encoding(levels, dim))
        self.register_buffer("has_level", torch.tensor(tokenizer.has_level, dtype=torch.bool))
        self.no_level_embedding = nn.Parameter(torch.randn(dim))

        # learned nonlinear representation of the value itself, alongside the raw scalar -
        # the raw scalar alone leaves all the work of extracting a useful representation to
        # the attention that follows; this gives the value its own capacity to do that first.
        self.value_encoder = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, values, frame_idx, feature_names=None):
        """values: (batch, n_features) -> (batch, n_features, 2 + 3 * dim)

        frame_idx: (batch,) tensor, one entry per row of `values`, giving which
        input timestep that row came from (0 = oldest). PMA pools each row's variable
        tokens permutation-equivariantly, so nothing about a row's own values marks
        which timestep it is - without this, telling t and t-1 apart would depend on
        the encoder learning to read raw column position downstream, which is fragile
        since the same shared tokenizer weights produce near-identical outputs for
        near-identical states. Passed through as a single raw scalar (like the value
        itself), not a dim-wide sinusoidal encoding - n_step_input is small (usually
        2), so it's just a 0/1 flag, and a full embedding would need far more data to
        learn than the network needs to pick up such a low-cardinality signal.

        feature_names, if given, must be a subset (or reordering) of the names this
        tokenizer was constructed with, in the same column order as `values`. Used to
        look up per-column buffers (variable embedding, level encoding, mean/std) by
        name for this call, instead of assuming `values` has every originally-known
        feature in its original order. Omit (default None) for the original, fixed
        full-feature-set behavior - existing callers are unaffected.
        """
        if feature_names is None:
            idx = None
            n_features = len(self.feature_names)
        else:
            try:
                idx = torch.tensor(
                    [self.feature_to_idx[name] for name in feature_names],
                    device=values.device,
                )
            except KeyError as e:
                msg = f"Unknown feature name {e.args[0]!r}: not in the set this tokenizer was built with."
                raise KeyError(msg) from e
            n_features = len(feature_names)

        if values.shape[1] != n_features:
            msg = f"values has {values.shape[1]} columns but {n_features} feature_names were given."
            raise ValueError(msg)

        batch_size = values.shape[0]

        variable_idx = self.variable_idx if idx is None else self.variable_idx[idx]
        level_pe_full = self.level_pe if idx is None else self.level_pe[idx]
        has_level = self.has_level if idx is None else self.has_level[idx]
        mean = self.mean if idx is None else self.mean[idx]
        std = self.std if idx is None else self.std[idx]

        variable_embedding = self.variable_embedding(variable_idx).unsqueeze(0).expand(batch_size, -1, -1)

        level_pe = level_pe_full.unsqueeze(0).expand(batch_size, -1, -1)
        no_level_embedding = self.no_level_embedding.view(1, 1, -1).expand(batch_size, n_features, -1)
        level_encoding = torch.where(has_level.view(1, -1, 1), level_pe, no_level_embedding)

        normalized_values = (values - mean) / std
        value = normalized_values.unsqueeze(-1)
        value_encoding = self.value_encoder(value)

        frame_encoding = frame_idx.float().view(batch_size, 1, 1).expand(batch_size, n_features, 1)

        return torch.cat([value, value_encoding, variable_embedding, level_encoding, frame_encoding], dim=-1)


class FlattenInputTransform(nn.Module):
    """No-op input_transform: passes raw variable values straight through, only reshaping the
    timestep axis into the feature axis and appending node_attributes_data. Zero learnable
    parameters, so configuring no input_transform at all doesn't change a model's behavior.
    """

    def __init__(self, feature_names=None):
        super().__init__()

    def forward(self, x, node_attributes_data, feature_names=None):
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)")
        return torch.cat((x_vars, node_attributes_data), dim=-1)


class PMAEmbedder(nn.Module):
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
        # pre-LN, same reasoning as ClsSelfAttentionPool's norm: normalizes what rho sees,
        # not the residual path itself.
        self.rho_norm = nn.LayerNorm(d_model)
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
        node_embedding = agg + self.rho(self.rho_norm(agg))
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
