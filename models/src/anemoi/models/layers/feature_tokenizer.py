# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import torch
import torch.nn as nn

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

    def forward(self, values, frame_idx, feature_names=None):
        """values: (batch, n_features) -> (batch, n_features, 2 + 2 * dim)

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

        frame_encoding = frame_idx.float().view(batch_size, 1, 1).expand(batch_size, n_features, 1)

        return torch.cat([value, variable_embedding, level_encoding, frame_encoding], dim=-1)
