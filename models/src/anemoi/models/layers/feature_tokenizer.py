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

    def forward(self, values):
        # values: (batch, n_features) -> (batch, n_features, 1 + 2 * dim)
        batch_size = values.shape[0]
        variable_embedding = self.variable_embedding(self.variable_idx).unsqueeze(0).expand(batch_size, -1, -1)

        level_pe = self.level_pe.unsqueeze(0).expand(batch_size, -1, -1)
        no_level_embedding = self.no_level_embedding.view(1, 1, -1).expand(batch_size, len(self.feature_names), -1)
        level_encoding = torch.where(self.has_level.view(1, -1, 1), level_pe, no_level_embedding)

        normalized_values = (values - self.mean) / self.std
        value = normalized_values.unsqueeze(-1)
        return torch.cat([value, variable_embedding, level_encoding], dim=-1)
