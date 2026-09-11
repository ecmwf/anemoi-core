# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch.nn as nn

from anemoi.models.layers.feature_tokenizer import FeatureTokenizer
from anemoi.models.layers.set_transformer import PMA


class Embedder(nn.Module):
    """FeatureTokenizer + PMA (Set Transformer, k=1 seed): turns a node's raw per-variable values into one vector."""

    def __init__(self, feature_names, dim, d_model, nhead, mean=None, std=None):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(feature_names, dim, mean=mean, std=std)
        self.input_proj = nn.Linear(1 + 2 * dim, d_model)
        self.pma = PMA(d_model, nhead, num_seeds=1)

    def forward(self, values):
        # values: (batch, n_features) -> (batch, d_model)
        encodings = self.feature_tokenizer(values)
        x = self.input_proj(encodings)
        return self.pma(x).squeeze(1)
