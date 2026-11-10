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
import torch.utils.checkpoint

from anemoi.models.layers.cls_pooling import ClsSelfAttentionPool
from anemoi.models.layers.input_transform import BaseInputTransform
from anemoi.models.layers.set_transformer import PMA
from anemoi.models.variable.variablevocabular import VariableVocabulary


class Tokenizer:
    """Turns variable names into indices: which physical variable, what pressure level, whether it has one."""

    def __init__(self, variable_metadata: dict):
        vocabulary = VariableVocabulary.from_foundation(variable_metadata)
        specifications = vocabulary.specification_by_name
        forcings = sorted(name for name, spec in specifications.items() if spec.kind == "forcing")
        forcing_to_idx = {name: len(vocabulary.param_to_id) + i for i, name in enumerate(forcings)}
        self.num_variables = len(vocabulary.param_to_id) + len(forcings)

        self.variable_idx = {}
        self.level = {}
        for name, spec in specifications.items():
            is_pressure = spec.kind == "physical" and spec.vertical_coordinate.type == "pressure"
            self.variable_idx[name] = (
                forcing_to_idx[name] if spec.kind == "forcing" else vocabulary.param_to_id[spec.param]
            )
            self.level[name] = float(spec.vertical_coordinate.level) if is_pressure else 0.0

    def lookup(self, variables: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variable index, level and has-level flag of each name, in the order given."""
        variable_idx = torch.tensor([self.variable_idx[name] for name in variables], device=device)
        levels = torch.tensor([self.level[name] for name in variables], device=device)
        return variable_idx, levels, levels > 0


def sinusoidal_positional_encoding(positions, dim):
    positions = positions.float().unsqueeze(-1)
    i = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    freqs = torch.exp(-i * (math.log(10000.0) / (dim // 2)))
    angles = positions * freqs
    pe = torch.zeros(positions.shape[0], dim, device=positions.device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class FeatureTokenizer(nn.Module):
    """One vector per feature: physical-variable embedding + level positional encoding, alongside the raw value."""

    def __init__(self, variable_metadata, emb_dim):
        super().__init__()
        self.tokenizer = Tokenizer(variable_metadata)
        self.emb_dim = emb_dim

        self.variable_embedding = nn.Embedding(self.tokenizer.num_variables, emb_dim)
        self.no_level_embedding = nn.Parameter(torch.randn(emb_dim))

        self.value_encoder = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, values, frame_idx, variables):
        """values: (batch, n_features) -> (batch, n_features, 2 + 3 * emb_dim)

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

        variables: names of the columns of `values`, in order.
        """
        batch_size, n_features = values.shape

        variable_idx, levels, has_level = self.tokenizer.lookup(variables, values.device)

        variable_embedding = self.variable_embedding(variable_idx).unsqueeze(0).expand(batch_size, -1, -1)

        level_pe = sinusoidal_positional_encoding(levels, self.emb_dim).unsqueeze(0).expand(batch_size, -1, -1)
        no_level_embedding = self.no_level_embedding.view(1, 1, -1).expand(batch_size, n_features, -1)
        level_encoding = torch.where(has_level.view(1, -1, 1), level_pe, no_level_embedding)

        value = values.unsqueeze(-1)
        value_encoding = self.value_encoder(value)

        frame_encoding = frame_idx.float().view(batch_size, 1, 1).expand(batch_size, n_features, 1)

        return torch.cat([value, value_encoding, variable_embedding, level_encoding, frame_encoding], dim=-1)


class PMAEmbedder(BaseInputTransform):
    """Turns a node's raw per-variable values into a single vector via attention pooling over
    the variables as a set, so the resulting encoding doesn't depend on a fixed variable count
    or order.

    Pools with PMA (Pooling by Multihead Attention, Lee et al. 2019): a learned query vector
    cross-attends over the variable tokens. The query is a parameter, not derived from the
    input, and the variable tokens do not attend back to it or to each other - only the
    query's output is kept as the node embedding.

    Algorithm::

        tokens = []
        for param in variables:
            level_encoding = level_pe[param] if param.has_level else no_level_embedding
            token = concat(value[param], value_encoder(value[param]), embedding(param), level_encoding, frame_id)
            tokens.append(token)

        tokens_proj = Linear(tokens)
        node_embedding = MHA(key=tokens_proj, value=tokens_proj, query=learned_query)

        output = concat(node_embedding, node_attributes_data)
    """

    def __init__(self, variable_metadata, emb_dim, out_dim, num_heads):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(variable_metadata, emb_dim)
        self.input_proj = nn.Linear(2 + 3 * emb_dim, out_dim)
        self.pma = PMA(out_dim, num_heads, num_seeds=1)
        self.output_dim = out_dim

    def forward(self, x, node_attributes_data, variables):
        """x: (batch, time, ensemble, grid, vars) -> (batch ensemble grid, time out_dim + attrs)

        Tokenizes and pools each (node, timestep) row independently, giving each row a
        frame_idx so it carries which timestep it came from, then concatenates timesteps back
        into the feature axis and appends node_attributes_data (static per-node features, with
        no time axis of their own).
        """
        batch, n_time, ensemble, grid, _n_vars = x.shape
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")
        frame_idx = torch.arange(n_time, device=x.device).repeat_interleave(ensemble * grid).repeat(batch)
        encodings = self.feature_tokenizer(x_vars, frame_idx, variables)
        x_vars = self.pma(self.input_proj(encodings)).squeeze(1)
        x_vars = einops.rearrange(
            x_vars,
            "(batch time ensemble grid) d -> (batch ensemble grid) (time d)",
            batch=batch,
            time=n_time,
            ensemble=ensemble,
            grid=grid,
        )
        return self.append_node_attributes(x_vars, node_attributes_data)


class DeepSetEmbedder(BaseInputTransform):
    """Turns a node's raw per-variable values into a single vector via Deep Sets pooling
    (Zaheer et al. 2017): each variable is tokenized and transformed independently (phi), then
    aggregated by a permutation-invariant mean - cheap and simple to train, with no
    cross-variable interaction during pooling, unlike attention-based pooling.

    rho is applied residually because the mean alone gives the gradient no direct path back
    to phi's output.

    Algorithm::

        tokens = []
        for param in variables:
            level_encoding = level_pe[param] if param.has_level else no_level_embedding
            token = concat(value[param], value_encoder(value[param]), embedding(param), level_encoding)
            tokens.append(token)

        mlp_out = MLP(tokens)                    # per variable
        agg = mean(mlp_out)
        node_embedding = agg + MLP(LayerNorm(agg))

        output = concat(node_embedding, node_attributes_data)
    """

    def __init__(self, variable_metadata, emb_dim, out_dim):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(variable_metadata, emb_dim)
        self.phi = nn.Sequential(
            nn.Linear(2 + 3 * emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.rho_norm = nn.LayerNorm(out_dim)
        self.output_dim = out_dim

    def forward(self, x, node_attributes_data, variables):
        """x: (batch, time, ensemble, grid, vars) -> (batch ensemble grid, time out_dim + attrs)"""
        batch, n_time, ensemble, grid, _n_vars = x.shape
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")
        frame_idx = torch.zeros(x_vars.shape[0], device=x.device)
        encodings = self.feature_tokenizer(x_vars, frame_idx, variables)
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
        return self.append_node_attributes(node_embedding, node_attributes_data)


class HierarchicalEmbedder(BaseInputTransform):
    """Pools a node's raw per-variable values in two stages, each a multihead attention call:
    first the variables at each pressure level into one token per level, then the level
    tokens into one final embedding - instead of pooling every variable across every level at
    once. Lets the model treat "which physical variable" (level-independent) and "which level"
    (added only once the per-level summary already exists) as separate concerns.

    Algorithm::

        level_tokens = []   # stage 1
        for level, level_group in level_groups:
            tokens = []
            for param in level_group:
                token = Linear(value[param]) + Linear(embedding(param))
                tokens.append(token)
            level_tokens.append(multihead_attention(tokens) + level_pe[level])

        level_tokens = stack(level_tokens)
        node_embedding = multihead_attention(level_tokens)      # stage 2

        output = concat(node_embedding, node_attributes_data)
    """

    def __init__(self, variable_metadata, emb_dim, out_dim, num_heads):
        super().__init__()
        self.tokenizer = Tokenizer(variable_metadata)

        self.variable_embedding = nn.Embedding(self.tokenizer.num_variables, emb_dim)
        self.variable_proj = nn.Linear(emb_dim, out_dim, bias=False)
        self.value_proj = nn.Linear(1, out_dim)
        self.stage1_pool = ClsSelfAttentionPool(out_dim, num_heads)
        self.stage2_pool = ClsSelfAttentionPool(out_dim, num_heads)
        self.output_dim = out_dim

    def _level_groups(self, variables, device):
        """Column indices of each level's variables, padded to the largest group, with the padding mask."""
        groups: dict[float, list[int]] = {}
        for i, name in enumerate(variables):
            groups.setdefault(self.tokenizer.level[name], []).append(i)
        group_levels = torch.tensor(sorted(groups), dtype=torch.float32, device=device)
        level_groups = [group for _level, group in sorted(groups.items())]

        max_group_size = max(len(group) for group in level_groups)
        group_idx_padded = torch.zeros(len(level_groups), max_group_size, dtype=torch.long, device=device)
        group_mask = torch.ones(len(level_groups), max_group_size, dtype=torch.bool, device=device)
        for i, group in enumerate(level_groups):
            group_idx_padded[i, : len(group)] = torch.tensor(group, dtype=torch.long, device=device)
            group_mask[i, : len(group)] = False
        return group_levels, group_idx_padded, group_mask

    def forward(self, x, node_attributes_data, variables):
        """x: (batch, time, ensemble, grid, vars) -> (batch ensemble grid, time out_dim + attrs)"""
        batch, n_time, ensemble, grid, _n_vars = x.shape
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")

        variable_idx, _levels, _has_level = self.tokenizer.lookup(variables, x.device)
        group_levels, group_idx_padded, group_mask = self._level_groups(variables, x.device)
        n_groups, max_group_size = group_idx_padded.shape

        variable_embedding = self.variable_embedding(variable_idx[group_idx_padded])
        variable_component = self.variable_proj(variable_embedding).unsqueeze(0)

        row_chunk_size = max(1, self.stage1_pool._MAX_CHUNK // n_groups)
        level_tokens_chunks = []
        for x_vars_chunk in x_vars.split(row_chunk_size, dim=0):
            n_rows_chunk = x_vars_chunk.shape[0]
            values = x_vars_chunk[:, group_idx_padded].unsqueeze(-1)
            tokens = self.value_proj(values) + variable_component

            tokens = tokens.reshape(n_rows_chunk * n_groups, max_group_size, -1)
            mask = group_mask.unsqueeze(0).expand(n_rows_chunk, -1, -1).reshape(n_rows_chunk * n_groups, -1)
            pooled = torch.utils.checkpoint.checkpoint(self.stage1_pool, tokens, mask, use_reentrant=False)
            level_tokens_chunks.append(pooled.reshape(n_rows_chunk, n_groups, -1))
        level_tokens = torch.cat(level_tokens_chunks, dim=0)

        level_tokens = level_tokens + sinusoidal_positional_encoding(group_levels, self.output_dim).unsqueeze(0)
        node_embedding = self.stage2_pool(level_tokens)

        node_embedding = einops.rearrange(
            node_embedding,
            "(batch time ensemble grid) d -> (batch ensemble grid) (time d)",
            batch=batch,
            time=n_time,
            ensemble=ensemble,
            grid=grid,
        )
        return self.append_node_attributes(node_embedding, node_attributes_data)


class LinearGroupedEmbedder(BaseInputTransform):
    """Turns a node's raw per-variable values into a single vector via a plain Linear per
    pressure level plus a separate Linear for the surface/single-level group, then
    attention-pools the resulting tokens (one per level, plus the surface token) into the
    final embedding.

    Each Linear has one input slot per variable the vocabulary knows of: one per physical
    variable on the pressure levels, one per single-level variable for the surface group.
    Slots of variables missing from the input are zero, so no attention or padding is needed
    within a level. Attention is kept only for the second stage, which combines the much
    smaller set of per-level tokens.

    Algorithm::

        for level in pressure_levels:
            input = stack(value[param] for param in pressure_params)   # zero where a param is missing
            level_token[level] = Linear(input) + level_pe[level]

        surface_input = stack(value[param] for param in surface_params)
        surface_token = Linear(surface_input)

        tokens = concat(level_token for each level, surface_token)
        node_embedding = multihead_attention(tokens)

        output = concat(node_embedding, node_attributes_data)
    """

    def __init__(self, variable_metadata, out_dim, num_heads):
        super().__init__()
        self.tokenizer = Tokenizer(variable_metadata)

        pressure_variables = sorted(
            {self.tokenizer.variable_idx[name] for name, level in self.tokenizer.level.items() if level > 0}
        )
        surface_names = sorted(name for name, level in self.tokenizer.level.items() if level == 0)
        self.pressure_slot = {variable: i for i, variable in enumerate(pressure_variables)}
        self.surface_slot = {name: i for i, name in enumerate(surface_names)}

        self.level_proj = nn.Linear(len(pressure_variables), out_dim)
        self.surface_proj = nn.Linear(len(surface_names), out_dim)
        self.pool = ClsSelfAttentionPool(out_dim, num_heads)
        self.output_dim = out_dim

    def forward(self, x, node_attributes_data, variables):
        """x: (batch, time, ensemble, grid, vars) -> (batch ensemble grid, time out_dim + attrs)"""
        batch, n_time, ensemble, grid, _n_vars = x.shape
        x_vars = einops.rearrange(x, "batch time ensemble grid vars -> (batch time ensemble grid) vars")
        n_rows = x_vars.shape[0]

        levels = sorted({self.tokenizer.level[name] for name in variables} - {0.0})
        level_pos = {level: i for i, level in enumerate(levels)}
        n_slots = len(self.pressure_slot)

        level_columns, level_targets, surface_columns, surface_targets = [], [], [], []
        for i, name in enumerate(variables):
            level = self.tokenizer.level[name]
            if level > 0:
                level_columns.append(i)
                level_targets.append(level_pos[level] * n_slots + self.pressure_slot[self.tokenizer.variable_idx[name]])
            else:
                surface_columns.append(i)
                surface_targets.append(self.surface_slot[name])

        level_values = x_vars.new_zeros(n_rows, len(levels) * n_slots)
        level_values[:, level_targets] = x_vars[:, level_columns]
        level_values = level_values.view(n_rows, len(levels), n_slots)
        group_levels = torch.tensor(levels, dtype=torch.float32, device=x.device)
        level_tokens = self.level_proj(level_values) + sinusoidal_positional_encoding(
            group_levels, self.output_dim
        ).unsqueeze(0)

        surface_values = x_vars.new_zeros(n_rows, len(self.surface_slot))
        surface_values[:, surface_targets] = x_vars[:, surface_columns]
        surface_token = self.surface_proj(surface_values).unsqueeze(1)

        tokens = torch.cat([level_tokens, surface_token], dim=1)
        node_embedding = self.pool(tokens)

        node_embedding = einops.rearrange(
            node_embedding,
            "(batch time ensemble grid) d -> (batch ensemble grid) (time d)",
            batch=batch,
            time=n_time,
            ensemble=ensemble,
            grid=grid,
        )
        return self.append_node_attributes(node_embedding, node_attributes_data)
