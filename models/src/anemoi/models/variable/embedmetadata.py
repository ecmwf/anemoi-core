# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from torch import nn

from anemoi.models.variable.variablevocabular import VariableVocabulary


class EmbedMetadata(nn.Module):
    """
    Initialize the metadata embedding layer.

    Constructs the learnable embeddings and continuous encoders used to represent
    physical variable metadata. The provided foundation vocabulary defines the
    superset of variables known to the model and the mappings from categorical
    metadata to their corresponding foundation indices.

    Parameters
    ----------
    vocabular : VariableVocabulary
        Foundation variable vocabulary containing the physical variable
        specifications and categorical index mappings.

    emb_dim : int
        Dimension of the resulting variable metadata embeddings.
    """

    def __init__(self, vocabular: VariableVocabulary, emb_dim: int) -> None:
        super().__init__()

        assert isinstance(
            vocabular, VariableVocabulary
        ), f"expecting vocabular object of class VariableVocabulary"

        self.vocabular = vocabular

        self.num_params = len(self.vocabular.param_to_id)
        self.num_vertical_level_types = len(self.vocabular.vertical_type_to_id)
        self.num_temporal_operators = len(self.vocabular.temporal_operator_to_id)

        # categorical
        self.emb_param = nn.Embedding(self.num_params, emb_dim)
        self.emb_vertical__level_types = nn.Embedding(
            self.num_vertical_level_types, emb_dim
        )
        self.emb_temporal_operator = nn.Embedding(self.num_temporal_operators, emb_dim)

        # continous
        self.encode_level_types = nn.Sequential(
            nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.encode_temporal_windows = nn.Sequential(
            nn.Linear(1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

        # masks
        self.no_vertical_levels = nn.Parameter(torch.zeros(emb_dim))
        self.no_temporal_window = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, variables: str | list[str]) -> torch.Tensor:
        """
        Embed metadata for the variables of the current domain.

        Maps the variable names of the current domain (e.g. global, stretched-grid,
        or LAM data) to their corresponding entries in the foundation variable
        vocabulary. The foundation vocabulary represents the superset of variables
        known to the model and provides the physical metadata used to construct the
        variable embeddings.

        Parameters
        ----------
        variables : str | list[str]
            Variable name or ordered list of variable names present in the current
            domain.

        Returns
        -------
        torch.Tensor
            Metadata embeddings with shape ``[num_variables, emb_dim]``.
        """
        if isinstance(variables, str):
            variables = [variables]

        current_vocabular = self.vocabular.get_variables(variables)

        device = self.emb_param.weight.device
        # categorical
        param_ids = current_vocabular.param_ids.to(device=device)
        temporal_operator_ids = current_vocabular.temporal_operator_ids.to(
            device=device
        )
        vertical_type_ids = current_vocabular.vertical_type_ids.to(device=device)

        # continous
        temporal_windows = current_vocabular.temporal_windows.to(device=device)
        vertical_levels = current_vocabular.vertical_levels.to(device=device)

        # mask
        has_temporal_window = current_vocabular.has_temporal_window.to(device=device)
        has_level = current_vocabular.has_vertical_level.to(device=device)

        emb_param = self.emb_param(param_ids)
        emb_temp_op = self.emb_temporal_operator(temporal_operator_ids)
        emb_vertical_type = self.emb_vertical__level_types(vertical_type_ids)

        _vertical_levels = torch.where(
            has_level,
            vertical_levels,
            torch.zeros_like(vertical_levels, device=device),
        )
        _temporal_windows = torch.where(
            has_temporal_window,
            temporal_windows,
            torch.zeros_like(temporal_windows, device=device),
        )

        emb_temp_windows = self.encode_temporal_windows(_temporal_windows[..., None])
        emb_vertical_levels = self.encode_level_types(_vertical_levels[..., None])

        emb_vertical_levels = torch.where(
            has_level[..., None],
            emb_vertical_levels,
            self.no_vertical_levels[None, :],
        )
        emb_temp_windows = torch.where(
            has_temporal_window[..., None],
            emb_temp_windows,
            self.no_temporal_window[None, :],
        )

        return (
            emb_param
            + emb_vertical_type
            + emb_temp_op
            + emb_vertical_levels
            + emb_temp_windows
        )
