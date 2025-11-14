# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch

LOGGER = logging.getLogger(__name__)


class NormaliserMixin:
    """Mixin class for normalising values.

    This class provides a method to normalise values within each group specified by an index tensor.
    Supported normalisation methods: None, 'l1', 'l2', 'unit-max', 'unit-range', 'unit-std'.
    """

    def compute_nongrouped_statistics(self, values: torch.Tensor, *_args) -> tuple[float, ...]:
        if self.norm == "l1":
            statistics = (torch.sum(values),)

        elif self.norm == "l2":
            statistics = (torch.norm(values),)

        elif self.norm == "unit-max":
            statistics = (torch.amax(values),)

        elif self.norm == "unit-range":
            statistics = torch.amin(values), torch.amax(values)

        elif self.norm == "unit-std":
            std = torch.std(values)
            if std == 0:
                LOGGER.warning(f"Std. dev. of the {self.__class__.__name__} values is 0. Normalisation is skipped.")
                return (1,)

            statistics = (std,)

        assert (
            statistics[-1] != 0
        ), f"Normalisation by zero encountered in {self.__class__.__name__} with norm '{self.norm}'."
        return statistics

    def compute_grouped_statistics(
        self, values: torch.Tensor, index: torch.Tensor, num_groups: int, dtype, device
    ) -> tuple[torch.Tensor, ...]:
        if self.norm == "l1":
            group_sum = torch.zeros(num_groups, values.shape[1], dtype=dtype, device=device)
            group_sum = group_sum.index_add(0, index, values)
            group_statistics = (group_sum[index],)

        elif self.norm == "l2":
            group_sq = torch.zeros(num_groups, values.shape[1], dtype=dtype, device=device)
            group_sq = group_sq.index_add(0, index, values**2)
            group_norm = torch.sqrt(group_sq)
            group_statistics = (group_norm[index],)

        elif self.norm == "unit-max":
            group_max = torch.full((num_groups, values.shape[1]), float("-inf"), dtype=dtype, device=device)
            group_max = group_max.index_reduce(0, index, values, reduce="amax")
            group_statistics = (group_max[index],)

        elif self.norm == "unit-range":
            group_min = torch.full((num_groups, values.shape[1]), float("inf"), dtype=dtype, device=device)
            group_max = torch.full((num_groups, values.shape[1]), float("-inf"), dtype=dtype, device=device)
            group_min = group_min.index_reduce(0, index, values, reduce="amin")
            group_max = group_max.index_reduce(0, index, values, reduce="amax")
            denom = group_max - group_min
            denom[denom == 0] = 1  # avoid division by zero
            group_statistics = group_min[index], denom[index]

        elif self.norm == "unit-std":
            # Compute mean
            group_sum = torch.zeros(num_groups, values.shape[1], dtype=dtype, device=device)
            group_count = torch.zeros(num_groups, values.shape[1], dtype=dtype, device=device)
            ones = torch.ones_like(values)
            group_sum = group_sum.index_add(0, index, values)
            group_count = group_count.index_add(0, index, ones)
            group_mean = group_sum / group_count.clamp(min=1)
            # Compute variance
            mean_expanded = group_mean[index]
            sq_diff = (values - mean_expanded) ** 2
            group_var = torch.zeros_like(group_sum)
            group_var = group_var.index_add(0, index, sq_diff)
            group_var = group_var / group_count.clamp(min=1)
            group_std = torch.sqrt(group_var)
            # Avoid division by zero
            group_std[group_std == 0] = 1
            group_statistics = (group_std[index],)

        assert torch.all(
            group_statistics[-1] != 0
        ), f"Normalisation by zero encountered in {self.__class__.__name__} with norm '{self.norm}'."
        return group_statistics

    def normalise(self, values: torch.Tensor, *args) -> torch.Tensor:
        """Normalise the given values.

        It supports different normalisation methods: None, 'l1',
        'l2', 'unit-max', 'unit-range' and 'unit-std'.

        Parameters
        ----------
        values : torch.Tensor of shape (N, M)
            Values to normalise.

        Returns
        -------
        torch.Tensor
            Normalised values.
        """
        assert self.norm in {
            None,
            "l1",
            "l2",
            "unit-std",
            "unit-max",
            "unit-range",
        }, f"Attribute normalisation '{self.norm}' is not valid. Options are: 'l1', 'l2', 'unit-max', 'unit-range', 'unit-std'."

        if self.norm is None:
            LOGGER.debug(f"{self.__class__.__name__} values are not normalised.")
            return values

        if self.norm_by_group:
            statistics = self.compute_grouped_statistics(values, *args, device=values.device, dtype=values.dtype)
        else:
            statistics = self.compute_nongrouped_statistics(values, *args)

        if self.norm == "unit-range":
            values = values - statistics[0]

        return values / statistics[-1]
