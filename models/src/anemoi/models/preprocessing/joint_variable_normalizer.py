# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.models.utils.variables import parse_feature_name


class JointVariableNormalizer(InputNormalizer):
    """Normalizes all levels of the same physical variable (e.g. u_50..u_1000) with a
    single shared mean/stdev, instead of each level independently - per-level normalization would
    flatten the real variance difference between levels (wind at 1000 hPa doesn't behave like wind
    at 50 hPa).

    Assumes an equal sample count per column (true here: every column comes from the same
    dataset/time range), so pooling directly from mean/stdev (without the raw sums/squares) is exact.
    """

    def __init__(self, config=None, data_indices: Optional[IndexCollection] = None, statistics: Optional[dict] = None):
        super().__init__(config=config, data_indices=data_indices, statistics=statistics)

        name_to_index_training_input = self.data_indices.data.input.name_to_index
        mean = statistics["mean"]
        stdev = statistics["stdev"]
        physical_variable = {
            name: parse_feature_name(name, variable_only=True) for name in name_to_index_training_input
        }

        group_sum, group_second_moment, group_count = {}, {}, {}
        for name, idx in name_to_index_training_input.items():
            var = physical_variable[name]
            group_sum[var] = group_sum.get(var, 0.0) + mean[idx]
            group_second_moment[var] = group_second_moment.get(var, 0.0) + (stdev[idx] ** 2 + mean[idx] ** 2)
            group_count[var] = group_count.get(var, 0) + 1

        for name, idx in name_to_index_training_input.items():
            method = self.methods.get(name, self.default)
            if method not in ("mean-std", "std"):
                continue  # min-max/max/none don't depend on mean/stdev, nothing to pool

            var = physical_variable[name]
            pooled_mean = group_sum[var] / group_count[var]
            pooled_second_moment = group_second_moment[var] / group_count[var]
            pooled_stdev = (pooled_second_moment - pooled_mean**2) ** 0.5

            if method == "mean-std":
                self._norm_mul[idx] = 1 / pooled_stdev
                self._norm_add[idx] = -pooled_mean / pooled_stdev
            elif method == "std":
                self._norm_mul[idx] = 1 / pooled_stdev
                self._norm_add[idx] = 0
