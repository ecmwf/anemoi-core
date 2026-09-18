# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A positivity-preserving output transform for precipitation-like variables.

Written 2026-09-18 for arm C3 of the regional precipitation campaign. Nothing that already
existed is modified: both classes below are new, so every configuration that does not name them
behaves exactly as it did before.

The idea is the one the campaign owner pre-registered. The model is trained on
``v = sqrt(tp)`` instead of on ``tp`` and the post-processor returns ``tp = v**2``. Whatever the
network produces, the squared value is non-negative, an exact zero stays an exact zero, and no
value can become infinite the way an exponential transform can.

Two classes are needed, because the transform has two halves that live in different places.

``SqrtRemapper`` is the transform itself. It runs before the normaliser on the way in and after
it on the way out.

``SqrtAwareInputNormalizer`` is the ordinary normaliser with one addition: the statistics of the
named variables may be given in the configuration instead of being taken from the data store.
This is necessary and not cosmetic. The store knows the mean and the standard deviation of
``tp``; once the remapper runs, the number the normaliser divides by has to be the standard
deviation of ``sqrt(tp)``, which nothing in anemoi derives. Writing it in the configuration also
means the run records, in one readable place, exactly what the normaliser saw.
"""

import logging
from typing import Optional

import numpy as np
import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import BasePreprocessor
from anemoi.models.preprocessing.normalizer import InputNormalizer

LOGGER = logging.getLogger(__name__)


class SqrtRemapper(BasePreprocessor):
    """Square root on the way in, square on the way out, for the named variables.

    Configured like any other pre-processor, and it must sit BEFORE the normaliser in the
    ``processors`` block, because the processor list is walked in order on the way in and in
    reverse on the way out::

        processors:
          remapper:
            _target_: anemoi.models.preprocessing.sqrt_positive.SqrtRemapper
            config:
              default: none
              sqrt:
                - cp
                - tp
          normalizer:
            _target_: anemoi.models.preprocessing.sqrt_positive.SqrtAwareInputNormalizer
            ...

    Forward:  ``v = sqrt(max(x, 0))``.
    Inverse:  ``x = v**2``.

    The inverse is deliberately the square and not ``v * abs(v)``. The square is what makes the
    post-processed field non-negative by construction, which is the whole point of the arm; the
    signed alternative would let the model emit negative precipitation again. The cost is that
    the mapping is only a true inverse on the non-negative half line, which is where the truth
    lives. The forward clamp exists for the same reason and, on a store whose accumulated
    precipitation is non-negative, never changes a value.

    One difference from ``anemoi.models.preprocessing.remapper.Remapper``: this class accepts the
    ``data_index`` keyword. The diffusion downscaler pre-processes the direct-prediction channels
    on their own, passing a two-column tensor together with the data-space indices of those two
    columns, and the existing remapper rejects any tensor whose width is not the full variable
    count. Since ``tp`` and ``cp`` are direct predictions in this lane, that subset call is the
    only way they are ever processed.
    """

    supported_methods = ("sqrt", "none")

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        super().__init__(config, data_indices, statistics)

        name_to_index_input = self.data_indices.data.input.name_to_index
        name_to_index_output = self.data_indices.data.output.name_to_index
        self.num_data_input_vars = len(name_to_index_input)
        self.num_data_output_vars = len(name_to_index_output)

        if self.default not in self.supported_methods:
            msg = f"SqrtRemapper: unknown default method {self.default!r}, expected one of {self.supported_methods}"
            raise KeyError(msg)
        if self.default == "sqrt":
            msg = "SqrtRemapper: refusing default: sqrt, which would remap every variable. Name the variables."
            raise ValueError(msg)

        for name, method in self.methods.items():
            if method not in self.supported_methods:
                msg = f"SqrtRemapper: unknown method {method!r} for {name}"
                raise KeyError(msg)
            if name not in name_to_index_input:
                msg = f"SqrtRemapper: {name} is not a variable of this dataset"
                raise KeyError(msg)

        self.sqrt_names = sorted(name for name, method in self.methods.items() if method == "sqrt")
        self.forward_data_indices = sorted(name_to_index_input[name] for name in self.sqrt_names)
        self.inverse_data_indices = sorted(
            name_to_index_output[name] for name in self.sqrt_names if name in name_to_index_output
        )
        self._forward_set = set(self.forward_data_indices)
        self._inverse_set = set(self.inverse_data_indices)

        LOGGER.info(
            "SqrtRemapper: sqrt on %s, data-input indices %s, data-output indices %s "
            "(inverse is the square, so the post-processed field is non-negative by construction)",
            self.sqrt_names,
            self.forward_data_indices,
            self.inverse_data_indices,
        )

    @staticmethod
    def _positions(x: torch.Tensor, data_index, table: set, n_full: int, what: str) -> list:
        """Columns of ``x`` that must be transformed.

        ``data_index`` is the data-space index of every column of ``x``; when it is absent the
        tensor must be the full variable vector and its columns are their own data indices.
        """
        if data_index is not None:
            listed = data_index.tolist() if torch.is_tensor(data_index) else list(data_index)
            if len(listed) != x.shape[-1]:
                msg = (
                    f"SqrtRemapper {what}: data_index has {len(listed)} entries but the tensor has "
                    f"{x.shape[-1]} columns"
                )
                raise ValueError(msg)
            return [j for j, d in enumerate(listed) if int(d) in table]
        if x.shape[-1] == n_full:
            return sorted(table)
        msg = (
            f"SqrtRemapper {what}: tensor has {x.shape[-1]} columns, which is neither the full "
            f"variable count ({n_full}) nor accompanied by a data_index. Refusing to guess which "
            "columns are which variable."
        )
        raise ValueError(msg)

    def transform(self, x: torch.Tensor, in_place: bool = True, data_index=None, **kwargs) -> torch.Tensor:
        if not in_place:
            x = x.clone()
        for j in self._positions(x, data_index, self._forward_set, self.num_data_input_vars, "transform"):
            x[..., j] = torch.sqrt(torch.clamp(x[..., j], min=0.0))
        return x

    def inverse_transform(self, x: torch.Tensor, in_place: bool = True, data_index=None, **kwargs) -> torch.Tensor:
        if not in_place:
            x = x.clone()
        for j in self._positions(x, data_index, self._inverse_set, self.num_data_output_vars, "inverse_transform"):
            x[..., j] = x[..., j] ** 2
        return x


class SqrtAwareInputNormalizer(InputNormalizer):
    """The ordinary input normaliser, plus a per-variable statistics override.

    ``InputNormalizer`` builds its multipliers from the statistics of the data store. When a
    ``SqrtRemapper`` runs first, the variable the normaliser actually sees is ``sqrt(tp)``, whose
    mean and standard deviation are not in the store and are not derived anywhere in the code.
    This subclass lets the configuration state them::

        normalizer:
          _target_: anemoi.models.preprocessing.sqrt_positive.SqrtAwareInputNormalizer
          config:
            default: mean-std
            std: [cp, tp]
            statistics_override:
              tp: {stdev: 0.00123, mean: 0.00045}
              cp: {stdev: 0.00089}

    Only the keys that are given are replaced, and they are replaced on a private copy of the
    statistics arrays, so the dictionary the caller passed in is left exactly as it was and every
    other consumer of those statistics is unaffected. Everything else, including the existing
    ``remap`` feature, behaves as in the parent class.
    """

    STAT_KEYS = ("minimum", "maximum", "mean", "stdev")

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        override = {} if config is None else (config.get("statistics_override", None) or {})
        stripped = {} if config is None else {k: v for k, v in config.items() if k != "statistics_override"}

        statistics = dict(statistics)
        for key in self.STAT_KEYS:
            statistics[key] = np.array(statistics[key], copy=True)

        name_to_index = data_indices.data.input.name_to_index
        applied = {}
        for variable, block in dict(override).items():
            if variable not in name_to_index:
                msg = f"SqrtAwareInputNormalizer: statistics_override names {variable}, which is not a variable here"
                raise KeyError(msg)
            i = int(name_to_index[variable])
            for key, value in dict(block).items():
                if key not in self.STAT_KEYS:
                    msg = (
                        f"SqrtAwareInputNormalizer: statistics_override[{variable}] has key {key!r}; "
                        f"expected one of {self.STAT_KEYS}"
                    )
                    raise KeyError(msg)
                if i >= statistics[key].size:
                    msg = (
                        f"SqrtAwareInputNormalizer: {variable} is variable {i} but the {key} array has "
                        f"only {statistics[key].size} entries"
                    )
                    raise IndexError(msg)
                statistics[key][i] = float(value)
                applied[f"{variable}.{key}"] = float(value)

        super().__init__(stripped, data_indices, statistics)
        self.statistics_override_applied = applied
        if applied:
            LOGGER.info("SqrtAwareInputNormalizer: statistics taken from the configuration: %s", applied)
        else:
            LOGGER.warning("SqrtAwareInputNormalizer: no statistics_override given; this is the plain normaliser")
