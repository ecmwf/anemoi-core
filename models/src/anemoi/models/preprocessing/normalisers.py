# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from abc import abstractmethod

import einops
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def build_normaliser(**kwargs):
    if kwargs.get("_target_") not in [
        "anemoi.models.preprocessing.normaliser.InputNormaliser",
        "anemoi.models.preprocessing.normalizer.InputNormalizer",
        None,
    ]:
        # If the normaliser is an Hydra instantiation, use the config directly
        # and does not use the default Normaliser class
        # and does not use the old class with a typo 'z'
        # This may never happens, but if it does, this is the place to implement it
        raise NotImplementedError("TODO: use hydra instanciate for this custom normaliser")

    return Normaliser(**kwargs)


def build_denormaliser(**kwargs):
    if kwargs.get("_target_") not in [
        "anemoi.models.preprocessing.normaliser.InputNormaliser",
        "anemoi.models.preprocessing.normalizer.InputNormalizer",
        None,
    ]:
        # If the normaliser is an Hydra instantiation, use the config directly
        # and does not use the default Normaliser class
        # and does not use the old class with a typo 'z'
        # This may never happens, but if it does, this is the place to implement it
        raise NotImplementedError("TODO: use hydra instanciate for this custom normaliser")

    return DeNormaliser(**kwargs)


class BaseNormaliser(torch.nn.Module):

    def __init__(self, normaliser, name_to_index, statistics, dimensions_order, **kwargs) -> None:
        """Initialize the normalizer.

        Parameters
        ----------
        config : DotDict
            configuration object of the processor
        data_indices : IndexCollection
            Data indices for input and output variables
        statistics : dict
            Data statistics dictionary
        """
        super().__init__()

        self.dimensions_order = dimensions_order

        self.methods = {}
        for k in name_to_index:
            self.methods[k] = normaliser.get(k, normaliser.get("default", "none"))

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]
        mean = statistics["mean"]
        stdev = statistics["stdev"]

        _norm_add = np.zeros((minimum.size,), dtype=np.float32)
        _norm_mul = np.ones((minimum.size,), dtype=np.float32)

        for name, i in name_to_index.items():
            method = self.methods[name]

            if method == "mean-std":
                LOGGER.debug(f"Normalizing: {name} is mean-std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = -mean[i] / stdev[i]
                continue

            if method == "std":
                LOGGER.debug(f"Normalizing: {name} is std-normalised.")
                if stdev[i] < (mean[i] * 1e-6):
                    warnings.warn(f"Normalizing: the field seems to have only one value {mean[i]}")
                _norm_mul[i] = 1 / stdev[i]
                _norm_add[i] = 0
                continue

            if method == "min-max":
                LOGGER.debug(f"Normalizing: {name} is min-max-normalised to [0, 1].")
                x = maximum[i] - minimum[i]
                if x < 1e-9:
                    warnings.warn(f"Normalizing: the field {name} seems to have only one value {maximum[i]}.")
                _norm_mul[i] = 1 / x
                _norm_add[i] = -minimum[i] / x
                continue

            if method == "max":
                LOGGER.debug(f"Normalizing: {name} is max-normalised to [0, 1].")
                _norm_mul[i] = 1 / maximum[i]
                continue

            if method == "none":
                LOGGER.info(f"Normalizing: {name} is not normalized.")
                continue

            raise ValueError[f"Unknown normalisation method for {name}: {method}"]

        # use self.dimensions_order to know which dimension is the "variables" dimension
        # new_shape is like : "1 variables 1" or "1 1 variables"
        new_shape = ["variables" if d == "variables" else "1" for d in self.dimensions_order]
        new_shape = " ".join([str(d) for d in new_shape])
        _norm_add = einops.rearrange(_norm_add, f"variables -> {new_shape}")
        _norm_mul = einops.rearrange(_norm_mul, f"variables -> {new_shape}")

        # register buffer - this will ensure they get copied to the correct device(s)
        self.register_buffer("_norm_mul", torch.from_numpy(_norm_mul), persistent=True)
        self.register_buffer("_norm_add", torch.from_numpy(_norm_add), persistent=True)

    @abstractmethod
    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def __repr__(self):
        res = self.__class__.__name__
        res += (
            f"(shape={tuple(self._norm_mul.shape)}, methods={', '.join(f'{k}={v}' for k, v in self.methods.items())})"
        )
        res = res[:50] + "..." if len(res) > 50 else res
        return res


class Normaliser(BaseNormaliser):
    """Normalizes input data with a configurable method."""

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        data.mul_(self._norm_mul).add_(self._norm_add)
        return data


class DeNormaliser(BaseNormaliser):
    """Normalizes input data with a configurable method."""

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        data.sub_(self._norm_add).div_(self._norm_mul)
        return data
