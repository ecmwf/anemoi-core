# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class NormaliserMixin:
    """Mixin class for normalising attributes."""

    def normalise(self, values: np.ndarray) -> np.ndarray:
        """Normalise the given values.

        It supports different normalisation methods: None, 'l1',
        'l2', 'unit-max', 'unit-range' and 'unit-std'.

        Parameters
        ----------
        values : np.ndarray of shape (N, M)
            Values to normalise.

        Returns
        -------
        np.ndarray
            Normalised values.
        """
        if self.norm is None:
            LOGGER.debug(f"{self.__class__.__name__} values are not normalised.")
            return values
        if self.norm == "l1":
            return values / np.sum(values)
        if self.norm == "l2":
            return values / np.linalg.norm(values)
        if self.norm == "unit-max":
            return values / np.amax(values)
        if self.norm == "unit-range":
            return (values - np.amin(values)) / (np.amax(values) - np.amin(values))
        if self.norm == "unit-std":
            std = np.std(values)
            if std == 0:
                LOGGER.warning(f"Std. dev. of the {self.__class__.__name__} values is 0. Normalisation is skipped.")
                return values
            return values / std
        raise ValueError(
            f"Attribute normalisation \"{self.norm}\" is not valid. Options are: 'l1', 'l2', 'unit-max' or 'unit-std'."
        )
