import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class NormalizerMixin:
    """Mixin class for normalizing attributes."""

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize the given values.

        It supports different normalization methods: None, 'l1',
        'l2', 'unit-max' and 'unit-std'.

        Parameters
        ----------
        values : np.ndarray of shape (N, M)
            Values to normalize.

        Returns
        -------
        np.ndarray
            Normalized values.
        """
        if self.norm is None:
            LOGGER.debug(f"{self.__class__.__name__} values are not normalized.")
            return values
        if self.norm == "l1":
            return values / np.sum(values)
        if self.norm == "l2":
            return values / np.linalg.norm(values)
        if self.norm == "unit-max":
            return values / np.amax(values)
        if self.norm == "unit-std":
            std = np.std(values)
            if std == 0:
                LOGGER.warning(f"Std. dev. of the {self.__class__.__name__} values is 0. Normalization is skipped.")
                return values
            return values / std
        raise ValueError(
            f"Attribute normalization \"{self.norm}\" is not valid. Options are: 'l1', 'l2', 'unit-max' or 'unit-std'."
        )
