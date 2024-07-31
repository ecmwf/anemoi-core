from abc import ABC
from abc import abstractmethod

import numpy as np


class BasePressureLevelScaler(ABC):
    """Configurable method converting pressure level of variable to PTL scaling.

    Scaling variables depending on pressure levels (50 to 1000).
    """

    def __init__(self, slope: float = 1.0 / 1000, minimum: float = 0.0) -> None:
        self.slope = slope
        self.minimum = minimum

    @abstractmethod
    def scaler(self, plev) -> np.ndarray: ...


class LinearPressureLevelScaler(BasePressureLevelScaler):
    """Linear with slope self.slope, yaxis shift by self.minimum."""

    def scaler(self, plev) -> np.ndarray:
        return plev * self.slope + self.minimum


class ReluPressureLevelScaler(BasePressureLevelScaler):
    """Linear above self.minimum, taking constant value self.minimum below."""

    def scaler(self, plev) -> np.ndarray:
        return max(self.minimum, plev * self.slope)


class PolynomialPressureLevelScaler(BasePressureLevelScaler):
    """Polynomial scaling, (slope * plev)^2, yaxis shift by self.minimum."""

    def scaler(self, plev) -> np.ndarray:
        return (self.slope * plev) ** 2 + self.minimum


class NoPressureLevelScaler(BasePressureLevelScaler):
    """Constant scaling by 1.0."""

    def __init__(self, slope=0.0, minimum=1.0) -> None:
        assert (
            minimum == 1.0 and slope == 0
        ), "self.minimum must be 1.0 and self.slope 0.0 for no scaling to fit with definition of linear function."
        super().__init__(slope=0.0, minimum=1.0)

    def scaler(self, plev) -> np.ndarray:
        # no scaling, always return 1.0
        return 1.0
