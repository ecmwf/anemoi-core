# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Physical constants used by hydrostatic and refractivity computations.

Single source of truth for the derived-geopotential bounding
(:class:`anemoi.models.layers.bounding.HydrostaticGeopotential`) and the GNSS-RO
refractivity operator in anemoi-training, so both use identical values.
"""

# Standard gravity (m s^-2): converts geopotential metres to geopotential (m^2 s^-2).
G0: float = 9.80665
# Specific gas constant of dry air (J kg^-1 K^-1).
R_D: float = 287.06
# Virtual temperature: T_v = T (1 + VIRTUAL_TEMP_COEFF q) with q the specific humidity (kg/kg);
# VIRTUAL_TEMP_COEFF = R_v/R_d - 1.
VIRTUAL_TEMP_COEFF: float = 0.608
# Ratio of the gas constants of dry air and water vapour, R_d/R_v.
EPS_RD_RV: float = 0.622

# Smith-Weintraub refractivity N = K1 p/T + K2 e/T^2 with p, e in hPa.
K1_DRY: float = 77.6  # K hPa^-1
K2_WET: float = 3.73e5  # K^2 hPa^-1

# Normalisation methods whose affine parameters can be reconstructed from dataset statistics.
AFFINE_NORMALISERS: tuple[str, ...] = ("none", "mean-std", "std", "min-max", "max")


def affine_normaliser(method: str, *, mean: float, stdev: float, minimum: float, maximum: float) -> tuple[float, float]:
    """Return ``(mul, add)`` such that ``x_normalised = x * mul + add`` for a normaliser method.

    Mirrors :class:`anemoi.models.preprocessing.normalizer.InputNormalizer` exactly.
    """
    if method == "none":
        return 1.0, 0.0
    if method == "mean-std":
        return 1.0 / stdev, -mean / stdev
    if method == "std":
        return 1.0 / stdev, 0.0
    if method == "min-max":
        span = maximum - minimum
        return 1.0 / span, -minimum / span
    if method == "max":
        return 1.0 / maximum, 0.0
    msg = f"Unknown normaliser method {method!r}; expected one of {AFFINE_NORMALISERS}"
    raise ValueError(msg)
