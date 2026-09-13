# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Iterable
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn

from anemoi.models.data_indices.tensor import InputTensorIndex
from anemoi.models.layers.activations import leaky_hardtanh


class BaseBounding(nn.Module, ABC):
    """Abstract base class for bounding strategies.

    This class defines an interface for bounding strategies which are used to apply a specific
    restriction to the predictions of a model.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        """Initializes the bounding strategy.
        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict, optional
            A dictionary containing the statistics of the variables.
        name_to_index_stats : dict, optional
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary
        """
        super().__init__()

        self.name_to_index = name_to_index
        self.variables = variables
        self.data_index = self._create_index(variables=self.variables)
        self.statistics = statistics
        self.name_to_index_stats = name_to_index_stats

    def _create_index(self, variables: list[str]) -> InputTensorIndex:
        return torch.tensor([i for name, i in self.name_to_index.items() if name in variables], dtype=torch.int)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the bounding to the predictions.

        Parameters
        ----------
        x : torch.Tensor
            The tensor containing the predictions that will be bounded.

        Returns
        -------
        torch.Tensor
        A tensor with the bounding applied.
        """
        pass


class ReluBounding(BaseBounding):
    """Initializes the bounding with a ReLU activation / zero clamping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.relu(x[..., self.data_index])
        return x


class LeakyReluBounding(BaseBounding):
    """Initializes the bounding with a Leaky ReLU activation / zero clamping."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.leaky_relu(x[..., self.data_index])
        return x


class NormalizedReluBounding(BaseBounding):
    """Bounding variable with a ReLU activation and customizable normalized thresholds."""

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: list[float],
        normalizer: list[str],
        statistics: dict,
        name_to_index_stats: dict,
    ) -> None:
        """Initializes the NormalizedReluBounding with the specified parameters.

        Parameters
        ----------
        variables : list[str]
            A list of strings representing the variables that will be bounded.
        name_to_index : dict
            A dictionary mapping the variable names to their corresponding indices.
        statistics : dict
            A dictionary containing the statistics of the variables (mean, std, min, max, etc.).
        min_val : list[float]
            The minimum values for the ReLU activation. It should be given in the same order as the variables.
        normalizer : list[str]
            A list of normalization types to apply, one per variable. Options: 'mean-std', 'min-max', 'max', 'std'.
        name_to_index_stats : dict
            A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
        """
        if len(normalizer) != len(variables):
            raise ValueError(
                "The length of the normalizer list must match the number of variables in NormalizedReluBounding."
            )
        if len(min_val) != len(variables):
            raise ValueError(
                "The length of the min_val list must match the number of variables in NormalizedReluBounding."
            )
        if not all(norm in {"mean-std", "min-max", "max", "std"} for norm in normalizer):
            raise ValueError(
                "Each normalizer must be one of: 'mean-std', 'min-max', 'max', 'std' in NormalizedReluBounding."
            )

        super().__init__(
            variables=variables,
            name_to_index=name_to_index,
            statistics=statistics,
            name_to_index_stats=name_to_index_stats,
        )

        # Silently skip variables absent from this dataset (matches BaseBounding._create_index).
        kept = [(ii, var) for ii, var in enumerate(variables) if var in name_to_index]
        self.variables = [var for _, var in kept]
        self.min_val = [min_val[ii] for ii, _ in kept]
        self.normalizer = [normalizer[ii] for ii, _ in kept]

        # Create data index for the variables to be bounded in order from configuration
        self.data_index = torch.tensor([name_to_index[var] for var in self.variables], dtype=self.data_index.dtype)
        # Compute normalized min values
        norm_min_val = torch.zeros(len(self.variables), dtype=torch.float32)
        for ii, variable in enumerate(self.variables):
            stat_index = self.name_to_index_stats[variable]
            if self.normalizer[ii] == "mean-std":
                mean = self.statistics["mean"][stat_index]
                std = self.statistics["stdev"][stat_index]
                norm_min_val[ii] = (self.min_val[ii] - mean) / std
            elif self.normalizer[ii] == "min-max":
                min_stat = self.statistics["min"][stat_index]
                max_stat = self.statistics["max"][stat_index]
                norm_min_val[ii] = (self.min_val[ii] - min_stat) / (max_stat - min_stat)
            elif self.normalizer[ii] == "max":
                max_stat = self.statistics["max"][stat_index]
                norm_min_val[ii] = self.min_val[ii] / max_stat
            elif self.normalizer[ii] == "std":
                std = self.statistics["stdev"][stat_index]
                norm_min_val[ii] = self.min_val[ii] / std
        # register the normalized min values as a buffer to ensure they are moved to the correct device
        self.register_buffer("norm_min_val", norm_min_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the ReLU activation with the normalized minimum values to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to process.

        Returns
        -------
        torch.Tensor
            The processed tensor with bounding applied.
        """
        x[..., self.data_index] = (
            torch.nn.functional.relu(x[..., self.data_index] - self.norm_min_val) + self.norm_min_val
        )
        return x


class NormalizedLeakyReluBounding(NormalizedReluBounding):
    """Initializes the bounding with a Leaky ReLU activation and customizable normalized thresholds."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = (
            torch.nn.functional.leaky_relu(x[..., self.data_index] - self.norm_min_val) + self.norm_min_val
        )
        return x


class HardtanhBounding(BaseBounding):
    """Initializes the bounding with specified minimum and maximum values for bounding.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    statistics : dict, optional
        A dictionary containing the statistics of the variables.
    name_to_index_stats : dict, optional
        A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        return x


class LeakyHardtanhBounding(HardtanhBounding):
    """Initializes the bounding with a Leaky HardTanh activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.data_index] = leaky_hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)
        return x


class FractionBounding(BaseBounding):
    """Initializes the FractionBounding with specified parameters.

    Parameters
    ----------
    variables : list[str]
        A list of strings representing the variables that will be bounded.
    name_to_index : dict
        A dictionary mapping the variable names to their corresponding indices.
    min_val : float
        The minimum value for the HardTanh activation.
    max_val : float
        The maximum value for the HardTanh activation.
    total_var : str
        A string representing a variable from which a secondary variable is derived. For
        example, in the case of convective precipitation (Cp), total_var = Tp (total precipitation).
    statistics : dict, optional
        A dictionary containing the statistics of the variables.
    name_to_index_stats : dict, optional
        A dictionary mapping the variable names to their corresponding indices in the statistics dictionary.
    """

    def __init__(
        self,
        *,
        variables: list[str],
        name_to_index: dict,
        min_val: float,
        max_val: float,
        total_var: str,
        statistics: Optional[dict] = None,
        name_to_index_stats: Optional[dict] = None,
    ) -> None:
        super().__init__(variables=variables, name_to_index=name_to_index)
        self.min_val = min_val
        self.max_val = max_val
        self.total_variable = self._create_index(variables=[total_var])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the HardTanh bounding  to the data_index variables
        x[..., self.data_index] = torch.nn.functional.hardtanh(
            x[..., self.data_index], min_val=self.min_val, max_val=self.max_val
        )
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x


class LeakyFractionBounding(FractionBounding):
    """Initializes the bounding with a Leaky HardTanh activation and a fraction of the total variable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the LeakyHardTanh bounding  to the data_index variables
        x[..., self.data_index] = leaky_hardtanh(x[..., self.data_index], min_val=self.min_val, max_val=self.max_val)
        # Calculate the fraction of the total variable
        x[..., self.data_index] *= x[..., self.total_variable]
        return x


def _build_dataset_boundings(
    model_config: Any,
    data_indices: Any,
    statistics: dict | None,
) -> nn.ModuleList:
    """Build the list of model-output bounding modules from configuration.

    This is a thin factory over Hydra's ``instantiate`` that reads the iterable
    ``model_config.model.bounding`` and instantiates each entry while injecting
    the common keyword arguments required by bounding modules:
    ``name_to_index``, ``statistics``, and ``name_to_index_stats``. The result
    is returned as an ``nn.ModuleList`` preserving the order of the config.

    Parameters
    ----------
    model_config : Any
        Object with a ``model`` attribute containing an iterable ``bounding``
        (e.g. a list of Hydra configs). If absent or empty, an empty
        ``nn.ModuleList`` is returned.
    data_indices : Any
        Object providing the mappings:
        ``data_indices.model.output.name_to_index`` and
        ``data_indices.data.input.name_to_index``. These are forwarded to each
        instantiated bounding module as ``name_to_index`` and
        ``name_to_index_stats`` respectively.
    statistics : dict | None
        Optional dataset/model statistics passed to each bounding module. Use
        ``None`` if not required by the configured classes.

    Returns
    -------
    torch.nn.ModuleList
        The instantiated bounding modules, in the same order as specified in
        ``model_config.model.bounding``. May be empty.
    """

    bounding_cfgs: Iterable[Any] = getattr(getattr(model_config, "model", object()), "bounding", []) or []

    return nn.ModuleList(
        [
            instantiate(
                cfg,
                name_to_index=data_indices.model.output.name_to_index,
                statistics=statistics,
                name_to_index_stats=data_indices.data.input.name_to_index,
            )
            for cfg in bounding_cfgs
        ]
    )


def build_boundings(
    model_config: Any,
    data_indices: Any,
    statistics: dict | None,
) -> nn.ModuleDict:
    """Build the model-output bounding modules from configuration.

    This is a thin factory that creates a ``nn.ModuleDict`` of bounding
    modules by invoking ``_build_dataset_boundings`` for each dataset
    specified in ``data_indices``.

    Parameters
    ----------
    model_config : Any
        Object with a ``model`` attribute containing an iterable ``bounding``
        (e.g. a list of Hydra configs). If absent or empty, an empty
        ``nn.ModuleDict`` is returned.
    data_indices : Any
        Dictionary mapping dataset names to data indices objects. Each
        data indices object must provide the mappings:
        ``data_indices.model.output.name_to_index`` and
        ``data_indices.data.input.name_to_index``. These are forwarded to each
        instantiated bounding module as ``name_to_index`` and
        ``name_to_index_stats`` respectively.
    statistics : dict | None
        Dictionary mapping dataset names to optional dataset/model statistics
        passed to each bounding module. Use ``None`` if not required by the
        configured classes.

    Returns
    -------
    torch.nn.ModuleDict
        Bounding modules per dataset name, each an ``nn.ModuleList`` in config order.
    """
    bounding_modules = nn.ModuleDict()
    for dataset_name, dataset_indices in data_indices.items():
        bounding_modules[dataset_name] = _build_dataset_boundings(
            model_config, dataset_indices, statistics[dataset_name]
        )
    return bounding_modules


class HydrostaticGeopotential(BaseBounding):
    """Derive geopotential on pressure levels by hydrostatic integration of the predicted column.

    The decoder's ``z_<p>`` heads above the anchor level are overwritten with

        Phi(p_u) = Phi(p_l) + R_d * mean(T_v(p_l), T_v(p_u)) * ln(p_l / p_u),
        T_v = T (1 + 0.608 q),

    integrated upward from the ``z_<levels[0]>`` head (which stays a free prediction), so
    geopotential becomes a differentiable function of the temperature and humidity column and
    is hydrostatically consistent by construction. This is the relation the IFS uses to
    diagnose geopotential from its temperature/humidity column, applied to the model's
    pressure ladder with the trapezoidal layer-mean virtual temperature (exact for a virtual
    temperature linear in ln p).

    The layer reads normalised model outputs, works in float32 physical units (geopotential at
    50 hPa exceeds the fp16 range) and writes back normalised values. Boundings receive only
    dataset statistics, so the per-variable normalisation method must be restated here; the
    training module checks it against the data normaliser at build time.

    Example config (must be the last bounding, after anything else touching z/t/q):

    .. code-block:: yaml

        - _target_: anemoi.models.layers.bounding.HydrostaticGeopotential
          levels: [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50]
          normalizer: {z: min-max, t: mean-std, q: mean-std}
    """

    def __init__(
        self,
        *,
        levels: list[int],
        name_to_index: dict,
        statistics: dict,
        name_to_index_stats: dict,
        normalizer: dict[str, str],
        geopotential_prefix: str = "z",
        temperature_prefix: str = "t",
        humidity_prefix: str = "q",
        geopotential_units: str = "m2/s2",
        check_finite: bool = False,
        variables: Optional[list[str]] = None,  # noqa: ARG002 - derived from levels; accepted for kwarg compatibility
    ) -> None:
        """Initialise the hydrostatic geopotential layer.

        Parameters
        ----------
        levels : list[int]
            Pressure levels in hPa, anchor first and strictly decreasing.
        name_to_index : dict
            Model-output variable name to tensor index (injected).
        statistics : dict
            Dataset statistics with ``mean``, ``stdev``, ``minimum``, ``maximum`` arrays (injected).
        name_to_index_stats : dict
            Data-input variable name to statistics index (injected).
        normalizer : dict[str, str]
            Normalisation method per prefix, e.g. ``{"z": "min-max", "t": "mean-std", "q": "mean-std"}``.
        geopotential_prefix, temperature_prefix, humidity_prefix : str
            Variable-name prefixes of the ladder.
        geopotential_units : {"m2/s2", "m"}
            Units of the ``z`` variables (geopotential or geopotential height).
        check_finite : bool
            Raise if the integrated column is non-finite (adds a device sync; debug only).
        variables : list[str], optional
            Unused; the ladder is derived from ``levels``.
        """
        from anemoi.models.physics.constants import G0
        from anemoi.models.physics.constants import R_D
        from anemoi.models.physics.constants import VIRTUAL_TEMP_COEFF
        from anemoi.models.physics.constants import affine_normaliser

        nn.Module.__init__(self)
        levels = [int(level) for level in levels]
        if len(levels) < 2 or any(a <= b for a, b in zip(levels[:-1], levels[1:])):
            raise ValueError(f"levels must be strictly decreasing (anchor first) with >= 2 entries, got {levels}")
        if geopotential_units not in ("m2/s2", "m"):
            raise ValueError(f"geopotential_units must be 'm2/s2' or 'm', got {geopotential_units!r}")
        prefixes = {"z": geopotential_prefix, "t": temperature_prefix, "q": humidity_prefix}
        missing_methods = [p for p in prefixes.values() if p not in normalizer]
        if missing_methods:
            raise ValueError(
                f"normalizer must give a method for each of {list(prefixes.values())}; missing {missing_methods}"
            )

        self.name_to_index = name_to_index
        self.statistics = statistics
        self.name_to_index_stats = name_to_index_stats
        self.levels = levels
        self.geopotential_units = geopotential_units
        self.check_finite = check_finite
        self.virtual_temp_coeff = VIRTUAL_TEMP_COEFF
        self.g0 = G0

        names = {key: [f"{prefix}_{level}" for level in levels] for key, prefix in prefixes.items()}
        self.variables = names["z"][1:]  # the overwritten (derived) heads
        missing = [n for group in names.values() for n in group if n not in name_to_index]
        if missing:
            raise ValueError(f"HydrostaticGeopotential: variables missing from the model output: {missing}")
        missing = [n for group in names.values() for n in group if n not in name_to_index_stats]
        if missing:
            raise ValueError(f"HydrostaticGeopotential: variables missing from the statistics index: {missing}")
        # Method restatement, checked against the data normaliser by the training module.
        self.normalizer_methods = {n: normalizer[prefixes[key]] for key, group in names.items() for n in group}

        stats = {k: statistics[k] for k in ("mean", "stdev")}
        stats["minimum"] = statistics["minimum"] if "minimum" in statistics else statistics["min"]
        stats["maximum"] = statistics["maximum"] if "maximum" in statistics else statistics["max"]
        for key, group in names.items():
            # Explicit ladder order: BaseBounding._create_index would follow name_to_index order.
            self.register_buffer(
                f"{key}_index", torch.tensor([name_to_index[n] for n in group], dtype=torch.long), persistent=True
            )
            mul, add = [], []
            for n in group:
                i = name_to_index_stats[n]
                m, a = affine_normaliser(
                    normalizer[prefixes[key]],
                    mean=float(stats["mean"][i]),
                    stdev=float(stats["stdev"][i]),
                    minimum=float(stats["minimum"][i]),
                    maximum=float(stats["maximum"][i]),
                )
                mul.append(m)
                add.append(a)
            mul_t = torch.tensor(mul, dtype=torch.float32)
            if not torch.isfinite(mul_t).all() or (mul_t == 0).any():
                raise ValueError(f"HydrostaticGeopotential: degenerate statistics for {group}")
            self.register_buffer(f"{key}_mul", mul_t, persistent=True)
            self.register_buffer(f"{key}_add", torch.tensor(add, dtype=torch.float32), persistent=True)
        p = torch.tensor(levels, dtype=torch.float64)
        self.register_buffer("layer_factor", (R_D * torch.log(p[:-1] / p[1:])).to(torch.float32), persistent=True)
        self.data_index = self.z_index[1:]

    def _physical(self, x: torch.Tensor, key: str) -> torch.Tensor:
        index = getattr(self, f"{key}_index")
        mul = getattr(self, f"{key}_mul")
        # Buffers are float32 in training (physical geopotential overflows fp16); float64 under gradcheck.
        return (x.index_select(-1, index).to(mul.dtype) - getattr(self, f"{key}_add")) / mul

    def integrate(self, x: torch.Tensor) -> torch.Tensor:
        """Return the hydrostatically integrated geopotential (m^2 s^-2) at every ladder level, ``(..., L)``."""
        t = self._physical(x, "t")
        q = self._physical(x, "q").clamp_min(0.0)
        tv = t * (1.0 + self.virtual_temp_coeff * q)
        phi0 = self._physical(x, "z")[..., :1]
        if self.geopotential_units == "m":
            phi0 = phi0 * self.g0
        dphi = 0.5 * (tv[..., :-1] + tv[..., 1:]) * self.layer_factor
        return torch.cat([phi0, phi0 + torch.cumsum(dphi, dim=-1)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            phi = self.integrate(x)[..., 1:]
            z_phys = phi / self.g0 if self.geopotential_units == "m" else phi
            # Renormalise BEFORE casting back: physical geopotential overflows fp16, O(1) values do not.
            z_norm = z_phys * self.z_mul[1:] + self.z_add[1:]
            if self.check_finite and not torch.isfinite(z_norm).all():
                raise RuntimeError("HydrostaticGeopotential: non-finite integrated column; check anchor and t/q.")
        x[..., self.data_index] = z_norm.to(x.dtype)
        return x
