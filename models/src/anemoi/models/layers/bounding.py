# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import math
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Iterable
from typing import Optional

import torch
from hydra.utils import get_class
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

    def decoder_forcing_names() -> list[str]:
        # Decoder-forcing channels in the order the task slices them (sorted data-input index).
        input_index_to_name = {int(i): name for name, i in data_indices.data.input.name_to_index.items()}
        decoder_forcing_idx = getattr(data_indices.data.input, "decoder_forcing", None)
        return [] if decoder_forcing_idx is None else [input_index_to_name[int(i)] for i in decoder_forcing_idx]

    modules = []
    for cfg in bounding_cfgs:
        target = cfg.get("_target_") if hasattr(cfg, "get") else None
        extra = {}
        if target is not None and getattr(get_class(target), "accepts_decoder_forcing", False):
            extra["decoder_forcing_names"] = decoder_forcing_names()
        modules.append(
            instantiate(
                cfg,
                name_to_index=data_indices.model.output.name_to_index,
                statistics=statistics,
                name_to_index_stats=data_indices.data.input.name_to_index,
                **extra,
            )
        )
    return nn.ModuleList(modules)


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

    With ``msl_anchor`` set, the anchor is instead diagnosed from the mean-sea-level pressure
    head by the same relation over the layer from sea level (Phi = 0) to ``levels[0]``,

        Phi(p_0) = R_d * T_v(p_0) * ln(msl / p_0),

    and ``z_<levels[0]>`` is overwritten too. msl is far more densely observed (SYNOP, ships,
    buoys) than radiosonde ``z_<levels[0]>``, and the msl and z heads become mutually consistent.
    The layer is taken isothermal at ``T_v(p_0)``; a standard lapse rate would shift the
    ~100 m layer by well under 1 m.

    With ``sp_anchor`` set, the anchor is the surface instead: geopotential equals the surface
    geopotential ``Phi_s`` (from the ``orography`` decoder forcing, read at the target time) at
    the predicted surface pressure. Per node, with ``k`` the first ladder level at or above the
    ground (``p_k <= sp``),

        Phi(p_k) = Phi_s + R_d * mean(T_v,s, T_v(p_k)) * ln(sp / p_k),

    where ``T_v,s`` extrapolates ``T_v`` linearly in ``ln p`` from levels ``k`` and ``k+1`` down
    to ``sp``. The column is then integrated up and down from ``k``, so below-ground levels are
    extrapolated with the model's own (unobserved) temperatures. Every ``z`` head is overwritten.
    On IFS analyses this reconstructs z_500 to 7.5 gpm RMSE globally and ~21 gpm over the
    Antarctic plateau, against 16/70 for the msl anchor and 20/124 for a z_1000 anchor, which
    integrate through below-ground layers. Requires the model to be called with
    ``decoder_forcings`` containing the orography variable.

    The layer reads normalised model outputs, works in float32 physical units (geopotential at
    50 hPa exceeds the fp16 range) and writes back normalised values. Boundings receive only
    dataset statistics, so the per-variable normalisation method must be restated here; the
    training module checks it against the data normaliser at build time.

    Example config (must be the last bounding, after anything else touching z/t/q):

    .. code-block:: yaml

        - _target_: anemoi.models.layers.bounding.HydrostaticGeopotential
          levels: [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50]
          normalizer: {z: min-max, t: mean-std, q: mean-std}
          # optional: anchor on msl instead of z_1000 (the normaliser must then name msl too)
          # msl_anchor: msl
          # normalizer: {z: min-max, t: mean-std, q: mean-std, msl: mean-std}
          # optional: anchor on surface pressure + orography (z must be a data.decoder_forcing)
          # sp_anchor: sp
          # orography: z
          # orography_normalizer: max
          # normalizer: {z: mean-std, t: mean-std, q: mean-std, sp: mean-std}
    """

    # The model builder injects ``decoder_forcing_names`` into classes that set this.
    accepts_decoder_forcing = True

    # Bounds on the extrapolated surface virtual temperature; they only bite on a wild column.
    SURFACE_TV_RANGE = (150.0, 350.0)

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
        msl_anchor: Optional[str] = None,
        msl_units: str = "Pa",
        sp_anchor: Optional[str] = None,
        sp_units: str = "Pa",
        orography: Optional[str] = None,
        orography_normalizer: Optional[str] = None,
        orography_units: str = "m",
        decoder_forcing_names: Optional[list[str]] = None,
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
        msl_anchor : str, optional
            Name of a mean-sea-level pressure output variable. If given, the anchor geopotential
            is diagnosed from it and ``z_<levels[0]>`` is overwritten as well; ``normalizer`` must
            then give a method for this name.
        msl_units : {"Pa", "hPa"}
            Units of the ``msl_anchor`` variable.
        sp_anchor : str, optional
            Name of a surface-pressure output variable. If given, the column is anchored on the
            surface (see class docstring) and every ``z`` head is overwritten; ``normalizer`` must
            then give a method for this name. Exclusive with ``msl_anchor``.
        sp_units : {"Pa", "hPa"}
            Units of the ``sp_anchor`` variable.
        orography : str, optional
            Decoder-forcing variable holding the surface height the ``sp`` targets refer to.
            Required with ``sp_anchor``.
        orography_normalizer : str, optional
            Normalisation method of ``orography`` (it can share a name with the ``z`` prefix, so it
            is not read from ``normalizer``). Required with ``sp_anchor``.
        orography_units : {"m", "m2/s2"}
            Units of ``orography``.
        decoder_forcing_names : list[str], optional
            Decoder-forcing channel names in tensor order (injected by the model builder).
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
        if msl_units not in ("Pa", "hPa"):
            raise ValueError(f"msl_units must be 'Pa' or 'hPa', got {msl_units!r}")
        if sp_units not in ("Pa", "hPa"):
            raise ValueError(f"sp_units must be 'Pa' or 'hPa', got {sp_units!r}")
        if orography_units not in ("m", "m2/s2"):
            raise ValueError(f"orography_units must be 'm' or 'm2/s2', got {orography_units!r}")
        if msl_anchor is not None and sp_anchor is not None:
            raise ValueError("msl_anchor and sp_anchor are exclusive; give at most one")
        if sp_anchor is not None and (orography is None or orography_normalizer is None):
            raise ValueError("sp_anchor needs orography and orography_normalizer")
        prefixes = {"z": geopotential_prefix, "t": temperature_prefix, "q": humidity_prefix}
        if msl_anchor is not None:
            prefixes["msl"] = msl_anchor
        if sp_anchor is not None:
            prefixes["sp"] = sp_anchor
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
        self.r_d = R_D
        self.msl_anchor = msl_anchor
        # msl in Pa over the anchor level in Pa.
        self.msl_scale = 1.0 if msl_units == "Pa" else 100.0
        self.anchor_pressure = 100.0 * levels[0]
        self.sp_anchor = sp_anchor
        self.sp_scale = 1.0 if sp_units == "Pa" else 100.0
        # First overwritten ladder level: the anchor stays free unless it is diagnosed from msl or sp.
        self.first_derived = 0 if (msl_anchor is not None or sp_anchor is not None) else 1

        names = {
            key: [f"{prefix}_{level}" for level in levels]
            for key, prefix in prefixes.items()
            if key not in ("msl", "sp")
        }
        if msl_anchor is not None:
            names["msl"] = [msl_anchor]
        if sp_anchor is not None:
            names["sp"] = [sp_anchor]
        self.variables = names["z"][self.first_derived :]  # the overwritten (derived) heads
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
        self.data_index = self.z_index[self.first_derived :]

        self.orography = orography if sp_anchor is not None else None
        if self.orography is not None:
            decoder_forcing_names = list(decoder_forcing_names or [])
            if orography not in decoder_forcing_names:
                raise ValueError(
                    f"HydrostaticGeopotential: orography {orography!r} must be a decoder forcing "
                    f"(data.datasets.<name>.decoder_forcing); decoder forcings are {decoder_forcing_names}"
                )
            if orography not in name_to_index_stats:
                raise ValueError(f"HydrostaticGeopotential: orography {orography!r} missing from the statistics index")
            i = name_to_index_stats[orography]
            m, a = affine_normaliser(
                orography_normalizer,
                mean=float(stats["mean"][i]),
                stdev=float(stats["stdev"][i]),
                minimum=float(stats["minimum"][i]),
                maximum=float(stats["maximum"][i]),
            )
            if not math.isfinite(m) or m == 0:
                raise ValueError(f"HydrostaticGeopotential: degenerate statistics for orography {orography!r}")
            self.orography_position = decoder_forcing_names.index(orography)
            self.orography_scale = G0 if orography_units == "m" else 1.0
            self.register_buffer("orography_mul", torch.tensor(m, dtype=torch.float32), persistent=True)
            self.register_buffer("orography_add", torch.tensor(a, dtype=torch.float32), persistent=True)
            self.normalizer_methods[orography] = orography_normalizer
            self.register_buffer("log_pressure", torch.log(100.0 * p).to(torch.float32), persistent=True)
            self.register_buffer("level_pressure", (100.0 * p).to(torch.float32), persistent=True)

    @property
    def requires_decoder_forcing(self) -> bool:
        """Whether ``forward`` needs the target-time decoder forcings (sp anchor only)."""
        return self.orography is not None

    def _physical(self, x: torch.Tensor, key: str) -> torch.Tensor:
        index = getattr(self, f"{key}_index")
        mul = getattr(self, f"{key}_mul")
        # Buffers are float32 in training (physical geopotential overflows fp16); float64 under gradcheck.
        return (x.index_select(-1, index).to(mul.dtype) - getattr(self, f"{key}_add")) / mul

    def _surface_anchored(
        self, x: torch.Tensor, tv: torch.Tensor, cum: torch.Tensor, decoder_forcing: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Column anchored on Phi_s at the predicted surface pressure, ``(..., L)``."""
        if decoder_forcing is None:
            raise RuntimeError(
                "HydrostaticGeopotential with sp_anchor needs the decoder forcings; "
                "call the model with decoder_forcings (only AnemoiModelEncProcDec passes them on)."
            )
        orog = decoder_forcing[..., self.orography_position : self.orography_position + 1].to(self.orography_mul.dtype)
        phi_s = (orog - self.orography_add) / self.orography_mul * self.orography_scale
        # Clamp only guards the log against a wildly unphysical untrained head.
        sp = (self._physical(x, "sp") * self.sp_scale).clamp_min(1.0)
        n_levels = tv.shape[-1]
        above = self.level_pressure <= sp
        # First level at or above the ground; with none (sp above the top level) use the top level.
        k = torch.where(above.any(-1, keepdim=True), above.to(torch.int8).argmax(-1, keepdim=True), n_levels - 1)
        k_up = (k + 1).clamp_max(n_levels - 1)
        tv_k = tv.gather(-1, k)
        lnp_k = self.log_pressure[k]
        dlnp = self.log_pressure[k_up] - lnp_k
        has_up = k_up != k
        slope = torch.where(has_up, (tv.gather(-1, k_up) - tv_k) / torch.where(has_up, dlnp, -1.0), 0.0)
        log_ratio = torch.log(sp) - lnp_k  # ln(sp / p_k), >= 0 unless no level is above ground
        tv_s = (tv_k + slope * log_ratio).clamp(*self.SURFACE_TV_RANGE)
        phi_k = phi_s + self.r_d * 0.5 * (tv_s + tv_k) * log_ratio
        return phi_k + cum - cum.gather(-1, k)

    def integrate(self, x: torch.Tensor, decoder_forcing: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the hydrostatically integrated geopotential (m^2 s^-2) at every ladder level, ``(..., L)``."""
        t = self._physical(x, "t")
        q = self._physical(x, "q").clamp_min(0.0)
        tv = t * (1.0 + self.virtual_temp_coeff * q)
        if self.sp_anchor is not None:
            dphi = 0.5 * (tv[..., :-1] + tv[..., 1:]) * self.layer_factor
            cum = torch.cat([torch.zeros_like(tv[..., :1]), torch.cumsum(dphi, dim=-1)], dim=-1)
            return self._surface_anchored(x, tv, cum, decoder_forcing)
        if self.msl_anchor is not None:
            # Clamp only guards the log against a wildly unphysical untrained head.
            msl = (self._physical(x, "msl") * self.msl_scale).clamp_min(1.0)
            phi0 = self.r_d * tv[..., :1] * torch.log(msl / self.anchor_pressure)
        else:
            phi0 = self._physical(x, "z")[..., :1]
            if self.geopotential_units == "m":
                phi0 = phi0 * self.g0
        dphi = 0.5 * (tv[..., :-1] + tv[..., 1:]) * self.layer_factor
        return torch.cat([phi0, phi0 + torch.cumsum(dphi, dim=-1)], dim=-1)

    def forward(self, x: torch.Tensor, decoder_forcing: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            phi = self.integrate(x, decoder_forcing)[..., self.first_derived :]
            z_phys = phi / self.g0 if self.geopotential_units == "m" else phi
            # Renormalise BEFORE casting back: physical geopotential overflows fp16, O(1) values do not.
            z_norm = z_phys * self.z_mul[self.first_derived :] + self.z_add[self.first_derived :]
            if self.check_finite and not torch.isfinite(z_norm).all():
                raise RuntimeError("HydrostaticGeopotential: non-finite integrated column; check anchor and t/q.")
        x[..., self.data_index] = z_norm.to(x.dtype)
        return x
