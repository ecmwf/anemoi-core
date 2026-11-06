# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Energy score of the local structure at wavelet scales, for ensemble training."""

from typing import Literal

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint

from anemoi.models.distributed.graph import all_to_all_transpose
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.spectral_transforms import octahedral_lons_per_lat
from anemoi.models.layers.spectral_transforms import reduced_gaussian_lons_per_lat
from anemoi.models.layers.spherical_wavelets import RieszStructureTensor
from anemoi.models.layers.spherical_wavelets import scale_degree_ranges
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.enums import TensorDim

# Added under square roots, so that their slopes stay finite where the value is exactly zero.
_TINY = 1e-12


def _grid_lons_per_lat(transform: str, grid: str | None, nlat: int | None, nlon: int | None) -> list[int]:
    """Number of points on each latitude ring of the data grid, from pole to pole."""
    if transform == "reduced_sht":
        if grid is None:
            msg = "The 'reduced_sht' transform needs the name of the grid, e.g. grid: n320."
            raise ValueError(msg)
        return reduced_gaussian_lons_per_lat(grid)
    if nlat is None:
        msg = f"The '{transform}' transform needs the number of latitudes, nlat."
        raise ValueError(msg)
    if transform == "octahedral_sht":
        return octahedral_lons_per_lat(nlat)
    if transform == "regular_sht":
        return [2 * nlat if nlon is None else nlon] * nlat
    msg = f"Unknown transform '{transform}'; use 'reduced_sht', 'octahedral_sht' or 'regular_sht'."
    raise ValueError(msg)


class WaveletEnergyScoreLoss(BaseLoss):
    """Energy score of the local amount and direction of structure at wavelet scales.

    At each chosen scale and grid point, ``RieszStructureTensor`` gives J_ee, J_nn and J_en: the structure
    tensor of the scale's Riesz pair, smoothed over a few wavelengths. From them

        v = ( sqrt(E), (J_ee - J_nn) / sqrt(E + floor), 2 J_en / sqrt(E + floor) ),   E = J_ee + J_nn.

    - sqrt(E) is the amount of structure at that scale around the point.
    - The other two entries point along twice the angle of the prevailing direction across the crests. Their
      length is sqrt(E) times how strongly the crests line up (0 to 1).
    - The floor is ``direction_floor`` times the field's area-weighted mean E, so very weak structure has
      little say in the direction. Each member and the target use their own floor. It is the same function of
      the field for all of them, so the score stays proper.

    The loss is the energy score of v over the members at each point, added over the scales with their
    weights, then scaled and reduced like the CRPS. It rewards the right amount of structure at each scale,
    lined up the right way, and tolerates small shifts in the position of each crest. The transforms need the
    whole grid: on a sharded grid the fields are spread over the variables for the transforms and back
    afterwards.

    The finest scale lies mostly beyond the truncation; give it weight 0.

    For best results, train with ``training.float32_matmul_precision: highest``.

    - With TF32 matrix products (the default "high") the transforms keep only about three digits. On real fields
      at O640 the gradient is then about 1 % off, and 13 % for precipitation.
    - Full float32 keeps it within about 1e-5 for most fields. For smooth fields such as z500 the finest scales
      come close to what float32 itself resolves (0.5 %).
    """

    def __init__(
        self,
        transform: Literal["reduced_sht", "octahedral_sht", "regular_sht"],
        scale_weights: list[float],
        *,
        truncation: int | None = None,
        smoothing_octaves: int = 3,
        direction_floor: float = 0.01,
        fair: bool = True,
        grid: Literal["n320", "N320"] | None = None,
        nlat: int | None = None,
        nlon: int | None = None,
        checkpoint_scales: bool = False,
        variables_per_checkpoint: int | None = None,
        ignore_nans: bool = False,
        no_autocast: bool = True,
    ) -> None:
        """Initialise the wavelet energy score.

        Parameters
        ----------
        transform : {"reduced_sht", "octahedral_sht", "regular_sht"}
            Layout of the data grid. All three use Gaussian latitudes, ordered ring by ring from north to south.
        scale_weights : list[float]
            Weight of each wavelet scale, from the largest; scale j covers 2^(j - 1) < l < 2^(j + 1). Scales with
            weight 0 are not computed.
        truncation : int, optional
            Highest total wavenumber. Defaults to nlat // 2 - 1.
        smoothing_octaves : int
            How many octaves below its own scale each structure tensor is smoothed. Default 3.
        direction_floor : float
            Floor under E in the direction entries, relative to the field's mean E. Default 0.01.
        fair : bool
            Use the fair (unbiased) energy score. Default True.
        grid : str, optional
            Name of the reduced Gaussian grid, for ``reduced_sht`` (only "n320").
        nlat : int, optional
            Number of latitudes, for ``octahedral_sht`` and ``regular_sht``.
        nlon : int, optional
            Number of longitudes, for ``regular_sht``. Defaults to 2 * nlat.
        checkpoint_scales : bool
            Recompute each scale in the backward pass, keeping the values of one scale in memory at a time. Lowers
            the peak memory and takes longer. Default False.
        variables_per_checkpoint : int, optional
            Split the variables into groups of this size and recompute each group of each scale in the backward
            pass. This includes what ``checkpoint_scales`` does and lowers the peak memory further. Each group adds
            a fixed time, so very small groups are slow with many variables. Default None: no groups.
        ignore_nans : bool
            Must be False: the transforms cannot handle missing values.
        no_autocast : bool
            Must be True: the transforms and the score always run in float32.
        """
        if ignore_nans:
            msg = "WaveletEnergyScoreLoss cannot handle missing values; ignore_nans must be False."
            raise ValueError(msg)
        if not no_autocast:
            msg = "WaveletEnergyScoreLoss always runs in float32; no_autocast must be True."
            raise ValueError(msg)
        if direction_floor <= 0.0:
            msg = f"direction_floor must be positive, got {direction_floor}."
            raise ValueError(msg)
        if variables_per_checkpoint is not None and variables_per_checkpoint < 1:
            msg = f"variables_per_checkpoint must be at least 1, got {variables_per_checkpoint}."
            raise ValueError(msg)
        super().__init__(ignore_nans=ignore_nans)

        lons_per_lat = _grid_lons_per_lat(transform, grid, nlat, nlon)
        truncation = len(lons_per_lat) // 2 - 1 if truncation is None else truncation
        ranges = scale_degree_ranges(truncation)
        if len(scale_weights) != len(ranges):
            msg = (
                f"scale_weights needs one weight per wavelet scale: {len(ranges)} scales at T{truncation}, "
                f"covering degrees {ranges} from the largest; got {len(scale_weights)}."
            )
            raise ValueError(msg)
        if any(weight < 0.0 for weight in scale_weights) or not any(weight > 0.0 for weight in scale_weights):
            msg = f"scale_weights must be at least 0, with at least one above 0; got {scale_weights}."
            raise ValueError(msg)

        scales = [j for j, weight in enumerate(scale_weights) if weight > 0.0]
        self.structure = RieszStructureTensor(lons_per_lat, truncation, scales, smoothing_octaves=smoothing_octaves)
        self.weights = [float(scale_weights[j]) for j in scales]
        self.direction_floor = direction_floor
        self.checkpoint_scales = checkpoint_scales
        self.variables_per_checkpoint = variables_per_checkpoint
        self.fair = fair

    @property
    def needs_shard_layout_info(self) -> bool:
        return True

    @property
    def name(self) -> str:
        prefix = "f" if self.fair else ""
        return f"{prefix}wavelet_energy_score"

    def structure_vectors(self, tensor: torch.Tensor) -> torch.Tensor:
        """Vectors v ``[..., 3, grid]`` from J_ee, J_nn and J_en ``[..., 3, grid]``, see the class description."""
        j_ee, j_nn, j_en = tensor.unbind(-2)
        energy = (j_ee + j_nn).clamp_min(0.0)
        floor = self.direction_floor * (energy * self.structure.area.to(energy.dtype)).sum(-1, keepdim=True)
        direction_scale = (energy + floor + _TINY).rsqrt()
        return torch.stack(
            [(energy + _TINY).sqrt(), (j_ee - j_nn) * direction_scale, 2.0 * j_en * direction_scale],
            dim=-2,
        )

    def energy_score(self, members: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Energy score ``[..., grid]`` of member vectors ``[..., e, 3, grid]`` against ``[..., 1, 3, grid]``."""

        def length(x: torch.Tensor) -> torch.Tensor:
            return (x.square().sum(-2) + _TINY).sqrt()

        ensemble_size = members.shape[-3]
        skill = length(members - target).mean(-2)
        spread = torch.zeros_like(skill)
        for member in range(ensemble_size - 1):
            spread = spread + length(members[..., member : member + 1, :, :] - members[..., member + 1 :, :, :]).sum(-2)
        pair_coefficient = 1.0 / (ensemble_size * (ensemble_size - 1)) if self.fair else 1.0 / ensemble_size**2
        return skill - pair_coefficient * spread

    def compile_for_training(self, **options) -> None:
        """Compile the grid-point work of the loss; the spectral transforms run as they are."""
        self._products = torch.compile(self._products, **options)
        self._score_from_smoothed = torch.compile(self._score_from_smoothed, **options)

    def _products(self, pair: torch.Tensor, east: torch.Tensor, north: torch.Tensor) -> torch.Tensor:
        return self.structure.products(pair, east, north)

    def _score_from_smoothed(
        self,
        smoothed: torch.Tensor,
        target_vectors: torch.Tensor,
        east: torch.Tensor,
        north: torch.Tensor,
    ) -> torch.Tensor:
        vectors = self.structure_vectors(self.structure.read_tensor(smoothed, east, north))
        return self.energy_score(vectors, target_vectors)

    def _score_of_scale(self, coeffs: torch.Tensor, target_vectors: torch.Tensor, k: int) -> torch.Tensor:
        """Energy scores ``[..., grid]`` at the k-th chosen scale, from the members' coefficients."""
        dtype = coeffs.real.dtype
        east, north = self.structure.east.to(dtype), self.structure.north.to(dtype)
        products = self._products(self.structure.riesz_pair(coeffs, k), east, north)
        return self._score_from_smoothed(self.structure.smooth(products, k), target_vectors, east, north)

    def _checkpointed_score_of_scale(self, coeffs: torch.Tensor, target_vectors: torch.Tensor, k: int) -> torch.Tensor:
        return checkpoint(
            self._score_of_scale,
            coeffs,
            target_vectors,
            k,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def _arranged_score_of_scale(self, coeffs: torch.Tensor, target_vectors: torch.Tensor, k: int) -> torch.Tensor:
        """Energy scores ``[b, t, v, grid]`` at the k-th chosen scale, in checkpoints if the loss is set to use them."""
        size = self.variables_per_checkpoint
        if size is not None:
            groups = zip(coeffs.split(size, dim=2), target_vectors.split(size, dim=2), strict=True)
            return torch.cat([self._checkpointed_score_of_scale(c, t, k) for c, t in groups], dim=2)
        if self.checkpoint_scales:
            return self._checkpointed_score_of_scale(coeffs, target_vectors, k)
        return self._score_of_scale(coeffs, target_vectors, k)

    def _target_vectors(self, target: torch.Tensor) -> list[torch.Tensor]:
        """Vectors of the target ``[b, t, v, 1, 3, grid]``, one tensor per chosen scale."""
        coeffs = self.structure.coefficients(target.float())
        return [self.structure_vectors(self.structure.tensor(coeffs, k)) for k in range(len(self.weights))]

    def point_scores(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """Weighted wavelet energy scores at each grid point, ``[b, t, 1, grid, v]``, on the full grid."""
        if self.fair and y_pred.shape[2] < 2:
            msg = "The fair WaveletEnergyScoreLoss needs at least two ensemble members."
            raise ValueError(msg)
        if y_pred.shape[-1] == 0:
            # On a sharded grid a rank can end up with no variables. It still has to take part in the exchange back
            # and in the backward pass, so it returns an empty score that is a view of the prediction.
            return y_pred[:, :, :1]
        pred = einops.rearrange(y_pred, "b t e g v -> b t v e g")
        target = einops.rearrange(y_target, "b t 1 g v -> b t v 1 g")
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            # The target goes through the transforms on its own, hence nothing about it is kept for the backward pass.
            target_vectors = self._target_vectors(target)
            coeffs = self.structure.coefficients(pred.float())
            scores = 0.0
            for k, weight in enumerate(self.weights):
                scores = scores + weight * self._arranged_score_of_scale(coeffs, target_vectors[k], k)
        return einops.rearrange(scores, "b t v g -> b t 1 g v")

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        grid_shard_sizes: ShardSizes = None,
        grid_dim: int | None = None,
        squash_mode: Squash_mode = "avg",
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None
        if is_sharded:
            if grid_shard_sizes is None or group is None:
                msg = "WaveletEnergyScoreLoss on a sharded grid needs grid_shard_sizes and the process group."
                raise ValueError(msg)
            # Spread the fields over the variables, so each rank holds whole fields.
            grid_dim = TensorDim.GRID if grid_dim is None else grid_dim
            variable_shard_sizes = get_shard_sizes(y_pred, TensorDim.VARIABLE, group)
            y_pred, y_target = (
                all_to_all_transpose(x, TensorDim.VARIABLE, variable_shard_sizes, grid_dim, grid_shard_sizes, group)
                for x in (y_pred, y_target)
            )

        n_grid_points = self.structure.sht.n_grid_points
        if y_pred.shape[TensorDim.GRID] != n_grid_points:
            msg = f"WaveletEnergyScoreLoss expects {n_grid_points} grid points, got {y_pred.shape[TensorDim.GRID]}."
            raise ValueError(msg)
        scores = self.point_scores(y_pred, y_target)

        if is_sharded:
            # Back to the grid layout, so the node weights of this rank's points apply.
            scores = all_to_all_transpose(
                scores,
                grid_dim,
                grid_shard_sizes,
                TensorDim.VARIABLE,
                variable_shard_sizes,
                group,
            )

        scores = self.scale(scores, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(scores, squash=squash, squash_mode=squash_mode, group=group if is_sharded else None)
