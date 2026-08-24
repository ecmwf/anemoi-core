# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``skip_input`` redirects the additive skip connection without touching the encoder.

Building a full ``AnemoiModelEncProcDec`` is heavy, so these call the unbound
``_assemble_input`` against a stub carrying only the attributes it touches.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

# Imported for their side effect: __subclasses__() only sees classes whose defining
# module has been imported, and test_forward_overrides_must_opt_out_of_skip_input
# walks the live tree.
import anemoi.models.models.autoencoder  # noqa: F401
import anemoi.models.models.ens_encoder_processor_decoder  # noqa: F401
import anemoi.models.models.hierarchical  # noqa: F401
import anemoi.models.models.hierarchical_autoencoder  # noqa: F401
import anemoi.models.models.transport_encoder_processor_decoder  # noqa: F401
from anemoi.models.layers.residual import ClimatologySkipConnection
from anemoi.models.layers.residual import SkipConnection
from anemoi.models.models.base import BaseGraphModel
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec

_BATCH, _TIME, _ENSEMBLE, _GRID, _VARS = 2, 3, 1, 5, 4
_STEP_OUTPUT = 2


def _make_stub() -> SimpleNamespace:
    """A stand-in exposing only what ``_assemble_input`` reads."""
    return SimpleNamespace(
        residual={"data": SkipConnection()},
        node_attributes=lambda _name, batch_size: torch.zeros(batch_size * _ENSEMBLE * _GRID, 2),
        n_step_output=_STEP_OUTPUT,
    )


def _assemble(x: torch.Tensor, skip_input: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    x_data_latent, x_skip, _ = AnemoiModelEncProcDec._assemble_input(
        _make_stub(),
        x,
        batch_size=_BATCH,
        grid_shard_sizes=None,
        dataset_name="data",
        skip_input=skip_input,
    )
    return x_data_latent, x_skip


def _expected_skip(source: torch.Tensor) -> torch.Tensor:
    """The last input step, broadcast over the output-time dim as _expand_time does."""
    return source[:, -1, ...].unsqueeze(1).expand(-1, _STEP_OUTPUT, -1, -1, -1)


def test_skip_input_none_reads_x() -> None:
    x = torch.randn(_BATCH, _TIME, _ENSEMBLE, _GRID, _VARS)

    _, x_skip = _assemble(x, None)

    # 5-D (batch, n_step_output, ensemble, grid, vars), as _assemble_output requires.
    assert x_skip.shape == (_BATCH, _STEP_OUTPUT, _ENSEMBLE, _GRID, _VARS)
    assert torch.equal(x_skip, _expected_skip(x))


def test_skip_input_redirects_residual_but_not_encoder() -> None:
    x = torch.randn(_BATCH, _TIME, _ENSEMBLE, _GRID, _VARS)
    background = torch.randn(_BATCH, _TIME, _ENSEMBLE, _GRID, _VARS)

    latent_default, skip_default = _assemble(x, None)
    latent_override, skip_override = _assemble(x, background)

    # The residual now reads the background instead of x ...
    assert torch.equal(skip_override, _expected_skip(background))
    assert not torch.equal(skip_override, skip_default)
    # ... while the encoder still sees exactly the same features.
    assert torch.equal(latent_override, latent_default)


def test_zero_skip_base_yields_pure_climatology(tmp_path) -> None:
    """The step-0 contract: an all-zero base means "no background", not a mosaic.

    ``DAForecaster.build_skip_input`` returns zeros at step 0, relying on the
    ``missing_value: 0.0`` sentinel that ``InputOnlyImputer`` already writes for
    absent observations. This pins that coupling.
    """
    npz_path = tmp_path / "clim.npz"
    np.savez(npz_path, var0=np.full(4, 7.0, dtype=np.float32))

    data_indices = MagicMock()
    data_indices.model.input.name_to_index = {"var0": 0}
    residual = ClimatologySkipConnection(
        climatology_path=str(npz_path),
        data_indices=data_indices,
        missing_value=0.0,
        fill_missing_only=True,
    )

    # A sparse observation mosaic: observed at grid 0 and 2, "missing" (0.0) elsewhere.
    observations = torch.zeros(1, 2, 1, 4, 1)
    observations[0, -1, 0, 0, 0] = 5.0
    observations[0, -1, 0, 2, 0] = 6.0

    mosaic = residual(observations)
    # Today's step-0 behaviour: observations punch through into the residual.
    assert torch.allclose(mosaic[0, 0, :, 0], torch.tensor([5.0, 7.0, 6.0, 7.0]))

    # With a zero base the observations are gone and climatology covers the field.
    pure = residual(torch.zeros_like(observations))
    assert torch.all(pure == 7.0)


def _subclasses(cls: type) -> Iterator[type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def test_base_model_does_not_advertise_skip_input() -> None:
    # forward() swallows unknown kwargs, so callers rely on this flag rather than
    # passing the tensor and hoping it is honoured.
    assert BaseGraphModel.supports_skip_input is False
    assert AnemoiModelEncProcDec.supports_skip_input is True


def test_forward_overrides_do_not_advertise_skip_input() -> None:
    """No shipped subclass that replaces forward claims to support skip_input.

    Checked against the live subclass tree rather than a hand-written list, so a new
    subclass is covered the moment it is defined.
    """
    offenders = [
        cls.__name__
        for cls in _subclasses(AnemoiModelEncProcDec)
        if cls.forward is not AnemoiModelEncProcDec.forward and cls.supports_skip_input
    ]
    assert not offenders, (
        f"{offenders} override forward without threading skip_input to _assemble_input, "
        f"yet advertise support for it."
    )


def test_overriding_forward_opts_out_automatically() -> None:
    # The opt-out is structural: a subclass author cannot forget it.
    class Overrides(AnemoiModelEncProcDec):
        def forward(self, x, **kwargs):  # noqa: ARG002
            return x

    assert Overrides.supports_skip_input is False


def test_subclass_inheriting_forward_keeps_support() -> None:
    class InheritsForward(AnemoiModelEncProcDec):
        pass

    assert InheritsForward.supports_skip_input is True


def test_subclass_may_opt_back_in_explicitly() -> None:
    # A subclass that does thread skip_input through its own forward says so.
    class ThreadsItProperly(AnemoiModelEncProcDec):
        supports_skip_input = True

        def forward(self, x, **kwargs):  # noqa: ARG002
            return x

    assert ThreadsItProperly.supports_skip_input is True
