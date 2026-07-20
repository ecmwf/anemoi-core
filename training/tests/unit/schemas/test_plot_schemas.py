# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from anemoi.training.diagnostics.callbacks.plot import PlottingSettings
from anemoi.training.schemas.diagnostics import BatchOutputPlotFnSchema
from anemoi.training.schemas.diagnostics import GraphPlotFnSchema
from anemoi.training.schemas.diagnostics import LossPlotFnSchema
from anemoi.training.schemas.diagnostics import PlotSchema
from anemoi.training.schemas.diagnostics import PlotSettingsSchema

# Fields that exist in PlottingSettings but are runtime-only (not config-facing)
# and intentionally absent from PlotSettingsSchema.
_RUNTIME_ONLY_FIELDS = {"save_basedir", "focus_areas", "dataset_names"}

# ---------------------------------------------------------------------------
# PlotSettingsSchema
# ---------------------------------------------------------------------------


def test_plot_settings_schema_defaults() -> None:
    s = PlotSettingsSchema()
    assert s.datashader is True
    assert s.projection_kind == "equirectangular"
    assert s.asynchronous is True
    assert s.colormaps is None
    assert s.precip_and_related_fields is None


def test_plot_settings_schema_override() -> None:
    s = PlotSettingsSchema(datashader=False, projection_kind="lambert_conformal")
    assert s.datashader is False
    assert s.projection_kind == "lambert_conformal"


def test_plot_settings_schema_precip_fields() -> None:
    s = PlotSettingsSchema(precip_and_related_fields=["tp", "cp"])
    assert s.precip_and_related_fields == ["tp", "cp"]


def test_plot_settings_schema_in_sync_with_plotting_settings() -> None:
    """Fail if PlottingSettings and PlotSettingsSchema drift out of sync.

    PlotSettingsSchema is the config-facing validator; PlottingSettings is the
    runtime object passed to plot_fn. They must agree on shared field names and
    defaults. Add new fields to _RUNTIME_ONLY_FIELDS above if they are
    intentionally absent from the schema.
    """
    runtime_fields = {
        name: field for name, field in PlottingSettings.model_fields.items() if name not in _RUNTIME_ONLY_FIELDS
    }
    schema_fields = PlotSettingsSchema.model_fields

    missing_from_schema = set(runtime_fields) - set(schema_fields)
    assert not missing_from_schema, (
        f"Fields present in PlottingSettings but missing from PlotSettingsSchema: "
        f"{missing_from_schema}. Add them to PlotSettingsSchema or to _RUNTIME_ONLY_FIELDS."
    )

    missing_from_runtime = set(schema_fields) - set(runtime_fields) - _RUNTIME_ONLY_FIELDS
    assert not missing_from_runtime, (
        f"Fields present in PlotSettingsSchema but missing from PlottingSettings: "
        f"{missing_from_runtime}. Add them to PlottingSettings."
    )

    for name in runtime_fields:
        runtime_default = PlottingSettings.model_fields[name].default
        schema_default = PlotSettingsSchema.model_fields[name].default
        assert runtime_default == schema_default, (
            f"Default for '{name}' differs: PlottingSettings={runtime_default!r}, "
            f"PlotSettingsSchema={schema_default!r}. Keep them in sync."
        )


# ---------------------------------------------------------------------------
# PlotSchema
# ---------------------------------------------------------------------------


def test_plot_schema_defaults_no_settings_key() -> None:
    # settings sub-model should be created with defaults even if not provided
    p = PlotSchema(callbacks=[])
    assert p.settings.datashader is True
    assert p.settings.asynchronous is True


def test_plot_schema_settings_override() -> None:
    p = PlotSchema(settings={"datashader": False}, callbacks=[])
    assert p.settings.datashader is False
    assert p.settings.asynchronous is True  # default preserved


def test_plot_schema_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        PlotSettingsSchema(datashader="not_a_bool")


# ---------------------------------------------------------------------------
# PlottingSettings.from_plot_config
# ---------------------------------------------------------------------------


def test_plotting_settings_from_plot_config_reads_settings_subnode() -> None:
    from anemoi.training.diagnostics.callbacks.plot import PlottingSettings

    cfg = OmegaConf.create({"settings": {"datashader": False, "projection_kind": "equirectangular"}})
    s = PlottingSettings.from_plot_config(cfg, save_basedir=None)
    assert s.datashader is False


def test_plotting_settings_from_plot_config_uses_defaults_when_settings_absent() -> None:
    from anemoi.training.diagnostics.callbacks.plot import PlottingSettings

    cfg = OmegaConf.create({})
    s = PlottingSettings.from_plot_config(cfg, save_basedir=None)
    assert s.datashader is True
    assert s.projection_kind == "equirectangular"
    assert s.asynchronous is True


def test_plotting_settings_from_plot_config_reads_precip_from_settings() -> None:
    from anemoi.training.diagnostics.callbacks.plot import PlottingSettings

    cfg = OmegaConf.create({"settings": {"precip_and_related_fields": ["tp"]}})
    s = PlottingSettings.from_plot_config(cfg, save_basedir=None)
    assert s.precip_and_related_fields == ["tp"]


# ---------------------------------------------------------------------------
# *PlotFnSchema — GenericSchema accepts custom _target_ and extra kwargs
# ---------------------------------------------------------------------------

_BUILTIN_BATCH_TARGET = "anemoi.training.diagnostics.evaluation.plotting.batch_output.sample_plot_fn"
_CUSTOM_TARGET = "my_package.my_module.my_custom_plot_fn"


def test_batch_output_plot_fn_schema_builtin_target() -> None:
    s = BatchOutputPlotFnSchema(_target_=_BUILTIN_BATCH_TARGET, _partial_=True)
    assert s.target_ == _BUILTIN_BATCH_TARGET
    assert s.partial_ is True


def test_batch_output_plot_fn_schema_custom_target_accepted() -> None:
    # GenericSchema uses str for _target_, so any dotted path is valid
    s = BatchOutputPlotFnSchema(_target_=_CUSTOM_TARGET, _partial_=True)
    assert s.target_ == _CUSTOM_TARGET


def test_batch_output_plot_fn_schema_extra_kwargs_allowed() -> None:
    # extra kwargs bound via _partial_: true must pass through
    s = BatchOutputPlotFnSchema(
        _target_=_BUILTIN_BATCH_TARGET,
        _partial_=True,
        per_sample=4,
        log_scale=False,
    )
    assert s.target_ == _BUILTIN_BATCH_TARGET


def test_batch_output_plot_fn_schema_missing_target_rejected() -> None:
    with pytest.raises(ValidationError):
        BatchOutputPlotFnSchema(_partial_=True)


def test_loss_plot_fn_schema_custom_target_accepted() -> None:
    s = LossPlotFnSchema(_target_=_CUSTOM_TARGET, _partial_=True)
    assert s.target_ == _CUSTOM_TARGET


def test_loss_plot_fn_schema_extra_kwargs_allowed() -> None:
    s = LossPlotFnSchema(_target_=_CUSTOM_TARGET, _partial_=True, my_kwarg=42)
    assert s.target_ == _CUSTOM_TARGET


def test_graph_plot_fn_schema_custom_target_accepted() -> None:
    s = GraphPlotFnSchema(_target_=_CUSTOM_TARGET, _partial_=True)
    assert s.target_ == _CUSTOM_TARGET


def test_graph_plot_fn_schema_extra_kwargs_allowed() -> None:
    s = GraphPlotFnSchema(_target_=_CUSTOM_TARGET, _partial_=True, my_kwarg=99)
    assert s.target_ == _CUSTOM_TARGET


def test_loss_curve_plot_schema_accepts_custom_plot_fn() -> None:
    from anemoi.training.schemas.diagnostics import LossCurvePlotSchema

    s = LossCurvePlotSchema(
        _target_="anemoi.training.diagnostics.callbacks.plot.LossCurvePlot",
        parameter_groups={"moisture": ["tp"]},
        plot_fn={"_target_": _CUSTOM_TARGET, "_partial_": True, "my_kwarg": 1},
    )
    assert s.plot_fn.target_ == _CUSTOM_TARGET


def test_graph_feature_plot_schema_accepts_custom_plot_fn() -> None:
    from anemoi.training.schemas.diagnostics import GraphFeaturePlotSchema

    s = GraphFeaturePlotSchema(
        _target_="anemoi.training.diagnostics.callbacks.plot.GraphFeaturePlot",
        every_n_epochs=5,
        plot_fn={"_target_": _CUSTOM_TARGET, "_partial_": True, "my_kwarg": 2},
    )
    assert s.plot_fn.target_ == _CUSTOM_TARGET


# ---------------------------------------------------------------------------
# validate_plot_fn
# ---------------------------------------------------------------------------


def test_validate_plot_fn_accepts_valid_function() -> None:
    from anemoi.training.diagnostics.evaluation.plotting.protocols import BatchOutputPlotFn
    from anemoi.training.diagnostics.evaluation.plotting.protocols import validate_plot_fn

    def good_fn(
        parameters: object,
        *,
        x: object,
        y_true: object,
        y_pred: object,
        latlons: object,
        **kwargs: object,
    ) -> None:
        pass

    validate_plot_fn(good_fn, BatchOutputPlotFn, "BatchOutputPlot")  # must not raise


def test_validate_plot_fn_rejects_missing_params() -> None:
    from anemoi.training.diagnostics.evaluation.plotting.protocols import BatchOutputPlotFn
    from anemoi.training.diagnostics.evaluation.plotting.protocols import validate_plot_fn

    def bad_fn(parameters: object, *, x: object, y_true: object) -> None:  # missing y_pred, latlons
        pass

    with pytest.raises(TypeError, match="missing required parameter"):
        validate_plot_fn(bad_fn, BatchOutputPlotFn, "BatchOutputPlot")


def test_validate_plot_fn_accepts_var_keyword() -> None:
    from anemoi.training.diagnostics.evaluation.plotting.protocols import BatchOutputPlotFn
    from anemoi.training.diagnostics.evaluation.plotting.protocols import validate_plot_fn

    def flexible_fn(**kwargs: object) -> None:  # **kwargs covers all requirements
        pass

    validate_plot_fn(flexible_fn, BatchOutputPlotFn, "BatchOutputPlot")  # must not raise


def test_validate_plot_fn_unwraps_partial() -> None:
    from anemoi.training.diagnostics.evaluation.plotting.protocols import BatchOutputPlotFn
    from anemoi.training.diagnostics.evaluation.plotting.protocols import validate_plot_fn

    def good_fn(
        parameters: object,
        *,
        x: object,
        y_true: object,
        y_pred: object,
        latlons: object,
        per_sample: int = 6,
        **kwargs: object,
    ) -> None:
        pass

    partial_fn = functools.partial(good_fn, per_sample=4)
    validate_plot_fn(partial_fn, BatchOutputPlotFn, "BatchOutputPlot")  # must not raise


def test_validate_plot_fn_unwraps_partial_catches_missing() -> None:
    from anemoi.training.diagnostics.evaluation.plotting.protocols import BatchOutputPlotFn
    from anemoi.training.diagnostics.evaluation.plotting.protocols import validate_plot_fn

    def bad_fn(  # missing y_pred, latlons
        parameters: object,
        *,
        x: object,
        y_true: object,
        per_sample: int = 6,
    ) -> None:
        pass

    partial_fn = functools.partial(bad_fn, per_sample=4)
    with pytest.raises(TypeError, match="missing required parameter"):
        validate_plot_fn(partial_fn, BatchOutputPlotFn, "BatchOutputPlot")
