# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``build_checkpoint_pipeline``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

import anemoi.training
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.checkpoint.builder import resumes_via_lightning
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.runner import run_checkpoint_pipeline
from anemoi.training.checkpoint.loading.strategies import WarmStartLoader
from anemoi.training.checkpoint.sources.base import CheckpointSource
from anemoi.training.checkpoint.sources.run import RunIdSource

_FREEZING_TARGET = "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage"
_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
_LOADERS = "anemoi.training.checkpoint.loading.strategies"
_WEIGHTS_ONLY = {"_target_": f"{_LOADERS}.WeightsOnlyLoader"}


def _checkpoint_template_dir() -> Path:
    """Absolute path to the shipped ``training/checkpoint`` config group."""
    return Path(anemoi.training.__file__).parent / "config" / "training" / "checkpoint"


def _load_template(group: str, name: str) -> DictConfig:
    """Load a shipped checkpoint template (``source/*`` or ``loading/*``)."""
    return OmegaConf.load(_checkpoint_template_dir() / group / f"{name}.yaml")


def _freezing_modifier(submodules: list[str] | None = None) -> dict:
    """Return a FreezingModifierStage config with the given submodules."""
    return {"_target_": _FREEZING_TARGET, "submodules_to_freeze": submodules or []}


def compose_test_config(
    *,
    source: str | None = None,
    loading: str | None = None,
    modifiers: list | None = None,
) -> DictConfig:
    """Build a minimal training config for the builder.

    ``modifiers`` items are template names (only ``"freezing"``) or config dicts.
    """
    checkpoint: dict = {}
    if source is not None:
        checkpoint["source"] = _load_template("source", source)
    if loading is not None:
        checkpoint["loading"] = _load_template("loading", loading)
    if modifiers:
        checkpoint["modifiers"] = [m if isinstance(m, dict) else _freezing_modifier() for m in modifiers]

    training: dict = {"checkpoint": checkpoint} if checkpoint else {}
    return OmegaConf.create({"training": training})


def test_builder_orders_stages_source_loader_modifiers() -> None:
    cfg = compose_test_config(
        source="local",
        loading="weights_only",
        modifiers=["freezing", "freezing"],
    )
    pipeline = build_checkpoint_pipeline(cfg)
    names = [type(s).__name__ for s in pipeline.stages]
    assert names[0].endswith("Source")
    assert names[1].endswith("Loader")
    assert all(n.endswith("Stage") for n in names[2:])


def test_builder_orders_source_loader_without_modifiers() -> None:
    cfg = compose_test_config(source="local", loading="weights_only", modifiers=[])
    pipeline = build_checkpoint_pipeline(cfg)
    names = [type(s).__name__ for s in pipeline.stages]
    assert names[0].endswith("Source")
    assert names[1].endswith("Loader")
    assert len(names) == 2


def test_hydra_defaults_compose() -> None:
    """Every shipped template under training/checkpoint/* composes without error."""
    templates = sorted(_checkpoint_template_dir().rglob("*.yaml"))
    assert templates, f"no checkpoint templates found under {_checkpoint_template_dir()}"
    for template in templates:
        cfg = OmegaConf.load(template)
        # Instantiate to catch unsupported kwargs.
        instantiate(cfg)


def test_modifiers_list_order_preserved() -> None:
    """Modifiers run in the order they are listed."""
    cfg = compose_test_config(
        source="local",
        loading="weights_only",
        modifiers=[_freezing_modifier(["encoder"]), _freezing_modifier(["decoder"])],
    )
    pipeline = build_checkpoint_pipeline(cfg)
    frozen = [stage.submodules_to_freeze for stage in pipeline.stages[2:]]
    assert frozen == [["encoder"], ["decoder"]]


def test_no_checkpoint_config_builds_empty_pipeline() -> None:
    """No training.checkpoint gives an empty pipeline."""
    cfg = OmegaConf.create({"training": {}})
    pipeline = build_checkpoint_pipeline(cfg)
    assert len(pipeline.stages) == 0


def _run_source_cfg(target: str = _RUN_SOURCE, **source: object) -> DictConfig:
    """Return a config with a run source and a weights-only loader."""
    return OmegaConf.create(
        {"training": {"checkpoint": {"source": {"_target_": target, **source}, "loading": _WEIGHTS_ONLY}}},
    )


def test_builder_injects_server2server_into_run_source() -> None:
    """Runtime server-to-server lineage is added to a RunIdSource config."""
    pipeline = build_checkpoint_pipeline(
        _run_source_cfg(run_id="abc"),
        parent_run_server2server="remote-parent",
        fork_run_server2server="remote-fork",
    )
    source = pipeline.stages[0]
    assert source.parent_run_server2server == "remote-parent"
    assert source.fork_run_server2server == "remote-fork"


def test_builder_server2server_defaults_leave_run_source_untouched() -> None:
    """Without runtime lineage, RunIdSource is built from the config only."""
    source = build_checkpoint_pipeline(_run_source_cfg(run_id="abc")).stages[0]
    assert source.parent_run_server2server is None
    assert source.fork_run_server2server is None


def test_builder_server2server_ignored_for_local_source() -> None:
    """Runtime lineage kwargs have no effect on other sources."""
    cfg = compose_test_config(source="local", loading="weights_only")
    pipeline = build_checkpoint_pipeline(cfg, parent_run_server2server="remote-parent")
    assert type(pipeline.stages[0]).__name__ == "LocalSource"


# --- resume: Trainer.fit(ckpt_path=) loads the checkpoint ---


class _EncoderNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)


class ResumeLoader(WarmStartLoader):
    """Loader that restores training state, with a name other than ``WarmStartLoader``."""


_RESUME_LOADER = f"{__name__}.ResumeLoader"
_S3_SOURCE = "anemoi.training.checkpoint.sources.s3.S3Source"


def _spy_torch_load(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []
    real_load = torch.load

    def _spy(*args: object, **kwargs: object) -> object:
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy)
    return calls


def _resume_cfg(tmp_path: Path, loading: str | None, modifiers: list | None = None) -> tuple[DictConfig, Path]:
    ckpt = tmp_path / "last.ckpt"
    torch.save({"state_dict": _EncoderNet().state_dict()}, ckpt)
    checkpoint: dict = {
        "source": {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource", "path": str(ckpt)},
    }
    if loading is not None:
        checkpoint["loading"] = {"_target_": loading}
    if modifiers:
        checkpoint["modifiers"] = modifiers
    return OmegaConf.create({"training": {"checkpoint": checkpoint}}), ckpt


def test_warm_start_resolves_the_source_only_and_emits_no_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``restores_training_state``, the runner only resolves the file and loads nothing."""
    cfg, ckpt = _resume_cfg(tmp_path, f"{_LOADERS}.WarmStartLoader")
    loads = _spy_torch_load(monkeypatch)
    model = _EncoderNet()
    before = {key: value.clone() for key, value in model.state_dict().items()}

    result = run_checkpoint_pipeline(model, cfg)

    assert build_checkpoint_pipeline(cfg).stages == []
    assert loads == []
    assert result.checkpoint_path == ckpt.resolve()
    assert result.weights_loaded is False
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_warm_start_still_runs_modifiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Modifiers still apply on a resume."""
    cfg, _ = _resume_cfg(
        tmp_path,
        f"{_LOADERS}.WarmStartLoader",
        modifiers=[{"_target_": _FREEZING_TARGET, "submodules_to_freeze": ["encoder"]}],
    )
    loads = _spy_torch_load(monkeypatch)
    model = _EncoderNet()

    run_checkpoint_pipeline(model, cfg)

    assert [type(stage).__name__ for stage in build_checkpoint_pipeline(cfg).stages] == ["FreezingModifierStage"]
    assert loads == []
    assert model.encoder.weight.requires_grad is False
    assert model.decoder.weight.requires_grad is True


def test_source_without_loading_is_a_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source without a loader is a resume: it resolves the file and loads nothing."""
    cfg, ckpt = _resume_cfg(tmp_path, None)
    loads = _spy_torch_load(monkeypatch)

    result = run_checkpoint_pipeline(_EncoderNet(), cfg)

    assert build_checkpoint_pipeline(cfg).stages == []
    assert loads == []
    assert result.checkpoint_path == ckpt.resolve()


def test_restoring_loader_subclass_is_treated_like_warm_start(tmp_path: Path) -> None:
    """The resume decision uses the class attribute, not the class name."""
    cfg, _ = _resume_cfg(tmp_path, _RESUME_LOADER)
    assert build_checkpoint_pipeline(cfg).stages == []


def test_weights_only_loader_still_loads_in_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-resume loaders load weights in the pipeline."""
    cfg, _ = _resume_cfg(tmp_path, f"{_LOADERS}.WeightsOnlyLoader")
    loads = _spy_torch_load(monkeypatch)
    model = _EncoderNet()

    pipeline = build_checkpoint_pipeline(cfg)
    pipeline.execute(CheckpointContext(model=model, config=cfg))

    assert [type(stage).__name__ for stage in pipeline.stages] == ["LocalSource", "WeightsOnlyLoader"]
    assert len(loads) == 1
    assert model.weights_initialized is True


@pytest.mark.parametrize("loading", [None, {"_target_": f"{_LOADERS}.WarmStartLoader"}])
def test_resume_from_a_remote_source_is_rejected(loading: dict | None) -> None:
    """Lightning needs a local file, thus a resume from S3 fails at build."""
    checkpoint = {"source": {"_target_": _S3_SOURCE, "url": "s3://bucket/last.ckpt"}}
    if loading is not None:
        checkpoint["loading"] = loading
    cfg = OmegaConf.create({"training": {"checkpoint": checkpoint}})
    with pytest.raises(CheckpointConfigError, match="S3Source cannot resume"):
        build_checkpoint_pipeline(cfg)


def test_remote_source_with_a_loading_strategy_builds() -> None:
    """A remote source with a pipeline loader is not a resume, thus it builds."""
    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {"_target_": _S3_SOURCE, "url": "s3://bucket/last.ckpt"},
                    "loading": {"_target_": f"{_LOADERS}.WeightsOnlyLoader"},
                },
            },
        },
    )
    pipeline = build_checkpoint_pipeline(cfg)
    assert [type(stage).__name__ for stage in pipeline.stages] == ["S3Source", "WeightsOnlyLoader"]


def test_warm_start_without_a_source_is_rejected() -> None:
    """A resume without a source fails at build."""
    cfg = OmegaConf.create({"training": {"checkpoint": {"loading": {"_target_": f"{_LOADERS}.WarmStartLoader"}}}})
    with pytest.raises(CheckpointConfigError, match=r"no training\.checkpoint\.source"):
        build_checkpoint_pipeline(cfg)


@pytest.mark.parametrize(
    ("loading", "expected"),
    [
        (None, True),
        (f"{_LOADERS}.WarmStartLoader", True),
        (_RESUME_LOADER, True),
        (f"{_LOADERS}.WeightsOnlyLoader", False),
        (f"{_LOADERS}.ColdStartLoader", False),
        (f"{_LOADERS}.TransferLearningLoader", False),
    ],
)
def test_resumes_via_lightning(loading: str | None, expected: bool) -> None:
    """Check which configs resume through Lightning."""
    checkpoint: dict = {"source": {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource", "path": "x"}}
    if loading is not None:
        checkpoint["loading"] = {"_target_": loading}
    assert resumes_via_lightning(OmegaConf.create({"training": {"checkpoint": checkpoint}})) is expected


class _NothingSource(CheckpointSource):
    """Source that resolves nothing on all ranks."""

    def resolve(self, context: CheckpointContext) -> Path | None:
        del context
        return None

    def process(self, context: CheckpointContext) -> CheckpointContext:
        return context


def test_resume_that_resolves_nothing_fails_on_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume that resolves no checkpoint on rank 0 raises a config error."""
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    cfg = OmegaConf.create({"training": {"checkpoint": {"source": {"_target_": f"{__name__}._NothingSource"}}}})

    with pytest.raises(CheckpointConfigError, match=r"resolved no checkpoint file"):
        run_checkpoint_pipeline(_EncoderNet(), cfg)


def test_resume_with_the_run_preset_and_no_run_id_names_the_key(tmp_path: Path) -> None:
    """The ``source=run`` preset without a run_id raises an error that names the key."""
    cfg = OmegaConf.create(
        {
            "training": {"checkpoint": {"source": {"_target_": _RUN_SOURCE, "run_id": None}}},
            "system": {"output": {"checkpoints": {"root": str(tmp_path / "ckpts")}}},
        },
    )
    with pytest.raises(CheckpointConfigError, match="requires a run_id"):
        run_checkpoint_pipeline(_EncoderNet(), cfg)


def test_resume_that_resolves_nothing_defers_on_other_ranks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-zero ranks defer to the error on rank 0."""
    monkeypatch.setattr(rank_zero_only, "rank", 1)
    cfg = OmegaConf.create(
        {
            "training": {"checkpoint": {"source": {"_target_": _RUN_SOURCE, "run_id": "missing"}}},
            "system": {"output": {"checkpoints": {"root": str(tmp_path / "ckpts")}}},
        },
    )
    result = run_checkpoint_pipeline(_EncoderNet(), cfg)

    assert result.checkpoint_path is None


@pytest.mark.parametrize(("template", "required_key"), [("local", "path"), ("run", "run_id")])
def test_source_presets_declare_their_required_key(template: str, required_key: str) -> None:
    """Each source preset declares its required key as null."""
    template_cfg = _load_template("source", template)
    assert required_key in template_cfg
    assert template_cfg[required_key] is None


def test_dry_run_suppresses_acquisition_but_keeps_modifiers() -> None:
    """A dry run has no source and no loader, but keeps the modifiers.

    ``anemoi-training mlflow prepare`` makes a run ID with no checkpoint yet.
    """
    cfg = compose_test_config(source="run", loading="weights_only", modifiers=["freezing"])

    names = [type(s).__name__ for s in build_checkpoint_pipeline(cfg, load_checkpoint=False).stages]

    assert names == ["FreezingModifierStage"]


def test_dry_run_suppresses_a_resume_too() -> None:
    """A dry run does not resolve a resume source. The ``run`` preset has no run_id."""
    cfg = compose_test_config(source="run")

    assert run_checkpoint_pipeline(_EncoderNet(), cfg, load_checkpoint=False).checkpoint_path is None


def test_load_checkpoint_defaults_to_acquiring() -> None:
    """Control: by default the same config builds the full pipeline."""
    cfg = compose_test_config(source="run", loading="weights_only")

    names = [type(s).__name__ for s in build_checkpoint_pipeline(cfg).stages]

    assert names == ["RunIdSource", "WeightsOnlyLoader"]


# --- source type comes from the class, not the name ---


class SiteRunSource(RunIdSource):
    """Subclass with a ``_target_`` that does not end in ``RunIdSource``."""


_SITE_RUN_SOURCE = f"{__name__}.SiteRunSource"


def test_builder_injects_server2server_into_a_run_source_subclass() -> None:
    cfg = _run_source_cfg(_SITE_RUN_SOURCE, run_id="abc")
    pipeline = build_checkpoint_pipeline(cfg, parent_run_server2server="remote-parent")

    source = pipeline.stages[0]
    assert isinstance(source, SiteRunSource)
    assert source.parent_run_server2server == "remote-parent"


def test_run_identity_is_read_from_a_run_source_subclass() -> None:
    from anemoi.training.checkpoint.sources.run import run_identity_from_config

    cfg = OmegaConf.create(
        {"training": {"checkpoint": {"source": {"_target_": _SITE_RUN_SOURCE, "run_id": "abc", "fork": True}}}},
    )
    assert run_identity_from_config(cfg) == (None, "abc")


def test_unresolvable_source_target_carries_no_identity_and_is_left_for_the_build_to_report() -> None:
    """The helper returns no identity; the pipeline build raises CheckpointConfigError."""
    from anemoi.training.checkpoint.sources.run import run_identity_from_config

    cfg = _run_source_cfg("no.such.Source", run_id="abc")
    assert run_identity_from_config(cfg) == (None, None)
    with pytest.raises(CheckpointConfigError):
        build_checkpoint_pipeline(cfg, parent_run_server2server="remote-parent")
