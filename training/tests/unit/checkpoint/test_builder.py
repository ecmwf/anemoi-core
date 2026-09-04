# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``build_checkpoint_pipeline`` (config wiring → pipeline)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

import anemoi.training
from anemoi.training.checkpoint.base import CheckpointContext
from anemoi.training.checkpoint.builder import build_checkpoint_pipeline
from anemoi.training.checkpoint.builder import resumes_via_lightning
from anemoi.training.checkpoint.exceptions import CheckpointConfigError
from anemoi.training.checkpoint.loading.strategies import WarmStartLoader

_FREEZING_TARGET = "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage"
_RUN_SOURCE = "anemoi.training.checkpoint.sources.run.RunIdSource"
_LOADERS = "anemoi.training.checkpoint.loading.strategies"


def _checkpoint_template_dir() -> Path:
    """Absolute path to the shipped ``training/checkpoint`` config group."""
    return Path(anemoi.training.__file__).parent / "config" / "training" / "checkpoint"


def _load_template(group: str, name: str) -> DictConfig:
    """Load a shipped checkpoint template (``source/*`` or ``loading/*``)."""
    return OmegaConf.load(_checkpoint_template_dir() / group / f"{name}.yaml")


def _freezing_modifier(submodules: list[str] | None = None) -> dict:
    """Inline FreezingModifierStage config.

    The freezing template ships with the modifier layer (PR #442), not with this
    layer, so the modifier config is constructed inline for the guarded tests.
    """
    return {"_target_": _FREEZING_TARGET, "submodules_to_freeze": submodules or []}


def compose_test_config(
    *,
    source: str | None = None,
    loading: str | None = None,
    modifiers: list | None = None,
) -> DictConfig:
    """Build a minimal training config that exercises the builder's namespaces.

    ``modifiers`` accepts either template names (currently only ``"freezing"``)
    or pre-built modifier config dicts, allowing order-sensitive assertions.
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
    pytest.importorskip("anemoi.training.checkpoint.modifiers.freezing", reason="PR #442")
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
        # Instantiation (not merely loading) is what catches an unsupported kwarg,
        # e.g. a `strict` key on a loader that does not accept one (-> TypeError).
        instantiate(cfg)


def test_modifiers_list_order_preserved() -> None:
    """D11: list order == execution order."""
    pytest.importorskip("anemoi.training.checkpoint.modifiers.freezing", reason="PR #442")
    cfg = compose_test_config(
        source="local",
        loading="weights_only",
        modifiers=[_freezing_modifier(["encoder"]), _freezing_modifier(["decoder"])],
    )
    pipeline = build_checkpoint_pipeline(cfg)
    frozen = [stage.submodules_to_freeze for stage in pipeline.stages[2:]]
    assert frozen == [["encoder"], ["decoder"]]


def test_no_checkpoint_config_builds_empty_pipeline() -> None:
    """Absent training.checkpoint => no pipeline; trainer behaves as today."""
    cfg = OmegaConf.create({"training": {}})
    pipeline = build_checkpoint_pipeline(cfg)
    assert len(pipeline.stages) == 0


def test_builder_injects_server2server_into_run_source() -> None:
    """Runtime server-to-server lineage is merged onto a RunIdSource source config."""
    cfg = OmegaConf.create({"training": {"checkpoint": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc"}}}})
    pipeline = build_checkpoint_pipeline(
        cfg,
        parent_run_server2server="remote-parent",
        fork_run_server2server="remote-fork",
    )
    # A source-only config is a resume: the stage is the resolve-only wrapper around the source.
    source = pipeline.stages[0].source
    assert source.parent_run_server2server == "remote-parent"
    assert source.fork_run_server2server == "remote-fork"


def test_builder_server2server_defaults_leave_run_source_untouched() -> None:
    """Without runtime lineage, a config-provided RunIdSource is built verbatim."""
    cfg = OmegaConf.create({"training": {"checkpoint": {"source": {"_target_": _RUN_SOURCE, "run_id": "abc"}}}})
    pipeline = build_checkpoint_pipeline(cfg)
    source = pipeline.stages[0].source
    assert source.parent_run_server2server is None
    assert source.fork_run_server2server is None


def test_builder_server2server_ignored_for_local_source() -> None:
    """Runtime lineage kwargs are a no-op for non-RunIdSource sources (no instantiation error)."""
    cfg = compose_test_config(source="local")
    pipeline = build_checkpoint_pipeline(cfg, parent_run_server2server="remote-parent")
    assert type(pipeline.stages[0].source).__name__.endswith("LocalSource")


# --- a resume has one owner: Trainer.fit(ckpt_path=) ------------------------------


class _EncoderNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)


class ResumeLoader(WarmStartLoader):
    """A loader that restores training state without being named ``WarmStartLoader``.

    Stands in for a third-party strategy that declares ``restores_training_state``; its
    ``_target_`` does not end in ``WarmStartLoader``, so a class-name match would miss it.
    """


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
    """For a loader declaring ``restores_training_state`` the pipeline resolves and loads nothing.

    ``Trainer.fit(ckpt_path=)`` reads the file and runs the corrections in
    ``on_load_checkpoint``; a load here would apply them to a copy Lightning discards.
    """
    cfg, ckpt = _resume_cfg(tmp_path, f"{_LOADERS}.WarmStartLoader")
    loads = _spy_torch_load(monkeypatch)
    model = _EncoderNet()
    before = {key: value.clone() for key, value in model.state_dict().items()}

    pipeline = build_checkpoint_pipeline(cfg)
    executed = asyncio.run(pipeline.execute(CheckpointContext(model=model, config=cfg)))

    assert [type(stage).__name__ for stage in pipeline.stages] == ["ResolveOnlySource"]
    assert loads == []
    assert executed.checkpoint_path == ckpt.resolve()
    assert executed.checkpoint_data is None
    assert not getattr(model, "weights_initialized", False)
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value)


def test_warm_start_still_runs_modifiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Modifiers change the model, not the checkpoint, so they still apply on a resume."""
    cfg, _ = _resume_cfg(
        tmp_path,
        f"{_LOADERS}.WarmStartLoader",
        modifiers=[{"_target_": _FREEZING_TARGET, "submodules_to_freeze": ["encoder"]}],
    )
    loads = _spy_torch_load(monkeypatch)
    model = _EncoderNet()

    pipeline = build_checkpoint_pipeline(cfg)
    asyncio.run(pipeline.execute(CheckpointContext(model=model, config=cfg)))

    assert [type(stage).__name__ for stage in pipeline.stages] == ["ResolveOnlySource", "FreezingModifierStage"]
    assert loads == []
    assert model.encoder.weight.requires_grad is False
    assert model.decoder.weight.requires_grad is True


def test_source_without_loading_is_a_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare source resumes: the file is resolved for ``Trainer.fit(ckpt_path=)`` and nothing is loaded.

    This is the configuration the user guide prescribes for restarting a run.
    """
    cfg, ckpt = _resume_cfg(tmp_path, None)
    loads = _spy_torch_load(monkeypatch)

    pipeline = build_checkpoint_pipeline(cfg)
    executed = asyncio.run(pipeline.execute(CheckpointContext(model=_EncoderNet(), config=cfg)))

    assert [type(stage).__name__ for stage in pipeline.stages] == ["ResolveOnlySource"]
    assert loads == []
    assert executed.checkpoint_path == ckpt.resolve()


def test_restoring_loader_subclass_is_treated_like_warm_start(tmp_path: Path) -> None:
    """The resume decision keys on the declared attribute, not the class name."""
    cfg, _ = _resume_cfg(tmp_path, _RESUME_LOADER)
    assert [type(stage).__name__ for stage in build_checkpoint_pipeline(cfg).stages] == ["ResolveOnlySource"]


def test_weights_only_loader_still_loads_in_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a resume is deferred to Lightning; every other loader applies weights here."""
    cfg, _ = _resume_cfg(tmp_path, f"{_LOADERS}.WeightsOnlyLoader")
    loads = _spy_torch_load(monkeypatch)
    model = _EncoderNet()

    pipeline = build_checkpoint_pipeline(cfg)
    asyncio.run(pipeline.execute(CheckpointContext(model=model, config=cfg)))

    assert [type(stage).__name__ for stage in pipeline.stages] == ["LocalSource", "WeightsOnlyLoader"]
    assert len(loads) == 1
    assert model.weights_initialized is True


def test_warm_start_from_a_remote_source_builds() -> None:
    """A remote source is downloaded to a local file for Lightning; it is no longer refused."""
    cfg = OmegaConf.create(
        {
            "training": {
                "checkpoint": {
                    "source": {"_target_": _S3_SOURCE, "url": "s3://bucket/last.ckpt"},
                    "loading": {"_target_": f"{_LOADERS}.WarmStartLoader"},
                },
            },
        },
    )
    pipeline = build_checkpoint_pipeline(cfg)
    assert [type(stage).__name__ for stage in pipeline.stages] == ["ResolveOnlySource"]
    assert type(pipeline.stages[0].source).__name__ == "S3Source"


def test_warm_start_without_a_source_is_rejected() -> None:
    """There is nothing for Lightning to read, so the composition fails at build."""
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
    """The one decision the builder and the trainer share."""
    checkpoint: dict = {"source": {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource", "path": "x"}}
    if loading is not None:
        checkpoint["loading"] = {"_target_": loading}
    assert resumes_via_lightning(OmegaConf.create({"training": {"checkpoint": checkpoint}})) is expected


def test_resume_that_resolves_nothing_fails_on_rank_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume whose source finds no checkpoint must not start from scratch in silence.

    The shipped ``source=run`` preset carries ``run_id: null``; without ``+run_id=`` the
    RunIdSource resolves nothing. That is a configuration error on rank 0, not a fresh run.
    """
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        monkeypatch.delenv(var, raising=False)
    cfg = OmegaConf.create(
        {
            "training": {"checkpoint": {"source": {"_target_": _RUN_SOURCE, "run_id": None}}},
            "system": {"output": {"checkpoints": {"root": str(tmp_path / "ckpts")}}},
        },
    )
    pipeline = build_checkpoint_pipeline(cfg)

    with pytest.raises(CheckpointConfigError, match="resolved no checkpoint file"):
        asyncio.run(pipeline.execute(CheckpointContext(model=_EncoderNet(), config=cfg)))


def test_resume_that_resolves_nothing_defers_on_other_ranks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-zero ranks defer to rank 0's error, as the sources do, so the job fails once."""
    monkeypatch.setenv("RANK", "1")
    cfg = OmegaConf.create(
        {
            "training": {"checkpoint": {"source": {"_target_": _RUN_SOURCE, "run_id": "missing"}}},
            "system": {"output": {"checkpoints": {"root": str(tmp_path / "ckpts")}}},
        },
    )
    pipeline = build_checkpoint_pipeline(cfg)

    executed = asyncio.run(pipeline.execute(CheckpointContext(model=_EncoderNet(), config=cfg)))

    assert executed.checkpoint_path is None
    assert executed.metadata["checkpoint_load_owner"] == "trainer"


@pytest.mark.parametrize(("template", "required_key"), [("local", "path"), ("run", "run_id")])
def test_source_presets_declare_their_required_key(template: str, required_key: str) -> None:
    """A source preset names the value the user has to supply.

    ``local.yaml`` shipped with only ``_target_`` and a comment claiming the path came
    from the pipeline context. Nothing supplies it (only ``RunIdSource`` ever sets
    ``context.checkpoint_path``, and it builds its own bare ``LocalSource``), so
    selecting the preset alone could only fail. Declaring the key null makes the
    requirement discoverable in the file, like its sibling already did.
    """
    template_cfg = _load_template("source", template)
    assert required_key in template_cfg
    assert template_cfg[required_key] is None


def test_dry_run_suppresses_acquisition_but_keeps_modifiers() -> None:
    """A dry run starts fresh: no source, no loader, modifiers still applied.

    ``anemoi-training mlflow prepare`` mints a run id with no checkpoint directory.
    The trainer clears ``start_from_checkpoint`` for it, and that has to reach the
    pipeline: otherwise the source stage looks for a checkpoint that was never
    written and the prepared run cannot be launched at all.
    """
    cfg = compose_test_config(source="run", loading="weights_only", modifiers=["freezing"])

    names = [type(s).__name__ for s in build_checkpoint_pipeline(cfg, load_checkpoint=False).stages]

    assert names == ["FreezingModifierStage"]


def test_dry_run_suppresses_a_resume_too() -> None:
    """The prepared run is resumed by RunIdSource; with nothing written yet there is nothing to resolve."""
    cfg = compose_test_config(source="run")

    assert build_checkpoint_pipeline(cfg, load_checkpoint=False).stages == []


def test_load_checkpoint_defaults_to_acquiring() -> None:
    """Positive control: the same config builds the full pipeline by default."""
    cfg = compose_test_config(source="run", loading="weights_only")

    names = [type(s).__name__ for s in build_checkpoint_pipeline(cfg).stages]

    assert names == ["RunIdSource", "WeightsOnlyLoader"]
