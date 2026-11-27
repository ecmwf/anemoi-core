# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib

import pytest
import torch
from omegaconf import OmegaConf

migration = importlib.import_module("anemoi.models.migrations.scripts.1795081699_remove_transformer_attention_options")

OPTIONS_OFF = {"softcap": 0.0, "use_alibi_slopes": False, "use_rotary_embeddings": False}


def _transformer(**options) -> dict:
    return {"num_heads": 16, "attention_implementation": "triton_attention", **OPTIONS_OFF, **options}


def _checkpoint(model: dict) -> dict:
    return {
        "hyper_parameters": {"config": OmegaConf.create({"model": model})},
        "state_dict": {"model.model.processor.proc.0.weight": torch.ones(2)},
    }


def test_options_are_removed_from_processor_encoders_and_decoders() -> None:
    checkpoint = _checkpoint(
        {
            "processor": _transformer(),
            "encoders": {"era5": _transformer(), "obs": _transformer()},
            "decoders": {"era5": _transformer()},
        }
    )
    state_dict = dict(checkpoint["state_dict"])

    migration.migrate(checkpoint)

    model = checkpoint["hyper_parameters"]["config"].model
    for section in (model.processor, model.encoders.era5, model.encoders.obs, model.decoders.era5):
        assert set(section) == {"num_heads", "attention_implementation"}
    assert checkpoint["state_dict"] == state_dict


def test_options_are_removed_from_single_encoder_and_decoder() -> None:
    checkpoint = _checkpoint({"processor": _transformer(), "encoder": _transformer(), "decoder": _transformer()})

    migration.migrate(checkpoint)

    model = checkpoint["hyper_parameters"]["config"].model
    for section in (model.processor, model.encoder, model.decoder):
        assert not set(section) & set(migration.REMOVED_OPTIONS)


def test_configs_without_the_options_are_unchanged() -> None:
    model = {"processor": {"num_layers": 16, "mlp_hidden_ratio": 4}, "encoders": {"era5": {"trainable_size": 8}}}
    checkpoint = _checkpoint(model)

    migration.migrate(checkpoint)

    assert OmegaConf.to_container(checkpoint["hyper_parameters"]["config"].model) == model


@pytest.mark.parametrize(
    ("option", "value"), [("softcap", 30.0), ("use_alibi_slopes", True), ("use_rotary_embeddings", True)]
)
def test_checkpoints_trained_with_an_option_switched_on_are_refused(option, value) -> None:
    checkpoint = _checkpoint({"processor": _transformer(), "decoders": {"era5": _transformer(**{option: value})}})

    with pytest.raises(ValueError, match=f"model.decoders.era5.{option}"):
        migration.migrate(checkpoint)


def test_options_are_removed_from_the_mappers_of_encoders_and_decoders() -> None:
    """In the training configs the options of a per-dataset encoder or decoder are in its mapper."""
    checkpoint = _checkpoint(
        {
            "processor": _transformer(),
            "encoders": {"0": {"source_datasets": ["data"], "mapper": _transformer()}},
            "decoders": {"0": {"target_datasets": ["data"], "mapper": _transformer()}},
        }
    )

    migration.migrate(checkpoint)

    model = checkpoint["hyper_parameters"]["config"].model
    for section in (model.encoders["0"].mapper, model.decoders["0"].mapper):
        assert set(section) == {"num_heads", "attention_implementation"}
    assert list(model.encoders["0"].source_datasets) == ["data"]


@pytest.mark.parametrize("path", ["encoders.0.mapper", "decoders.0.mapper"])
@pytest.mark.parametrize(
    ("option", "value"), [("softcap", 30.0), ("use_alibi_slopes", True), ("use_rotary_embeddings", True)]
)
def test_mappers_trained_with_an_option_switched_on_are_refused(path, option, value) -> None:
    model = {
        "processor": _transformer(),
        "encoders": {"0": {"mapper": _transformer()}},
        "decoders": {"0": {"mapper": _transformer()}},
    }
    section = model
    for key in path.split("."):
        section = section[key]
    section[option] = value
    checkpoint = _checkpoint(model)

    with pytest.raises(ValueError, match=f"model.{path}.{option}"):
        migration.migrate(checkpoint)
