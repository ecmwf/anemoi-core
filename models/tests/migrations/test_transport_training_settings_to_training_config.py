# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import importlib

from omegaconf import OmegaConf

migrate = importlib.import_module(
    "anemoi.models.migrations.scripts.1794662454_transport_training_settings_to_training_config"
).migrate


def _edm_checkpoint() -> dict:
    """A stored run config as the transport configs wrote it before the move, with its interpolations."""
    config = OmegaConf.create(
        {
            "model": {
                "model": {
                    "transport": {
                        "objective": "edm_diffusion",
                        "sigma_data": 1.0,
                        "sigma_max": 80.0,
                        "sigma_min": 0.02,
                        "rho": 7.0,
                        "training_condition": {
                            "distribution": "karras",
                            "sigma_max": "${model.model.transport.sigma_max}",
                            "sigma_min": "${model.model.transport.sigma_min}",
                            "rho": "${model.model.transport.rho}",
                        },
                        "inference_defaults": {"sampling_schedule": {"schedule_type": "karras", "sigma_max": 100.0}},
                    },
                },
            },
            "training": {"transport": {"prediction_mode": "tendency", "objective": "edm_diffusion"}, "max_steps": 10},
        },
    )
    OmegaConf.set_struct(config, True)
    return {"hyper_parameters": {"config": config}, "state_dict": {}}


def test_moves_the_training_condition_with_its_values_into_the_training_config() -> None:
    config = migrate(_edm_checkpoint())["hyper_parameters"]["config"]

    assert OmegaConf.to_container(config.training.transport) == {
        "training_condition": {"distribution": "karras", "sigma_max": 80.0, "sigma_min": 0.02, "rho": 7.0},
    }
    assert sorted(config.model.model.transport) == ["inference_defaults", "objective", "sigma_data"]
    assert config.training.max_steps == 10
    assert config.model.model.transport.inference_defaults.sampling_schedule.sigma_max == 100.0


def test_moves_a_flow_matching_training_condition() -> None:
    ckpt = _edm_checkpoint()
    transport = ckpt["hyper_parameters"]["config"].model.model.transport
    transport.objective = "flow_matching"
    transport.training_condition = {"distribution": "uniform_time"}

    config = migrate(ckpt)["hyper_parameters"]["config"]

    assert OmegaConf.to_container(config.training.transport) == {"training_condition": {"distribution": "uniform_time"}}


def test_is_idempotent() -> None:
    once = migrate(_edm_checkpoint())
    twice = migrate(copy.deepcopy(once))

    assert OmegaConf.to_container(twice["hyper_parameters"]["config"]) == OmegaConf.to_container(
        once["hyper_parameters"]["config"]
    )


def test_leaves_other_models_unchanged() -> None:
    config = OmegaConf.create({"model": {"model": {"_target_": "x"}}, "training": {"max_steps": 1}})
    ckpt = migrate({"hyper_parameters": {"config": copy.deepcopy(config)}})

    assert ckpt["hyper_parameters"]["config"] == config
