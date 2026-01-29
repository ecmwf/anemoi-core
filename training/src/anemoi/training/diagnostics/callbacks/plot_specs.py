from collections.abc import Callable
from dataclasses import dataclass

import pytorch_lightning as pl

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.tasks import GraphAutoEncoder
from anemoi.training.train.tasks import GraphInterpolator


@dataclass(frozen=True)
class TaskSpec:
    mode: str
    get_init_step: Callable[[int, pl.LightningModule], int]
    get_output_times: Callable[[BaseSchema, pl.LightningModule], int]


TIME_INTERP_SPEC = TaskSpec(
    mode="time_interp",
    get_init_step=lambda rollout_step: rollout_step,
    get_output_times=lambda config, _: len(config.training.explicit_times.target),
)

AUTOENCODER_SPEC = TaskSpec(
    mode="autoencoder",
    get_init_step=lambda rollout_step: rollout_step,
    get_output_times=lambda _, pl_module: getattr(pl_module, "rollout", 0),
)

FORECAST_SPEC = TaskSpec(
    mode="forecast",
    get_init_step=lambda _: 0,
    get_output_times=lambda _, pl_module: getattr(pl_module, "rollout", 0),
)


TASK_SPECS: list[tuple[type[pl.LightningModule], TaskSpec]] = [
    (GraphInterpolator, TIME_INTERP_SPEC),
    (GraphAutoEncoder, AUTOENCODER_SPEC),
    (pl.LightningModule, FORECAST_SPEC),  # fallback for everything else
]


def _get_task_spec(pl_module: pl.LightningModule) -> TaskSpec:
    for module_type, spec in TASK_SPECS:
        if isinstance(pl_module, module_type):
            return spec
    raise TypeError(f"No TaskSpec registered for {type(pl_module).__name__}")


__all__ = ["_get_task_spec"]
