from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.tasks import GraphAutoEncoder
from anemoi.training.train.tasks import GraphInterpolator


@dataclass(frozen=True)
class TaskSpec:
    get_init_step: Callable[[int, pl.LightningModule], int]
    get_output_times: Callable[[BaseSchema, pl.LightningModule], int]
    get_t_index: Callable[[int, int], int]  # rollout_step, multi_step → t_index
    get_y_true: Callable[[Any, int, int], Any]
    n_plots_sample: int


TIME_INTERP_SPEC = TaskSpec(
    get_init_step=lambda rollout_step: rollout_step,
    get_output_times=lambda config, _: len(config.training.explicit_times.target),
    get_t_index=lambda rollout_step, multi_step: multi_step + rollout_step,
    get_y_true=lambda data, rollout_step: data[rollout_step + 1, ...].squeeze(),
    n_plots_sample=6,
)

AUTOENCODER_SPEC = TaskSpec(
    get_init_step=lambda _: 0,
    get_output_times=lambda _, pl_module: getattr(pl_module, "rollout", 0),
    get_t_index=lambda _: 0,
    get_y_true=None,
    n_plots_sample=3,
)

FORECAST_SPEC = TaskSpec(
    get_init_step=lambda _: 0,
    get_output_times=lambda _, pl_module: getattr(pl_module, "rollout", 0),
    get_t_index=lambda rollout_step, multi_step: multi_step + rollout_step,
    get_y_true=lambda data, rollout_step: data[rollout_step + 1, ...].squeeze(),
    n_plots_sample=6,
)


PLOT_TASK_SPECS: list[tuple[type[pl.LightningModule], TaskSpec]] = [
    (GraphInterpolator, TIME_INTERP_SPEC),
    (GraphAutoEncoder, AUTOENCODER_SPEC),
    (pl.LightningModule, FORECAST_SPEC),  # fallback for everything else
]


def _get_task_spec(pl_module: pl.LightningModule) -> TaskSpec:
    for module_type, spec in PLOT_TASK_SPECS:
        if isinstance(pl_module, module_type):
            return spec
    raise TypeError(f"No TaskSpec registered for {type(pl_module).__name__}")


__all__ = ["_get_task_spec"]
