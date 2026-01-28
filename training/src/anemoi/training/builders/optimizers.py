# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from anemoi.training.builders.components import build_component

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    import torch

    from anemoi.training.config_types import Settings


def _normalize_betas(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    return tuple(value) if isinstance(value, list) else value


def build_optimizer_builder_from_config(
    config: Settings,
) -> Callable[[Iterable[torch.nn.Parameter], float], torch.optim.Optimizer]:
    """Return a callable that instantiates the optimizer from config."""
    opt_cfg: Any = config.training.optimizer
    if hasattr(opt_cfg, "model_dump"):
        opt_cfg = opt_cfg.model_dump(by_alias=True)
    betas = opt_cfg.get("betas") if isinstance(opt_cfg, dict) else getattr(opt_cfg, "betas", None)
    betas = _normalize_betas(betas)

    def _builder(*, params: Iterable[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
        kwargs: dict[str, Any] = {"params": params, "lr": lr}
        if betas is not None:
            kwargs["betas"] = betas
        return build_component(opt_cfg, **kwargs)

    return _builder
