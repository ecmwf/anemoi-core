# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Every migration treats PEFT adapter keys like the plain keys of the layer they adapt.

anemoi-training can add LoRA adapters to Linear layers (``PeftModifierStage``). An adapted
layer at path ``P`` saves ``P.base_layer.weight`` / ``P.base_layer.bias`` and
``P.lora_A.default.weight`` / ``P.lora_B.default.weight`` instead of ``P.weight`` /
``P.bias``. A migration that renames ``P`` has to move those keys the same way, or an
adapter checkpoint stops loading. Renaming by path component does that; matching the
full key (``key == "...weight"``, ``key.endswith(".weight")``) does not.

When you add a migration and this test fails with a ``KeyError``, give it the keys or
hyper-parameters it reads in ``_checkpoint`` below.
"""

import copy
import importlib
import inspect
import pkgutil

import pytest
import torch
from omegaconf import OmegaConf

from anemoi.models.migrations import scripts

PLAIN = (".weight", ".bias")
ADAPTER = (
    ".base_layer.weight",
    ".base_layer.bias",
    ".lora_A.default.weight",
    ".lora_B.default.weight",
)

# Linear layers in each historical layout a migration rewrites.
LINEARS = [
    "model.model.processor.proc.1.blocks.0.lin_key",  # chunked processor (chunking_fix)
    "model.model.processor.proc.0.blocks.1.node_dst_mlp.0",  # chunked, pre-GLU MLP
    "model.model.processor.proc.2.lin_value",  # flat processor
    "model.model.encoder.proc.lin_query",  # single-dataset encoder (move_to_multiple_datasets)
    "model.model.decoder.proc.node_src_mlp.0",  # pre-GLU mapper MLP
    "model.model.encoder.data.proc.lin_qkv",  # transformer mapper block with old norms
]

# Non-Linear keys the migrations read.
OTHER_KEYS = [
    *(f"model.model.{block}.trainable.trainable" for block in ("encoder", "processor", "decoder")),
    *(f"model.model.{block}.edge_inc" for block in ("encoder", "processor", "decoder")),
    "model.model.node_attributes.data.latlons_data",
    "model.model.node_attributes.data.trainable_tensors.data.trainable",
    *(
        f"model.model.encoder.data.proc.{norm}.{p}"
        for norm in ("layer_norm_mpl", "layer_norm_attention", "layer_norm_mlp")
        for p in ("weight", "bias")
    ),
    "model.pre_processors.normalizer._norm_mul",
]


def _migrations() -> list[tuple[str, object]]:
    found = []
    for info in pkgutil.iter_modules(scripts.__path__):
        module = importlib.import_module(f"{scripts.__name__}.{info.name}")
        if hasattr(module, "migrate"):
            found.append((info.name, module.migrate))
    return found


def _checkpoint(suffixes: tuple[str, ...]) -> dict:
    """A checkpoint with every Linear in ``LINEARS`` saved with ``suffixes``.

    Each Linear's tensors are filled with its index in ``LINEARS``, so its keys can be
    followed through a rename. The other keys hold -1.
    """
    state_dict = {
        prefix + suffix: torch.full((2,), float(i)) for i, prefix in enumerate(LINEARS) for suffix in suffixes
    }
    state_dict.update({key: torch.full((2,), -1.0) for key in OTHER_KEYS})
    hyper_parameters = {
        "config": OmegaConf.create({"model": {"processor": {"num_layers": 4, "num_chunks": 2}}, "training": {}}),
        "data_indices": {},
        "statistics": {},
        "statistics_tendencies": {},
        "supporting_arrays": {},
        "graph_data": {"data": None},
    }
    return {"state_dict": state_dict, "hyper_parameters": hyper_parameters}


def _destinations(state_dict: dict) -> dict[int, str]:
    """Where each Linear's ``.weight`` ended up, by its index in ``LINEARS``."""
    return {
        int(value[0]): key[: -len(".weight")]
        for key, value in state_dict.items()
        if key.endswith(".weight") and value[0] >= 0 and not key.endswith((".base_layer.weight", ".default.weight"))
    }


def test_the_property_covers_every_migration_script() -> None:
    assert len(_migrations()) >= 10


@pytest.mark.parametrize(("name", "migrate"), _migrations(), ids=[name for name, _ in _migrations()])
def test_migration_moves_adapter_keys_with_their_layer(name: str, migrate: object) -> None:
    kwargs = {"model": None} if "model" in inspect.signature(migrate).parameters else {}
    plain = migrate(copy.deepcopy(_checkpoint(PLAIN)), **kwargs)["state_dict"]
    adapted = migrate(copy.deepcopy(_checkpoint(ADAPTER)), **kwargs)["state_dict"]

    destinations = _destinations(plain)
    expected = {destinations[i] + suffix for i in range(len(LINEARS)) if i in destinations for suffix in ADAPTER}
    got = {key for key in adapted if key.endswith(ADAPTER)}
    assert got == expected, (
        f"{name} moved adapter keys differently from the plain keys of their layer. "
        f"Missing: {sorted(expected - got)[:4]}. Unexpected: {sorted(got - expected)[:4]}"
    )
