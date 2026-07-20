#!/usr/bin/env python3
# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Rebuild an Anemoi model from a checkpoint *without unpickling the model object*.

This is the "Case 2" workflow described in ``no-pickle.md``::

    model = MyModelClass(config=...)          # plain constructor
    weights = torch.load("weights.pt")        # just tensors
    model.load_state_dict(weights)

instead of the fragile "Case 1" ``model = torch.load(...)`` that unpickles a whole
``nn.Module`` (and drags Hydra / the training stack into the inference process).

The constructor of ``AnemoiModelInterface`` takes a
:class:`anemoi.utils.parametrisation.Parametrisation` and rebuilds every sub-module
through ``params.create_module`` -- no Hydra import. Here we use the JSON-backed
:class:`~anemoi.utils.parametrisation.DictParametrisation`, built from the resolved
config stored in the checkpoint metadata.

Where the inputs come from
--------------------------
Everything except the weights is assembled from the checkpoint's JSON metadata alone --
no unpickling (``anemoi.models.checkpoint.build_model_inputs``):

* ``"config"`` -- the fully-resolved training config (what the native backend wants).
* the ``"reconstruction"`` bundle -- ``data_indices`` (rebuilt from ``name_to_index``),
  the timestep counts, and a structural summary of the graph. From the summary and the
  data indices we build **placeholder** graph / statistics of the right shapes.

The weights then come from the ``state_dict`` and fill in the placeholder buffers (graph
edges, node coordinates, normalizer scale/offset). Producer side: training writes the
bundle via ``anemoi.models.checkpoint.add_reconstruction_metadata`` (wired into
``anemoi.training.utils.checkpoint.save_inference_checkpoint``).

Usage
-----
    python rebuild_from_checkpoint.py /path/to/checkpoint.ckpt
"""

from __future__ import annotations

import argparse
import logging

import torch

from anemoi.models.checkpoint import build_model_inputs
from anemoi.models.interface import AnemoiModelInterface
from anemoi.utils.parametrisation import DictParametrisation

LOGGER = logging.getLogger(__name__)


def strip_state_dict_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    """Drop a single leading ``prefix`` from every key.

    A Lightning checkpoint stores the interface as ``self.model`` on the task module, so
    its weight keys look like ``model.model.<...>`` / ``model.pre_processors.<...>``.
    A standalone ``AnemoiModelInterface`` expects ``model.<...>`` / ``pre_processors.<...>``,
    i.e. one fewer ``model.`` level. If the prefix is absent the keys are returned as-is.
    """
    if not all(k.startswith(prefix) for k in state_dict):
        return dict(state_dict)
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def load_weights(checkpoint_path: str) -> dict:
    """Load just the tensor ``state_dict`` from the checkpoint.

    For a checkpoint that stores a plain state_dict this can use ``weights_only=True``
    (no unpickling). A Lightning checkpoint also carries pickled ``hyper_parameters``, so it
    needs ``weights_only=False`` to open — but only the ``state_dict`` tensors are used here.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - Lightning checkpoints carry non-tensor objects
        LOGGER.warning("weights_only=True failed; falling back to weights_only=False for the state_dict.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def rebuild_model(checkpoint_path: str) -> AnemoiModelInterface:
    """Rebuild an ``AnemoiModelInterface`` from ``checkpoint_path`` and load its weights."""
    # 1. Assemble ALL constructor inputs from the JSON metadata only -- no unpickling.
    #    config + rebuilt data_indices + placeholder graph/statistics (filled by the weights).
    inputs = build_model_inputs(checkpoint_path)
    params = DictParametrisation(inputs.pop("config"))

    # 2. Rebuild the model -- create_module builds every sub-module without Hydra.
    model = AnemoiModelInterface(params=params, **inputs)

    # 3. Load the weights; this fills the placeholder graph / statistics / coordinate buffers
    #    with their real values.
    state_dict = strip_state_dict_prefix(load_weights(checkpoint_path))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        LOGGER.warning("Missing keys when loading weights: %s", missing_keys)
    if unexpected_keys:
        LOGGER.warning("Unexpected keys when loading weights: %s", unexpected_keys)

    model.eval()
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to the Anemoi checkpoint file")
    args = parser.parse_args()

    model = rebuild_model(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("Rebuilt %s with %d parameters (no unpickling).", type(model.model).__name__, n_params)


if __name__ == "__main__":
    main()
