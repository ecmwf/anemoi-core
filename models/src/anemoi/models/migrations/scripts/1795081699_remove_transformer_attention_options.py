# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from anemoi.models.migrations import CkptType
from anemoi.models.migrations import MigrationMetadata

# DO NOT CHANGE -->
metadata = MigrationMetadata(
    versions={
        "migration": "1.0.0",
        "anemoi-models": "%NEXT_ANEMOI_MODELS_VERSION%",
    },
)
# <-- END DO NOT CHANGE

# Attention options of the transformer processor, encoders and decoders that the model has no
# counterpart for.
REMOVED_OPTIONS = ("softcap", "use_alibi_slopes", "use_rotary_embeddings")


def _attention_sections(model_config: Any) -> dict[str, Any]:
    """The processor, encoder, decoder and mapper settings in the model config, by their path in it.

    Covers the single ``encoder``/``decoder`` sections and the per-dataset ``encoders``/``decoders``.
    In a per-dataset section the attention options are in its ``mapper``, so the ``mapper`` of each
    section is covered too.
    """
    sections = {name: model_config.get(name) for name in ("processor", "encoder", "decoder")}
    for group in ("encoders", "decoders"):
        for name, section in (model_config.get(group) or {}).items():
            sections[f"{group}.{name}"] = section
    for path, section in list(sections.items()):
        mapper = section.get("mapper") if hasattr(section, "get") else None
        if hasattr(mapper, "get"):
            sections[f"{path}.mapper"] = mapper
    return {path: section for path, section in sections.items() if hasattr(section, "get")}


def migrate(ckpt: CkptType) -> CkptType:
    """Migrate the checkpoint.

    Removes the attention options ``softcap``, ``use_alibi_slopes`` and
    ``use_rotary_embeddings`` from the processor, encoder, decoder and mapper configs.
    The model weights are unchanged: none of these options has weights of its own.

    Before: ``model.processor.softcap: 0.0``, ``model.processor.use_alibi_slopes: False``,
            ``model.processor.use_rotary_embeddings: False``
    After:  none of the three keys

    A checkpoint trained with any of them switched on cannot be migrated, since
    the model would attend differently from how it was trained. Use an
    anemoi-models release from before this migration for such checkpoints.

    Parameters
    ----------
    ckpt : CkptType
        The checkpoint dict.

    Returns
    -------
    CkptType
        The migrated checkpoint dict.

    Raises
    ------
    ValueError
        If the checkpoint was trained with softcapping, ALiBi slopes or flash-attn rotary embeddings.
    """
    sections = _attention_sections(ckpt["hyper_parameters"]["config"].model)

    switched_on = [
        f"model.{path}.{option}"
        for path, section in sections.items()
        for option in REMOVED_OPTIONS
        if section.get(option)
    ]
    if switched_on:
        raise ValueError(
            f"This checkpoint was trained with {', '.join(switched_on)}, which this version of anemoi-models "
            "does not support. Use an anemoi-models release from before this migration to run it."
        )

    for section in sections.values():
        for option in REMOVED_OPTIONS:
            section.pop(option, None)
    return ckpt
