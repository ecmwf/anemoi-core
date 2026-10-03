# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch

from anemoi.models.layers.categorical import CategoricalEmbedding
from anemoi.models.layers.categorical import build_categorical_embeddings

CODES = [49001, 21009, 1004, 49003, 49002]


def test_codes_map_to_rows_in_list_order() -> None:
    emb = CategoricalEmbedding(CODES, embedding_dim=4)
    raw = torch.tensor([[49001.0, 21009.0, 1004.0, 49003.0, 49002.0]])
    assert emb.codes_to_index(raw).tolist() == [[2, 3, 4, 5, 6]]


def test_missing_and_unknown() -> None:
    emb = CategoricalEmbedding(CODES, embedding_dim=4)
    raw = torch.tensor([[0.0, float("nan"), 7.0, 99999.0, -3.0, 1.0]])
    assert emb.codes_to_index(raw).tolist() == [[0, 0, 1, 1, 1, 1]]


def test_appending_keeps_existing_indices() -> None:
    raw = torch.tensor([[float(c) for c in CODES]])
    old = CategoricalEmbedding(CODES, embedding_dim=4).codes_to_index(raw)
    new = CategoricalEmbedding([*CODES, 5, 60000], embedding_dim=4).codes_to_index(raw)
    assert torch.equal(old, new)


def test_adjacent_large_codes_get_distinct_rows() -> None:
    # Regression for bf16 collapse: these normalise to the same bf16 value under mean-std.
    emb = CategoricalEmbedding([49001, 49002, 49003, 21009, 21010], embedding_dim=4)
    idx = emb.codes_to_index(torch.tensor([[49001.0, 49002.0, 49003.0, 21009.0, 21010.0]]))
    assert len(set(idx.flatten().tolist())) == 5


@pytest.mark.parametrize(
    ("codes", "match"),
    [([], "at least one"), ([1, 0], "reserved"), ([3, 3], "Duplicate"), ([1.5], "integers"), ([2**24], "2\\*\\*24")],
)
def test_invalid_vocabularies(codes: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CategoricalEmbedding(codes, embedding_dim=2)


def test_non_integer_input_raises() -> None:
    emb = CategoricalEmbedding([3, 4, 5], embedding_dim=2).eval()
    with pytest.raises(ValueError, match="non-integer"):
        emb(torch.tensor([[3.0, 3.5]]))


def test_half_precision_input_raises() -> None:
    emb = CategoricalEmbedding(CODES, embedding_dim=2)
    with pytest.raises(TypeError, match="float32"):
        emb(torch.tensor([[49001.0]], dtype=torch.bfloat16))


def test_unknown_replacement_only_in_training() -> None:
    torch.manual_seed(0)
    emb = CategoricalEmbedding(list(range(1, 11)), embedding_dim=2, unknown_prob=0.3)
    raw = torch.arange(0, 11, dtype=torch.float32).repeat(4000, 3)  # (batch, cells)
    clean = emb.codes_to_index(raw)

    emb.eval()
    eval_rows = emb(raw)
    assert torch.equal(eval_rows, emb.embedding(clean))

    emb.train()
    replaced = emb._replace_with_unknown(clean)
    known = clean >= 2
    frac = (replaced[known] == CategoricalEmbedding.UNKNOWN).float().mean().item()
    assert abs(frac - 0.3) < 0.02
    # MISSING is never replaced, and replacement is per (sample, code): all cells agree.
    assert torch.equal(replaced[clean == 0], clean[clean == 0])
    per_sample = replaced.reshape(4000, 3, 11)
    assert torch.equal(per_sample[:, 0], per_sample[:, 1])
    assert torch.equal(per_sample[:, 0], per_sample[:, 2])


def test_extended_vocabulary_loads_from_prefix_checkpoint() -> None:
    old = CategoricalEmbedding([10, 20], embedding_dim=3)
    new = CategoricalEmbedding([10, 20, 30], embedding_dim=3)
    new.load_state_dict(old.state_dict())
    torch.testing.assert_close(new.embedding.weight[:4], old.embedding.weight)
    torch.testing.assert_close(new.embedding.weight[4], old.embedding.weight[CategoricalEmbedding.UNKNOWN])
    assert new.codes.tolist() == [10, 20, 30]


def test_reordered_vocabulary_does_not_load() -> None:
    old = CategoricalEmbedding([10, 20], embedding_dim=3)
    new = CategoricalEmbedding([20, 10, 30], embedding_dim=3)
    with pytest.raises(RuntimeError):
        new.load_state_dict(old.state_dict())


def test_gradient_only_reaches_used_rows() -> None:
    emb = CategoricalEmbedding(CODES, embedding_dim=4)
    emb(torch.tensor([[0.0, 49001.0]])).sum().backward()
    used = emb.embedding.weight.grad.abs().sum(-1) > 0
    assert used.tolist() == [True, False, True, False, False, False, False]


def test_autocast_keeps_lookup_exact() -> None:
    emb = CategoricalEmbedding(CODES, embedding_dim=4).eval()
    raw = torch.tensor([[49001.0, 49002.0, 49003.0]])
    expected = emb(raw)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = emb(raw)
    torch.testing.assert_close(out.float(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_and_autocast() -> None:
    emb = CategoricalEmbedding(CODES, embedding_dim=4, unknown_prob=0.1).cuda()
    raw = torch.tensor([[49001.0, 49002.0, 0.0]], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        idx = emb.codes_to_index(raw)
        out = emb(raw)
    assert idx.tolist() == [[2, 6, 0]]
    assert out.shape == (1, 3, 4)


def test_build_categorical_embeddings_overrides() -> None:
    specs = {"a": {"codes": [1, 2], "embedding_dim": 4, "unknown_prob": 0.1}, "b": {"codes": [5]}}
    built = build_categorical_embeddings(specs)
    assert built["a"].embedding_dim == 4 and built["a"].unknown_prob == 0.1
    assert built["b"].embedding_dim == 8 and built["b"].unknown_prob == 0.0
    overridden = build_categorical_embeddings(specs, embedding_dim=3, unknown_prob=0.2)
    assert all(e.embedding_dim == 3 and e.unknown_prob == 0.2 for e in overridden.values())
