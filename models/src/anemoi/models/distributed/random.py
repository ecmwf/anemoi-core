# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextlib import contextmanager
from dataclasses import dataclass

import torch

_MAX_SEED = 2**32 - 1
_SEED_MODULUS = _MAX_SEED + 1
_RANK_SEED_STRIDE = 1_000_003
_SYNC_GROUP_SEED_STRIDE = 10_000_019


@dataclass
class TorchRNGState:
    """Saved PyTorch random state for CPU and any CUDA devices already in use."""

    cpu: torch.Tensor
    cuda: dict[int, torch.Tensor]


_synced_seed: int | None = None
_synced_state: TorchRNGState | None = None
_synced_context_depth = 0


def _normalize_seed(seed: int) -> int:
    """Wrap a seed into the range accepted by Lightning, NumPy, and PyTorch."""
    return int(seed) % _SEED_MODULUS


def get_independent_torch_seed(base_seed: int, global_rank: int) -> int:
    """Return the default PyTorch seed for one rank.

    This seed is different for each global rank. It is used by ordinary PyTorch
    random calls, for example dropout.
    """
    return _normalize_seed(base_seed + _RANK_SEED_STRIDE * (global_rank + 1))


def get_synced_torch_seed(base_seed: int, sync_group_id: int = 0) -> int:
    """Return the seed for the synced PyTorch random stream.

    Ranks that pass the same ``sync_group_id`` get the same synced stream.
    Different model groups use different ``sync_group_id`` values, so their
    synced streams do not overlap.
    """
    return _normalize_seed(base_seed + _SYNC_GROUP_SEED_STRIDE * sync_group_id)


def _cuda_is_initialized() -> bool:
    """Return whether CUDA random state can be read without starting CUDA."""
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def _get_cuda_rng_state() -> dict[int, torch.Tensor]:
    """Save CUDA random state for every visible CUDA device already in use."""
    if not _cuda_is_initialized():
        return {}
    return {device: torch.cuda.get_rng_state(device).clone() for device in range(torch.cuda.device_count())}


def _get_current_rng_state() -> TorchRNGState:
    """Save the current PyTorch random state."""
    return TorchRNGState(cpu=torch.get_rng_state().clone(), cuda=_get_cuda_rng_state())


def _clone_rng_state(state: TorchRNGState) -> TorchRNGState:
    """Copy a saved random state so later changes cannot modify the original."""
    return TorchRNGState(
        cpu=state.cpu.clone(),
        cuda={device: device_state.clone() for device, device_state in state.cuda.items()},
    )


def _set_current_rng_state(state: TorchRNGState) -> None:
    """Replace PyTorch's current random state with a saved state."""
    torch.set_rng_state(state.cpu)
    for device, device_state in state.cuda.items():
        torch.cuda.set_rng_state(device_state, device=device)


def _make_rng_state(seed: int) -> TorchRNGState:
    """Create a saved PyTorch random state for ``seed`` without changing the caller's state."""
    # Save the current stream so this function leaves it unchanged.
    current_state = _get_current_rng_state()

    # Seed PyTorch, save the fresh state, then put the caller's state back.
    torch.manual_seed(seed)
    if _cuda_is_initialized():
        torch.cuda.manual_seed_all(seed)
    rng_state = _get_current_rng_state()
    _set_current_rng_state(current_state)
    return rng_state


def _ensure_synced_state() -> None:
    """Check that the synced stream exists and add CUDA devices if needed."""
    if _synced_seed is None or _synced_state is None:
        msg = (
            "Synchronized Torch RNG source has not been initialized. "
            "Call seed_torch_rng_sources(...) before use_synced_torch_rng()."
        )
        raise RuntimeError(msg)

    # CUDA can be started after setup. When that happens, add matching CUDA
    # state to the synced stream before the first synced CUDA draw.
    current_cuda_devices = set(_get_cuda_rng_state())
    missing_cuda_devices = current_cuda_devices.difference(_synced_state.cuda)
    if missing_cuda_devices:
        new_state = _make_rng_state(_synced_seed)
        for device in missing_cuda_devices:
            _synced_state.cuda[device] = new_state.cuda[device]


def seed_torch_rng_sources(
    base_seed: int,
    global_rank: int,
    *,
    sync_group_id: int = 0,
    seed_default: bool = True,
    reset_synced: bool = False,
) -> int:
    """Set up the default and synced PyTorch random streams.

    The default stream is seeded differently on every rank. The synced stream is
    kept separately and is only used inside ``use_synced_torch_rng``.

    Parameters
    ----------
    base_seed : int
        Base seed used to derive the rank-local and synced seeds.
    global_rank : int
        Global rank used to derive the default rank-local seed.
    sync_group_id : int, optional
        Group id used to derive the synced seed.
    seed_default : bool, optional
        If True, seed PyTorch's default random stream.
    reset_synced : bool, optional
        If True, restart the synced stream from its seed.

    Returns
    -------
    int
        Seed used for the default stream on this rank.
    """
    global _synced_seed, _synced_state

    # Ordinary model randomness, such as dropout, should differ between ranks.
    independent_seed = get_independent_torch_seed(base_seed, global_rank)

    # Synced model randomness, such as model initialisation, should match within
    # one model group and differ across different model groups.
    synced_seed = get_synced_torch_seed(base_seed, sync_group_id)

    if reset_synced or _synced_seed != synced_seed or _synced_state is None:
        _synced_seed = synced_seed
        _synced_state = _make_rng_state(synced_seed)

    if seed_default:
        torch.manual_seed(independent_seed)

    return independent_seed


def synced_torch_rng_checkpoint_context() -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
    """Return activation checkpoint helpers that preserve the synced random stream.

    PyTorch activation checkpointing already saves and restores PyTorch's own
    random state. The synced stream also has state stored in this module, so we
    save and restore that state for the checkpoint's second pass.
    """
    forward_seed: int | None = None
    forward_state: TorchRNGState | None = None

    @contextmanager
    def forward_context() -> Iterator[None]:
        """Save the synced stream before the original checkpointed forward pass."""
        nonlocal forward_seed, forward_state
        if _synced_seed is not None and _synced_state is not None:
            forward_seed = _synced_seed
            forward_state = _clone_rng_state(_synced_state)
        yield

    @contextmanager
    def recompute_context() -> Iterator[None]:
        """Use the saved synced stream during the checkpoint's second pass."""
        global _synced_seed, _synced_state

        if forward_state is None:
            yield
            return

        # Keep the real synced stream reached by the original forward pass.
        current_seed = _synced_seed
        current_state = _clone_rng_state(_synced_state) if _synced_state is not None else None
        try:
            # The checkpoint's second pass must see the same synced stream that
            # the original forward pass saw.
            _synced_seed = forward_seed
            _synced_state = _clone_rng_state(forward_state)
            yield
        finally:
            # Put back the real synced stream so checkpointing does not consume
            # synced random numbers twice.
            _synced_seed = current_seed
            _synced_state = current_state

    return forward_context(), recompute_context()


@contextmanager
def use_synced_torch_rng() -> Iterator[None]:
    """Temporarily draw PyTorch random numbers from the synced stream.

    Outside this block, PyTorch uses the default rank-local stream. Inside this
    block, PyTorch uses the synced stream. When the block exits, the synced
    stream keeps the progress made inside the block and the default stream is
    restored.
    """
    global _synced_context_depth, _synced_state

    _ensure_synced_state()

    if _synced_context_depth > 0:
        # We are already using the synced stream. Keep using it and only let the
        # outer block save the new synced state when all nested blocks have left.
        _synced_context_depth += 1
        try:
            yield
        finally:
            _synced_context_depth -= 1
        return

    # Save the default rank-local stream, then temporarily install the synced one.
    default_state = _get_current_rng_state()
    _set_current_rng_state(_synced_state)
    _synced_context_depth = 1
    try:
        yield
    finally:
        # Save how far the synced stream moved, then restore the rank-local stream.
        _synced_state = _get_current_rng_state()
        _synced_context_depth = 0
        _set_current_rng_state(default_state)
