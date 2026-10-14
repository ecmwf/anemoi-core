# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Flat, neighbour-only (asynchronous) halo exchange against the legacy all_to_all implementation."""

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from distributed_runner import run_distributed_test
from test_halo import GRAPH_CASES
from test_halo import _make_graph

from anemoi.models.distributed import halo
from anemoi.models.distributed.graph import _HaloExchangeParallelSection
from anemoi.models.distributed.graph import halo_exchange
from anemoi.models.distributed.graph import halo_exchange_start
from anemoi.models.distributed.graph import halo_exchange_wait


def _legacy_halo_exchange(x: torch.Tensor, info: halo.HaloInfo, group: dist.ProcessGroup) -> torch.Tensor:
    return _HaloExchangeParallelSection.apply(x, info, group)


def _test_halo_exchange_rank(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
    mode: str,
) -> None:
    with patch.dict(os.environ, {"ANEMOI_HALO_COMM": mode}):
        for case in GRAPH_CASES:
            partition, edge_index = _make_graph(case, device)
            if partition.num_parts != world_size:
                continue
            build = halo.build_halo_info if partition.src_splits is None else halo.build_halo_info_bipartite
            info = build(partition, edge_index, group)
            assert rank not in info.peers
            assert info.num_send_nodes == sum(info.send_counts)

            torch.manual_seed(rank)
            x = torch.randn(info.num_local_src_nodes, 4, device=device, requires_grad=True)
            grad_out = torch.randn(info.total_src_nodes, 4, device=device)

            expected = _legacy_halo_exchange(x, info, group)
            (expected * grad_out).sum().backward()
            expected_grad = x.grad.clone()
            x.grad = None

            # synchronous wrapper
            actual = halo_exchange(x, info, group)
            torch.testing.assert_close(actual, expected)
            (actual * grad_out).sum().backward()
            torch.testing.assert_close(x.grad, expected_grad)
            x.grad = None

            # split start/wait with independent local work in between
            handle = halo_exchange_start(x, info, group)
            local = 2.0 * x
            halo_rows = halo_exchange_wait(handle)
            actual = torch.cat([local, halo_rows]) - torch.cat([x, torch.zeros_like(halo_rows)])
            torch.testing.assert_close(actual, expected)
            (actual * grad_out).sum().backward()
            torch.testing.assert_close(x.grad, expected_grad)
            x.grad = None

            # reduced wire dtype: halo rows come back in that dtype, gradients in the input dtype
            handle = halo_exchange_start(x, info, group, comm_dtype=torch.float64)
            halo_rows = halo_exchange_wait(handle)
            assert halo_rows.dtype == torch.float64
            torch.testing.assert_close(halo_rows.float(), expected[info.num_local_src_nodes :])
            (halo_rows.float() * grad_out[info.num_local_src_nodes :]).sum().backward()
            assert x.grad.dtype == x.dtype


@pytest.mark.distributed
@pytest.mark.parametrize("mode", ["p2p", "a2a"])
def test_halo_exchange_matches_legacy(distributed_backend: str, mode: str) -> None:
    if distributed_backend == "gloo" and mode == "a2a":
        pytest.skip("all_to_all_single with uneven splits is not supported by gloo")
    run_distributed_test(_test_halo_exchange_rank, backend=distributed_backend, world_size=3, mode=mode)
