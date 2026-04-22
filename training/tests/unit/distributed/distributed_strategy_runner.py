# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.models import AnemoiDiffusionModelEncProcDec
from anemoi.models.models import AnemoiEnsModelEncProcDec
from anemoi.training.distributed.groups import build_ensemble_layout
from anemoi.training.distributed.groups import build_model_layout
from anemoi.training.distributed.groups import create_ensemble_process_groups
from anemoi.training.distributed.strategy import register_gradient_scaling_hooks

GLOBAL_DEFAULT_ATOL = 1e-5
GLOBAL_DEFAULT_RTOL = 1e-5


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    msg = f"Invalid value for {name}: {raw!r}. Use one of 1/0, true/false, yes/no, on/off."
    raise ValueError(msg)


def _float32_matmul_precision(default: str = "highest") -> str:
    raw = os.getenv("ANEMOI_DISTRIBUTED_TEST_PRECISION", default)
    value = raw.strip().lower()
    if value not in {"highest", "high", "medium"}:
        msg = f"Invalid value for ANEMOI_DISTRIBUTED_TEST_PRECISION: {raw!r}. Use one of: highest, high, medium."
        raise ValueError(msg)
    return value


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None

    try:
        value = float(raw)
    except ValueError as exc:
        msg = f"Invalid value for {name}: {raw!r}. Use a non-negative float."
        raise ValueError(msg) from exc

    if value < 0.0:
        msg = f"Invalid value for {name}: {raw!r}. Use a non-negative float."
        raise ValueError(msg)

    return value


def _tolerances() -> tuple[float, float]:
    atol = _env_float("ANEMOI_DISTRIBUTED_TEST_ATOL")
    rtol = _env_float("ANEMOI_DISTRIBUTED_TEST_RTOL")
    return (
        GLOBAL_DEFAULT_ATOL if atol is None else atol,
        GLOBAL_DEFAULT_RTOL if rtol is None else rtol,
    )


def _configure_numeric_settings(*, deterministic: bool, matmul_precision: str) -> None:
    torch.set_float32_matmul_precision(matmul_precision)

    if not deterministic:
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_edge_data(
    src_size: int,
    dst_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.arange(src_size, device=device).repeat_interleave(dst_size)
    dst = torch.arange(dst_size, device=device).repeat(src_size)
    edge_index = torch.stack((src, dst), dim=0)
    edge_attr = ((src.float() + 1.0) / (src_size + 1.0) + (dst.float() + 1.0) / (dst_size + 1.0)).unsqueeze(-1)
    return edge_index, edge_attr


def _build_tiny_graph(device: torch.device) -> HeteroData:
    num_data_nodes = 11
    num_hidden_nodes = 11

    graph = HeteroData()
    coords_data = torch.linspace(-1.0, 1.0, num_data_nodes, device=device)
    coords_hidden = torch.linspace(-0.9, 0.9, num_hidden_nodes, device=device)
    graph["data"].x = torch.stack((coords_data, coords_data.flip(0)), dim=1)
    graph["hidden"].x = torch.stack((coords_hidden, coords_hidden.flip(0)), dim=1)

    edge_index_dh, edge_attr_dh = _build_edge_data(
        num_data_nodes,
        num_hidden_nodes,
        device,
    )
    edge_index_hh, edge_attr_hh = _build_edge_data(
        num_hidden_nodes,
        num_hidden_nodes,
        device,
    )
    edge_index_hd, edge_attr_hd = _build_edge_data(
        num_hidden_nodes,
        num_data_nodes,
        device,
    )

    graph[("data", "to", "hidden")].edge_index = edge_index_dh
    graph[("data", "to", "hidden")].edge_attr = edge_attr_dh
    graph[("hidden", "to", "hidden")].edge_index = edge_index_hh
    graph[("hidden", "to", "hidden")].edge_attr = edge_attr_hh
    graph[("hidden", "to", "data")].edge_index = edge_index_hd
    graph[("hidden", "to", "data")].edge_attr = edge_attr_hd

    return graph


def _training_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "src" / "anemoi" / "training" / "config"


def _build_tiny_diffusion_config(num_heads: int) -> dict:
    overrides = [
        "model=graphtransformer_diffusion",
        "training.multistep_input=1",
        "training.multistep_output=1",
        "model.num_channels=16",
        "model.bounding=[]",
        "model.trainable_parameters.hidden=1",
        "model.attributes.edges=[edge_attr]",
        "model.processor.trainable_size=0",
        "model.processor.num_layers=1",
        "model.processor.num_chunks=1",
        f"model.processor.num_heads={num_heads}",
        "model.processor.mlp_hidden_ratio=2",
        "model.processor.qk_norm=False",
        "model.processor.graph_attention_backend=pyg",
        "model.encoder.trainable_size=0",
        "model.encoder.num_chunks=1",
        f"model.encoder.num_heads={num_heads}",
        "model.encoder.mlp_hidden_ratio=2",
        "model.encoder.qk_norm=False",
        "model.encoder.shard_strategy=heads",
        "model.encoder.graph_attention_backend=pyg",
        "model.decoder.trainable_size=0",
        "model.decoder.num_chunks=1",
        f"model.decoder.num_heads={num_heads}",
        "model.decoder.mlp_hidden_ratio=2",
        "model.decoder.qk_norm=False",
        "model.decoder.shard_strategy=heads",
        "model.decoder.graph_attention_backend=pyg",
    ]

    if 16 % num_heads != 0:
        # Keep q/k/v projection width divisible by the requested head count.
        attn_channels = num_heads * 3
        overrides.extend(
            [
                f"+model.processor.attn_channels={attn_channels}",
                f"+model.encoder.attn_channels={attn_channels}",
                f"+model.decoder.attn_channels={attn_channels}",
            ],
        )

    with initialize_config_dir(
        version_base=None,
        config_dir=str(_training_config_dir()),
        job_name="test_diffusion_strategy_runner",
    ):
        cfg = compose(config_name="diffusion", overrides=overrides)

    OmegaConf.resolve(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def _is_skip_scaled(name: str) -> bool:
    return ("trainable" in name) or ("no_gradscaling" in name)


def _build_tiny_ensemble_config(num_heads: int) -> dict:
    overrides = [
        "graph=multi_scale",
        "model=graphtransformer_ens",
        "training.multistep_input=1",
        "training.multistep_output=1",
        "model.num_channels=16",
        "model.bounding=[]",
        "model.trainable_parameters.hidden=1",
        "model.attributes.edges=[edge_attr]",
        "model.noise_injector.noise_std=0",
        "model.noise_injector.noise_channels_dim=4",
        "model.noise_injector.noise_mlp_hidden_dim=8",
        "model.processor.trainable_size=0",
        "model.processor.num_layers=1",
        "model.processor.num_chunks=1",
        f"model.processor.num_heads={num_heads}",
        "model.processor.mlp_hidden_ratio=2",
        "model.processor.qk_norm=False",
        "model.processor.graph_attention_backend=pyg",
        "model.processor.edge_pre_mlp=False",
        "model.encoder.trainable_size=0",
        "model.encoder.num_chunks=1",
        f"model.encoder.num_heads={num_heads}",
        "model.encoder.mlp_hidden_ratio=2",
        "model.encoder.qk_norm=False",
        "model.encoder.shard_strategy=heads",
        "model.encoder.graph_attention_backend=pyg",
        "model.encoder.edge_pre_mlp=False",
        "model.decoder.trainable_size=0",
        "model.decoder.num_chunks=1",
        f"model.decoder.num_heads={num_heads}",
        "model.decoder.mlp_hidden_ratio=2",
        "model.decoder.qk_norm=False",
        "model.decoder.shard_strategy=heads",
        "model.decoder.graph_attention_backend=pyg",
        "model.decoder.edge_pre_mlp=False",
    ]

    with initialize_config_dir(
        version_base=None,
        config_dir=str(_training_config_dir()),
        job_name="test_ensemble_strategy_runner",
    ):
        cfg = compose(config_name="ensemble_crps", overrides=overrides)

    OmegaConf.resolve(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def _compute_reference_grads(
    model: AnemoiDiffusionModelEncProcDec,
    dataset_name: str,
    x_full: torch.Tensor,
    y_noised_full: torch.Tensor,
    sigma: torch.Tensor,
    target_full: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    out_full = model.fwd_with_preconditioning(
        {dataset_name: x_full},
        {dataset_name: y_noised_full},
        {dataset_name: sigma},
        model_comm_group=None,
        grid_shard_shapes=None,
    )[dataset_name]
    denom = float(target_full.numel())
    loss = ((out_full - target_full) ** 2).sum() / denom
    loss.backward()
    grads = {name: (None if p.grad is None else p.grad.detach().clone()) for name, p in model.named_parameters()}
    return out_full.detach(), grads


def _compute_sharded_grads(
    rank: int,
    world_size: int,
    model: AnemoiDiffusionModelEncProcDec,
    dataset_name: str,
    x_full: torch.Tensor,
    y_noised_full: torch.Tensor,
    sigma: torch.Tensor,
    target_full: torch.Tensor,
    model_comm_group: dist.ProcessGroup,
    apply_hooks: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if apply_hooks:
        group_size = float(model_comm_group.size()) if model_comm_group is not None else 1.0
        register_gradient_scaling_hooks(model, model_comm_group_size=group_size)

    model.zero_grad(set_to_none=True)
    split_sizes = get_balanced_partition_sizes(x_full.shape[3], world_size)
    x_local = torch.split(x_full, split_sizes, dim=3)[rank].contiguous()
    y_noised_local = torch.split(y_noised_full, split_sizes, dim=3)[rank].contiguous()
    target_local = torch.split(target_full, split_sizes, dim=3)[rank].contiguous()

    out_local = model.fwd_with_preconditioning(
        {dataset_name: x_local},
        {dataset_name: y_noised_local},
        {dataset_name: sigma},
        model_comm_group=model_comm_group,
        grid_shard_shapes={dataset_name: split_sizes},
    )[dataset_name]
    denom = float(target_full.numel())
    loss_local = ((out_local - target_local) ** 2).sum() / denom
    loss_local.backward()

    out_shapes = [list(out_local.shape) for _ in split_sizes]
    for shard_rank, shard_size in enumerate(split_sizes):
        out_shapes[shard_rank][3] = shard_size
    out_full = gather_tensor(
        out_local,
        dim=3,
        shapes=out_shapes,
        mgroup=model_comm_group,
    ).detach()

    grads_avg: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        grad = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param, device=param.device)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=model_comm_group)
        grad /= float(world_size)  # emulate DDP gradient averaging
        grads_avg[name] = grad
    return out_full, grads_avg


def _run_diffusion_gradient_scaling_parity(
    rank: int,
    world_size: int,
    device: torch.device,
    model_comm_group: dist.ProcessGroup,
) -> None:
    dataset_name = "data"
    num_heads = 3
    num_vars = 2
    if num_heads < world_size:
        msg = f"Diffusion parity singleton-head case supports at most {num_heads} ranks, got world_size={world_size}."
        raise ValueError(msg)

    out_atol, out_rtol = _tolerances()
    grad_atol, grad_rtol = _tolerances()

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    graph_data = _build_tiny_graph(device=device)
    data_indices = {
        dataset_name: IndexCollection(
            data_config=OmegaConf.create(
                {"forcing": [], "diagnostic": [], "target": []},
            ),
            name_to_index={f"var{i}": i for i in range(num_vars)},
        ),
    }
    model_config = _build_tiny_diffusion_config(num_heads=num_heads)
    statistics = {dataset_name: None}

    ref_model = AnemoiDiffusionModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    ).to(device)
    sharded_hooked_model = AnemoiDiffusionModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    ).to(device)
    sharded_unhooked_model = AnemoiDiffusionModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    ).to(device)
    sharded_hooked_model.load_state_dict(ref_model.state_dict(), strict=True)
    sharded_unhooked_model.load_state_dict(ref_model.state_dict(), strict=True)

    ref_model.train()
    sharded_hooked_model.train()
    sharded_unhooked_model.train()

    num_nodes = graph_data["data"].x.shape[0]
    x_full = (
        torch.arange(num_nodes * num_vars, dtype=torch.float32, device=device).reshape(
            1,
            1,
            1,
            num_nodes,
            num_vars,
        )
        / 19.0
    )
    y_noised_full = torch.cos(x_full * 0.7)
    sigma = torch.full((1, 1, 1, 1, 1), 0.25, dtype=torch.float32, device=device)
    target_full = torch.sin(x_full * 0.5 + y_noised_full * 0.3)

    ref_out, ref_grads = _compute_reference_grads(
        ref_model,
        dataset_name,
        x_full,
        y_noised_full,
        sigma,
        target_full,
    )
    hooked_out, hooked_grads_avg = _compute_sharded_grads(
        rank,
        world_size,
        sharded_hooked_model,
        dataset_name,
        x_full,
        y_noised_full,
        sigma,
        target_full,
        model_comm_group,
        apply_hooks=True,
    )
    unhooked_out, unhooked_grads_avg = _compute_sharded_grads(
        rank,
        world_size,
        sharded_unhooked_model,
        dataset_name,
        x_full,
        y_noised_full,
        sigma,
        target_full,
        model_comm_group,
        apply_hooks=False,
    )

    torch.testing.assert_close(hooked_out, ref_out, atol=out_atol, rtol=out_rtol)
    torch.testing.assert_close(unhooked_out, ref_out, atol=out_atol, rtol=out_rtol)

    found_unhooked_mismatch = False
    for name, ref_grad in ref_grads.items():
        assert ref_grad is not None, f"Expected gradient for parameter '{name}' in reference model."
        torch.testing.assert_close(
            hooked_grads_avg[name],
            ref_grad,
            atol=grad_atol,
            rtol=grad_rtol,
        )

        if (not _is_skip_scaled(name)) and (
            not torch.allclose(
                unhooked_grads_avg[name],
                ref_grad,
                atol=grad_atol,
                rtol=grad_rtol,
            )
        ):
            found_unhooked_mismatch = True

    if world_size > 1:
        assert found_unhooked_mismatch, "Expected at least one non-skipped parameter to require gradient scaling."


def _compute_ensemble_mode(
    *,
    rank: int,
    world_size: int,
    model: AnemoiEnsModelEncProcDec,
    dataset_name: str,
    x_two_members: torch.Tensor,
    target_two_members: torch.Tensor,
    ens_comm_group_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model_layout = build_model_layout(
        world_size=world_size,
        global_rank=rank,
        model_comm_group_size=1,
    )
    ens_layout = build_ensemble_layout(
        world_size=world_size,
        global_rank=rank,
        ens_comm_group_size=ens_comm_group_size,
        model_comm_group_size=1,
        model_comm_group_rank=model_layout.model_comm_group_rank,
    )
    ens_groups = create_ensemble_process_groups(ens_layout)

    if ens_layout.ens_comm_subgroup_size == 1:
        x_local = x_two_members
    else:
        i0 = ens_layout.ens_comm_subgroup_rank
        x_local = x_two_members[:, :, i0 : i0 + 1, :, :].contiguous()

    model.zero_grad(set_to_none=True)
    y_pred_local = model(
        {dataset_name: x_local},
        fcstep=0,
        model_comm_group=None,
        grid_shard_shapes=None,
    )[dataset_name]
    gather_shapes = [list(y_pred_local.shape) for _ in range(ens_layout.ens_comm_subgroup_size)]
    y_pred_ens = gather_tensor(
        y_pred_local,
        dim=2,
        shapes=gather_shapes,
        mgroup=ens_groups.ens_comm_subgroup,
    )
    loss = ((y_pred_ens - target_two_members) ** 2).sum() / float(
        target_two_members.numel(),
    )
    loss.backward()

    grads_avg: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        grad = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param, device=param.device)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ens_groups.ens_comm_subgroup)
        grads_avg[name] = grad
    return y_pred_ens.detach(), grads_avg


def _run_ensemble_partitioning_parity(
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    if world_size % 2 != 0:
        msg = f"Ensemble parity requires an even world size, got {world_size}."
        raise ValueError(msg)

    parity_atol, parity_rtol = _tolerances()

    dataset_name = "data"
    num_heads = 8
    num_vars = 2
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    graph_data = _build_tiny_graph(device=device)
    data_indices = {
        dataset_name: IndexCollection(
            data_config=OmegaConf.create(
                {"forcing": [], "diagnostic": [], "target": []},
            ),
            name_to_index={f"var{i}": i for i in range(num_vars)},
        ),
    }
    model_config = _build_tiny_ensemble_config(num_heads=num_heads)
    statistics = {dataset_name: None}

    model_local_ens = AnemoiEnsModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    ).to(device)
    model_dist_ens = AnemoiEnsModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    ).to(device)
    model_dist_ens.load_state_dict(model_local_ens.state_dict(), strict=True)
    model_local_ens.train()
    model_dist_ens.train()

    num_nodes = graph_data["data"].x.shape[0]
    x_member0 = (
        torch.arange(num_nodes * num_vars, dtype=torch.float32, device=device).reshape(
            1,
            1,
            num_nodes,
            num_vars,
        )
        / 17.0
    )
    x_member1 = torch.cos(x_member0 * 0.8 + 0.3)
    x_two_members = torch.stack(
        (x_member0, x_member1),
        dim=2,
    )  # (batch, time, ens=2, grid, vars)
    target_two_members = torch.sin(x_two_members * 0.35 + 0.1)

    out_local_ens, grads_local_ens = _compute_ensemble_mode(
        rank=rank,
        world_size=world_size,
        model=model_local_ens,
        dataset_name=dataset_name,
        x_two_members=x_two_members,
        target_two_members=target_two_members,
        ens_comm_group_size=1,
    )
    out_dist_ens, grads_dist_ens = _compute_ensemble_mode(
        rank=rank,
        world_size=world_size,
        model=model_dist_ens,
        dataset_name=dataset_name,
        x_two_members=x_two_members,
        target_two_members=target_two_members,
        ens_comm_group_size=2,
    )

    torch.testing.assert_close(
        out_local_ens,
        out_dist_ens,
        atol=parity_atol,
        rtol=parity_rtol,
    )
    for name, grad_local in grads_local_ens.items():
        torch.testing.assert_close(
            grads_dist_ens[name],
            grad_local,
            atol=parity_atol,
            rtol=parity_rtol,
        )


def _compute_ensemble_mode_world_step(
    *,
    rank: int,
    world_size: int,
    model: AnemoiEnsModelEncProcDec,
    dataset_name: str,
    x_two_members: torch.Tensor,
    target_two_members: torch.Tensor,
    ens_comm_group_size: int,
    base_lr: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model_layout = build_model_layout(
        world_size=world_size,
        global_rank=rank,
        model_comm_group_size=1,
    )
    ens_layout = build_ensemble_layout(
        world_size=world_size,
        global_rank=rank,
        ens_comm_group_size=ens_comm_group_size,
        model_comm_group_size=1,
        model_comm_group_rank=model_layout.model_comm_group_rank,
    )
    ens_groups = create_ensemble_process_groups(ens_layout)

    if ens_layout.ens_comm_subgroup_size == 1:
        x_local = x_two_members
    else:
        i0 = ens_layout.ens_comm_subgroup_rank
        x_local = x_two_members[:, :, i0 : i0 + 1, :, :].contiguous()

    model.zero_grad(set_to_none=True)
    y_pred_local = model(
        {dataset_name: x_local},
        fcstep=0,
        model_comm_group=None,
        grid_shard_shapes=None,
    )[dataset_name]
    gather_shapes = [list(y_pred_local.shape) for _ in range(ens_layout.ens_comm_subgroup_size)]
    y_pred_ens = gather_tensor(
        y_pred_local,
        dim=2,
        shapes=gather_shapes,
        mgroup=ens_groups.ens_comm_subgroup,
    )
    loss = ((y_pred_ens - target_two_members) ** 2).sum() / float(
        target_two_members.numel(),
    )
    loss.backward()

    # Emulate DDP world gradient averaging and effective LR scaling from ensemble training.
    # In GraphEnsForecaster: lr ~ (... * num_gpus_per_node) / num_gpus_per_ensemble.
    effective_lr = base_lr * float(world_size) / float(ens_comm_group_size)
    deltas: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            grad = (
                param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param, device=param.device)
            )
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            grad /= float(world_size)
            delta = effective_lr * grad
            param.sub_(delta)
            deltas[name] = delta
    return y_pred_ens.detach(), deltas


def _run_ensemble_world_step_scaling_parity(
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    if world_size % 2 != 0:
        msg = f"Ensemble world-step parity requires an even world size, got {world_size}."
        raise ValueError(msg)

    out_atol, out_rtol = _tolerances()
    raw_mismatch_atol, raw_mismatch_rtol = _tolerances()
    scaled_delta_atol, scaled_delta_rtol = _tolerances()

    dataset_name = "data"
    num_heads = 8
    num_vars = 2
    base_lr = 1e-3
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    graph_data = _build_tiny_graph(device=device)
    data_indices = {
        dataset_name: IndexCollection(
            data_config=OmegaConf.create(
                {"forcing": [], "diagnostic": [], "target": []},
            ),
            name_to_index={f"var{i}": i for i in range(num_vars)},
        ),
    }
    model_config = _build_tiny_ensemble_config(num_heads=num_heads)
    statistics = {dataset_name: None}

    model_local_ens = AnemoiEnsModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    ).to(device)
    model_dist_ens = AnemoiEnsModelEncProcDec(
        model_config=model_config,
        data_indices=data_indices,
        statistics=statistics,
        graph_data=graph_data,
    ).to(device)
    model_dist_ens.load_state_dict(model_local_ens.state_dict(), strict=True)
    model_local_ens.train()
    model_dist_ens.train()

    num_nodes = graph_data["data"].x.shape[0]
    x_member0 = (
        torch.arange(num_nodes * num_vars, dtype=torch.float32, device=device).reshape(
            1,
            1,
            num_nodes,
            num_vars,
        )
        / 13.0
    )
    x_member1 = torch.sin(x_member0 * 0.6 + 0.2)
    x_two_members = torch.stack((x_member0, x_member1), dim=2)
    target_two_members = torch.cos(x_two_members * 0.4 + 0.1)

    out_local_ens, delta_local = _compute_ensemble_mode_world_step(
        rank=rank,
        world_size=world_size,
        model=model_local_ens,
        dataset_name=dataset_name,
        x_two_members=x_two_members,
        target_two_members=target_two_members,
        ens_comm_group_size=1,
        base_lr=base_lr,
    )
    out_dist_ens, delta_dist = _compute_ensemble_mode_world_step(
        rank=rank,
        world_size=world_size,
        model=model_dist_ens,
        dataset_name=dataset_name,
        x_two_members=x_two_members,
        target_two_members=target_two_members,
        ens_comm_group_size=2,
        base_lr=base_lr,
    )

    torch.testing.assert_close(
        out_local_ens,
        out_dist_ens,
        atol=out_atol,
        rtol=out_rtol,
    )

    found_nonzero_delta = False
    found_raw_mismatch = False
    for name, delta_ref in delta_local.items():
        if torch.count_nonzero(delta_ref).item() > 0:
            found_nonzero_delta = True
        if not torch.allclose(
            delta_dist[name],
            delta_ref,
            atol=raw_mismatch_atol,
            rtol=raw_mismatch_rtol,
        ):
            found_raw_mismatch = True

        # Expected scaling relation for this topology:
        # world-averaged gradients differ by subgroup split and LR differs by ens group size.
        torch.testing.assert_close(
            delta_dist[name] * 4.0,
            delta_ref,
            atol=scaled_delta_atol,
            rtol=scaled_delta_rtol,
        )

    assert found_nonzero_delta, "Expected at least one parameter to receive a non-zero optimizer update."
    assert found_raw_mismatch, "Expected raw world-step deltas to differ between the two ensemble layouts."


def _run_rank(
    rank: int,
    world_size: int,
    backend: str,
    init_file: str,
    suite: str,
    deterministic: bool,
    matmul_precision: str,
) -> None:
    _configure_numeric_settings(
        deterministic=deterministic,
        matmul_precision=matmul_precision,
    )

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        if suite == "diffusion":
            _run_diffusion_gradient_scaling_parity(
                rank,
                world_size,
                device,
                dist.group.WORLD,
            )
        elif suite == "ensemble":
            _run_ensemble_partitioning_parity(rank, world_size, device)
        elif suite == "ensemble_world_step":
            _run_ensemble_world_step_scaling_parity(rank, world_size, device)
        else:
            msg = f"Unknown suite '{suite}'."
            raise ValueError(msg)
        dist.barrier(group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distributed strategy gradient parity runner",
    )
    parser.add_argument("--backend", choices=["gloo", "nccl"], required=True)
    parser.add_argument(
        "--suite",
        choices=["diffusion", "ensemble", "ensemble_world_step"],
        default="diffusion",
    )
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    deterministic = _env_flag("ANEMOI_DISTRIBUTED_TEST_DETERMINISTIC", default=True)
    matmul_precision = _float32_matmul_precision(default="highest")

    if deterministic and args.backend == "nccl":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if args.world_size < 2:
        msg = f"world-size must be >= 2, got {args.world_size}"
        raise ValueError(msg)

    if args.backend == "nccl" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        msg = f"NCCL backend requires at least {args.world_size} CUDA devices, found {torch.cuda.device_count()}."
        raise RuntimeError(msg)

    fd, path = tempfile.mkstemp(prefix="anemoi_train_dist_init_", suffix=".tmp")
    os.close(fd)
    init_file = Path(path)
    try:
        mp.spawn(
            _run_rank,
            args=(
                args.world_size,
                args.backend,
                str(init_file),
                args.suite,
                deterministic,
                matmul_precision,
            ),
            nprocs=args.world_size,
            join=True,
        )
    finally:
        init_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
