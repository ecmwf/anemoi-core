# (C) Copyright 2025 Anemoi contributors.
# (C) Copyright 2026 2026 NVIDIA CORPORATION & AFFILIATES.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from typing import Tuple

# check if triton is installed
# If pytorch is installed on CPU then torch is not available
try:
    import triton
    import triton.language as tl
    from anemoi.models.triton.conversion_utils import fast_edge_index_to_csc
except ImportError:
    msg =  "Error. The 'triton' backend was selected for the GraphTransformer but Triton is not installed. "
    msg += "To use this backend please install Triton. Otherwise, select a different backend for the GraphTransformer in the models config."
    raise ValueError(msg)


@triton.jit
def _graphtransformer_fwd_kernel(
    Q_ptr,  # [N_dst, H, C]
    K_ptr,  # [N_src, H, C]
    V_ptr,  # [N_src, H, C]
    E_ptr,  # [E, H, C]
    CSC_offsets_ptr,  # [N_dst+1]
    CSC_indices_ptr,  # [E]
    CSC_map_to_coo_ptr, # [E]
    M_ptr,  # [N_dst, H]
    OUT_ptr,  # [N_dst, H, C]
    H: tl.constexpr,
    H_PER_BLK: tl.constexpr,
    C: tl.constexpr,
    C_PAD: tl.constexpr,
    USE_FAST_EXP2: tl.constexpr,
):
    dst_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)
    # option: batch_idx = tl.program_id(2)

    H_mask = None
    H_C_mask = None
    H_PAD_VAL = None
    H_C_PAD_VAL = None

    H_off = head_idx * H_PER_BLK + tl.arange(0, H_PER_BLK)
    C_off = tl.arange(0, C_PAD)
    H_off = H_off[:, None]
    C_off = C_off[None, :]
    H_C_off = H_off * C + C_off

    if (H % H_PER_BLK == 0) and (C == C_PAD):
        H_mask = None
        H_C_mask = None
        H_PAD_VAL = None
        H_C_PAD_VAL = None
    elif H % H_PER_BLK == 0:
        H_mask = None
        C_mask = (C_off < C)
        H_C_mask = C_mask
        H_PAD_VAL = None
        H_C_PAD_VAL = 0.0
    else:
        H_mask = (H_off < H)
        C_mask = (C_off < C)
        H_C_mask = H_mask & C_mask
        H_PAD_VAL = 0.0
        H_C_PAD_VAL = 0.0

    acc = tl.zeros((H_PER_BLK, C_PAD), dtype=tl.float32)  # output accumulator, pending normalization by l_i
    l_i = tl.full((H_PER_BLK, 1), value=1.0, dtype=tl.float32)  # sum of attention weights
    m_i = tl.full((H_PER_BLK, 1), value=-float("inf"), dtype=tl.float32)  # running max for stability

    neigh_start = tl.load(CSC_offsets_ptr + dst_idx)
    neigh_end = tl.load(CSC_offsets_ptr + dst_idx + 1)

    key_base_ptrs = K_ptr + H_C_off
    val_base_ptrs = V_ptr + H_C_off
    e_base_ptrs = E_ptr + H_C_off

    dst_off = dst_idx * (H * C) + H_C_off
    q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32)

    if USE_FAST_EXP2:
        qk_scale: tl.constexpr = (1.0 / tl.sqrt(float(C))) * 1.44269504089
    else:
        qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    # for _ in tl.range(num_edges, warp_specialize=True):
    for eidx in range(neigh_start, neigh_end):
        src_idx = tl.load(CSC_indices_ptr + eidx)
        coo_idx = tl.load(CSC_map_to_coo_ptr + eidx)
        e_ptrs = e_base_ptrs + coo_idx * (H * C)
        e = tl.load(e_ptrs, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)

        src_off = src_idx * (H * C)
        key_ptrs = key_base_ptrs + src_off
        val_ptrs = val_base_ptrs + src_off

        k = tl.load(key_ptrs, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
        v = tl.load(val_ptrs, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)

        k_e = k + e
        v_e = v + e

        qk = tl.sum(q * k_e, axis=-1, keep_dims=True) * qk_scale  # Shape: [H_PER_BLK, 1]

        m_ij = tl.maximum(m_i, qk)  # new running max
        qk = qk - m_ij
        if USE_FAST_EXP2:
            p = tl.math.exp2(qk)
            alpha = tl.math.exp2(m_i - m_ij)
        else:
            p = tl.math.exp(qk)
            alpha = tl.math.exp(m_i - m_ij)

        # update accumulators with correction
        l_i = l_i * alpha + p
        acc = acc * alpha

        # add current contribution
        acc = acc + p * v_e

        m_i = m_ij

    # final normalization: divide by sum of attention weights
    acc = acc / l_i
    out_ptrs = OUT_ptr + dst_off
    tl.store(out_ptrs, acc, mask=H_C_mask)

    # store m_i + log(l_i) for backward
    m_ptrs = M_ptr + dst_idx * H + H_off
    if USE_FAST_EXP2:
        m_i = m_i + tl.math.log2(l_i)
    else:
        m_i = m_i + tl.log(l_i)

    tl.store(m_ptrs, m_i, mask=H_mask)


@triton.jit
def _graphtransformer_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    E_ptr,
    OUT_ptr,  # saved forward outputs o_i
    M_ptr,  # saved m_i + ln l_i
    D_OUT_ptr,  # [N_dst * H * C]
    CSC_offsets_ptr,  # [N_dst+1]
    CSC_indices_ptr,  # [E]
    CSC_map_to_coo_ptr, # [E]
    D_Q_ptr,  # OUT
    D_ptr,  # [N_dst * H]
    H: tl.constexpr,
    H_PER_BLK: tl.constexpr,
    C: tl.constexpr,
    C_PAD: tl.constexpr,
    USE_FAST_EXP2: tl.constexpr,
):
    dst_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)
    # option: batch_idx = tl.program_id(2)

    H_mask = None
    H_C_mask = None
    H_PAD_VAL = None
    H_C_PAD_VAL = None

    H_off = head_idx * H_PER_BLK + tl.arange(0, H_PER_BLK)
    C_off = tl.arange(0, C_PAD)
    H_off = H_off[:, None]
    C_off = C_off[None, :]
    H_C_off = H_off * C + C_off

    if (H % H_PER_BLK == 0) and (C == C_PAD):
        H_mask = None
        H_C_mask = None
        H_PAD_VAL = None
        H_C_PAD_VAL = None
    elif H % H_PER_BLK == 0:
        H_mask = None
        C_mask = (C_off < C)
        H_C_mask = C_mask
        H_PAD_VAL = None
        H_C_PAD_VAL = 0.0
    else:
        H_mask = (H_off < H)
        C_mask = (C_off < C)
        H_C_mask = H_mask & C_mask
        H_PAD_VAL = 0.0
        H_C_PAD_VAL = 0.0

    dst_off = dst_idx * (H * C) + H_C_off
    neigh_start = tl.load(CSC_offsets_ptr + dst_idx)
    neigh_end = tl.load(CSC_offsets_ptr + dst_idx + 1)

    dq = tl.zeros((H_PER_BLK, C_PAD), dtype=tl.float32)

    d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
    out = tl.load(OUT_ptr + dst_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
    q = tl.load(Q_ptr + dst_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
    m_i = tl.load(M_ptr + dst_idx * H + H_off, mask=H_mask, other=H_PAD_VAL).to(tl.float32)

    # D_j = <d_out, out> for one-pass computation of dQ
    Dj = tl.sum(d_out * out, axis=-1, keep_dims=True)  # [H, 1]

    key_base_ptrs = K_ptr + H_C_off
    val_base_ptrs = V_ptr + H_C_off
    e_base_ptrs = E_ptr + H_C_off

    qk_scale_gradq: tl.constexpr = 1.0 / tl.sqrt(float(C))
    if USE_FAST_EXP2:
        qk_scale: tl.constexpr = (1.0 / tl.sqrt(float(C))) * 1.44269504089
    else:
        qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for eidx in range(neigh_start, neigh_end):
        src_idx = tl.load(CSC_indices_ptr + eidx)
        src_off = src_idx * (H * C)
        coo_idx = tl.load(CSC_map_to_coo_ptr + eidx)

        # recompute score and alpha
        e_ptrs = e_base_ptrs + coo_idx * (H * C)
        e = tl.load(e_ptrs, mask=H_C_mask).to(tl.float32)
        k = tl.load(key_base_ptrs + src_off, mask=H_C_mask).to(tl.float32)
        ke = k + e
        s_ij = tl.sum(q * ke, axis=-1, keep_dims=True) * qk_scale  # [H_PER_BLK, 1]
        if USE_FAST_EXP2:
            alpha_ij = tl.math.exp2(s_ij - m_i)
        else:
            alpha_ij = tl.exp(s_ij - m_i)

        # propagate through softmax
        v = tl.load(val_base_ptrs + src_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
        ve = v + e
        dalpha = tl.sum(d_out * ve, axis=-1, keep_dims=True)  # [H_PER_BLK, 1]
        dS = alpha_ij * (dalpha - Dj)  # [H_PER_BLK, 1]

        # update gradient on query
        dq += dS * ke * qk_scale_gradq

    # store D_j and dQ
    d_ptrs = D_ptr + dst_idx * H + H_off
    tl.store(d_ptrs, Dj.to(Dj.type.element_ty), mask=H_mask)
    d_q_ptrs = D_Q_ptr + dst_off
    tl.store(d_q_ptrs, dq.to(D_Q_ptr.type.element_ty), mask=H_C_mask)


@triton.jit
def _graphtransformer_bwd_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    E_ptr,
    D_ptr,  # [N_dst * H] D_j from pass dst-pass
    M_ptr,  # [N_dst * H] saved m_j from fwd
    D_OUT_ptr,  # [N_dst * H * C]
    CSR_offsets_ptr,  # [N_src+1]
    CSR_indices_ptr,  # [E]
    CSR_map_to_coo_ptr, # [E]
    D_K_ptr,  # [N_src * H * C]
    D_V_ptr,  # [N_src * H * C]
    D_E_ptr,  # [M * H * C]
    H: tl.constexpr,
    H_PER_BLK: tl.constexpr,
    C: tl.constexpr,
    C_PAD: tl.constexpr,
    USE_FAST_EXP2: tl.constexpr,
):
    src_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)
    # option: batch_idx = tl.program_id(2)

    H_mask = None
    H_C_mask = None
    H_PAD_VAL = None
    H_C_PAD_VAL = None

    H_off = head_idx * H_PER_BLK + tl.arange(0, H_PER_BLK)
    C_off = tl.arange(0, C_PAD)
    H_off = H_off[:, None]
    C_off = C_off[None, :]
    H_C_off = H_off * C + C_off

    if (H % H_PER_BLK == 0) and (C == C_PAD):
        H_mask = None
        H_C_mask = None
        H_PAD_VAL = None
        H_C_PAD_VAL = None
    elif H % H_PER_BLK == 0:
        H_mask = None
        C_mask = (C_off < C)
        H_C_mask = C_mask
        H_PAD_VAL = None
        H_C_PAD_VAL = 0.0
    else:
        H_mask = (H_off < H)
        C_mask = (C_off < C)
        H_C_mask = H_mask & C_mask
        H_PAD_VAL = 0.0
        H_C_PAD_VAL = 0.0

    src_off = src_idx * (H * C) + H_C_off
    neigh_start = tl.load(CSR_offsets_ptr + src_idx)
    neigh_end = tl.load(CSR_offsets_ptr + src_idx + 1)

    # src-side k, v (shared for all edges)
    k = tl.load(K_ptr + src_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
    v = tl.load(V_ptr + src_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)

    accK = tl.zeros((H_PER_BLK, C_PAD), dtype=tl.float32)
    accV = tl.zeros((H_PER_BLK, C_PAD), dtype=tl.float32)

    qk_scale_grad_kv: tl.constexpr = 1.0 / tl.sqrt(float(C))
    if USE_FAST_EXP2:
        qk_scale: tl.constexpr = (1.0 / tl.sqrt(float(C))) * 1.44269504089
    else:
        qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    q_base_ptrs = Q_ptr + H_C_off
    d_out_base_ptrs = D_OUT_ptr + H_C_off
    m_base_ptrs = M_ptr + H_off
    dj_base_ptrs = D_ptr + H_off
    e_base_ptrs = E_ptr + H_C_off
    d_e_base_ptrs = D_E_ptr + H_C_off

    for eidx in range(neigh_start, neigh_end):
        dst_idx = tl.load(CSR_indices_ptr + eidx)
        coo_idx = tl.load(CSR_map_to_coo_ptr + eidx)

        # get saved tensors for dst node
        dst_off = dst_idx * (H * C)
        q = tl.load(q_base_ptrs + dst_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
        d_out = tl.load(d_out_base_ptrs + dst_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)
        m_j = tl.load(m_base_ptrs + dst_idx * H, mask=H_mask, other=H_PAD_VAL).to(tl.float32)
        Dj = tl.load(dj_base_ptrs + dst_idx * H, mask=H_mask, other=H_PAD_VAL).to(tl.float32)

        e_off = coo_idx * (H * C)
        e = tl.load(e_base_ptrs + e_off, mask=H_C_mask, other=H_C_PAD_VAL).to(tl.float32)

        ke = k + e
        ve = v + e

        # some recomputations from dst-pass
        s_ij = tl.sum(q * ke, axis=-1, keep_dims=True) * qk_scale  # [H_PER_BLK, 1]
        if USE_FAST_EXP2:
            alpha_ij = tl.math.exp2(s_ij - m_j)
        else:
            alpha_ij = tl.exp(s_ij - m_j)
        dalpha = tl.sum(d_out * ve, axis=-1, keep_dims=True)  # [H_PER_BLK, 1]
        dS = alpha_ij * (dalpha - Dj)

        # per-edge k, v contributions, summing up to per-edge e contribution
        dV_edge = alpha_ij * d_out
        dK_edge = dS * q * qk_scale_grad_kv
        dE_edge = dV_edge + dK_edge

        tl.store(d_e_base_ptrs + e_off, dE_edge.to(D_E_ptr.type.element_ty), mask=H_C_mask)

        accK += dK_edge
        accV += dV_edge

    # write final accumulated per-src grads
    tl.store(D_K_ptr + src_off, accK.to(D_K_ptr.type.element_ty), mask=H_C_mask)
    tl.store(D_V_ptr + src_off, accV.to(D_V_ptr.type.element_ty), mask=H_C_mask)


# TODO(Jan): single bwd pass for non-bipartite graphs


@torch.library.custom_op("anemoi::sparse_graph_attention_coo", mutates_args=())
def sparse_graph_attention_coo(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v, e = q.contiguous(), k.contiguous(), v.contiguous(), e.contiguous()
    num_src_nodes, num_dst_nodes = k.size(0), q.size(0)
    csr_offsets, csr_indices, map_csr_to_coo, csc_offsets, csc_indices, map_csc_to_coo = fast_edge_index_to_csc(
        edge_index.contiguous(), num_src_nodes, num_dst_nodes
    )

    N_dst, H, C = q.shape

    H_PAD = triton.next_power_of_2(H)
    C_PAD = triton.next_power_of_2(C)

    # TODO: tune
    MAX_BLOCK_SIZE = 512
    H_PER_BLK = min(H_PAD, max(1, MAX_BLOCK_SIZE // C_PAD))
    BLOCK_SIZE = H_PER_BLK * C_PAD

    out = torch.zeros(N_dst, H, C, dtype=q.dtype, device=q.device)
    m = torch.empty(N_dst, H, dtype=torch.float32, device=q.device)

    grid = (N_dst, triton.cdiv(H, H_PER_BLK), 1)
    _graphtransformer_fwd_kernel[grid](
        q, k, v, e,
        csc_offsets, csc_indices, map_csc_to_coo,
        m, out,
        H, H_PER_BLK, C, C_PAD, True,
    )

    return out, m


@sparse_graph_attention_coo.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    N_dst, H, _ = q.shape
    return torch.empty_like(q), torch.empty((N_dst, H), dtype=torch.float32, device=q.device)


def _sparse_graph_attention_coo_setup_context(ctx, inputs, output):
    q, k, v, e, edge_index = inputs
    out, m = output
    ctx.save_for_backward(q, k, v, e, edge_index, out, m)


@torch.library.custom_op("anemoi::sparse_graph_attention_coo_backward_impl", mutates_args=())
def sparse_graph_attention_coo_backward_impl(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, e: torch.Tensor,
    edge_index: torch.Tensor, out: torch.Tensor, m: torch.Tensor, grad_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N_dst, H, C = q.shape
    N_src = k.size(0)

    csr_offsets, csr_indices, map_csr_to_coo, csc_offsets, csc_indices, map_csc_to_coo = fast_edge_index_to_csc(
        edge_index.contiguous(), N_src, N_dst
    )

    H_PAD = triton.next_power_of_2(H)
    C_PAD = triton.next_power_of_2(C)
    MAX_BLOCK_SIZE = 512
    H_PER_BLK = min(H_PAD, max(1, MAX_BLOCK_SIZE // C_PAD))

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    de = torch.empty_like(e)
    # D_j = <d_out_j, out_j> per destination node, computed in pass 1 and consumed in pass 2
    D = torch.empty((N_dst, H), dtype=torch.float32, device=q.device)

    grid_dst = (N_dst, triton.cdiv(H, H_PER_BLK), 1)
    _graphtransformer_bwd_dq_kernel[grid_dst](
        q, k, v, e, out, m,
        grad_out,
        csc_offsets, csc_indices, map_csc_to_coo,
        dq, D,
        H, H_PER_BLK, C, C_PAD, True,
    )

    grid_src = (N_src, triton.cdiv(H, H_PER_BLK), 1)
    _graphtransformer_bwd_dkdv_kernel[grid_src](
        q, k, v, e,
        D, m, grad_out,
        csr_offsets, csr_indices, map_csr_to_coo,
        dk, dv, de,
        H, H_PER_BLK, C, C_PAD, True,
    )

    return dq, dk, dv, de


@sparse_graph_attention_coo_backward_impl.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, e: torch.Tensor,
    edge_index: torch.Tensor, out: torch.Tensor, m: torch.Tensor, grad_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(e)


def sparse_graph_attention_coo_backward(
    ctx, grad_out: torch.Tensor, grad_m: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
    q, k, v, e, edge_index, out, m = ctx.saved_tensors
    dq, dk, dv, de = sparse_graph_attention_coo_backward_impl(
        q, k, v, e, edge_index, out, m, grad_out.contiguous()
    )
    return dq, dk, dv, de, None


sparse_graph_attention_coo.register_autograd(
    sparse_graph_attention_coo_backward,
    setup_context=_sparse_graph_attention_coo_setup_context,
)
