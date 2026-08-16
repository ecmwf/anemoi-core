# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math
import os
from typing import Optional

import torch
from torch import Tensor

# check if triton is installed
# If pytorch is installed on CPU then torch is not available
try:
    import triton
    import triton.language as tl
except ImportError:
    raise ValueError(
        "Error. The 'triton' backend was selected for the GraphTransformer but Triton is not installed. To use this backend please install Triton. Otherwise, select a different backend for the GraphTransformer in the models config."
    )


@triton.jit
def build_masks_and_offsets(H: tl.constexpr, C: tl.constexpr, H_pad: tl.constexpr, C_pad: tl.constexpr):
    """Pads H and C to the nearest power of 2 if needed.

    This is required to support non-square numbers of heads and/or channels.
    Returns a mask for H, H*C and an offset for accessing into a 2D H*C matrix, ignoring padded values

    masking apparently has a price, so if H and C are already powers of 2, nothing is returned
    If H is already a power of 2 but C is not, a simpler H*C mask is returned

    This function assumes a matrix layout of shape [H,C] for mask_H_C and H_C_off
    """

    # default mask (assume no padded values)
    H_mask = True
    H_C_mask = True

    if H == H_pad and C == C_pad:
        H_C_off = tl.arange(0, H * C)

    elif H == H_pad:  # just C is not square, we can avoid mask_H
        C_pad_off = tl.arange(0, C_pad)[None, :]  # (1, C_pad)
        H_off = tl.arange(0, H)[:, None]  # (H, 1)

        # 2D mask for H * C
        # e.g 1 2 X X
        #     5 6 X X
        #     X X X X
        # But this kernel loads in 1d, hence we reshape to 1d
        # shape (H_pad, 1) & shape (1, C_pad) => shape (H_pad, C_pad) => shape (H_pad * C_pad, )
        H_C_mask_2d = (C_pad_off < C) & (H_off < H)  # (H, C_pad)
        H_C_mask = tl.reshape(H_C_mask_2d, (H * C_pad,))
        H_C_off = tl.reshape(H_off * C + C_pad_off, (H * C_pad,))

    else:  # H and C both not square
        H_pad_off = tl.arange(0, H_pad)[:, None]
        C_pad_off = tl.arange(0, C_pad)[None, :]

        # mask for H
        H_mask = tl.arange(0, H_pad) < H

        # 2D mask for H * C
        # e.g 1 2 X X
        #     5 6 X X
        #     X X X X
        # But this kernel loads in 1d, hence we reshape to 1d
        # shape (H_pad, 1) & shape (1, C_pad) => shape (H_pad, C_pad) => shape (H_pad * C_pad, )
        H_C_mask_2d = (C_pad_off < C) & (H_pad_off < H)  # (H, C_pad)
        H_C_mask = tl.reshape(H_C_mask_2d, (H_pad * C_pad,))

        # tl.arange(H_pad, C_pad) doesnt work, because the arrays its offseting into aren't padded
        # Therefore we make our own range, using unpadded major dimension (C)
        H_C_off = tl.reshape(H_pad_off * C + C_pad_off, (H_pad * C_pad,))

    return H_mask, H_C_mask, H_C_off


@triton.jit
def _gt_fwd(
    Q_ptr,  # [N_dst, H, C]
    K_ptr,  # [N_src, H, C]
    V_ptr,  # [N_src, H, C]
    E_ptr,  # [M, H, C]
    M_ptr,  # [M, H]
    ROW_ptr,  # [M]
    COLPTR_ptr,  # [N_dst+1]
    OUT_ptr,  # [N_dst, H, C]
    N_dst,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    pid = tl.program_id(0)
    dst_idx = pid
    if dst_idx >= N_dst:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    dst_start = dst_idx * H * C
    dst_off = dst_start + H_C_off

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    if num_edges == 0:
        zeros = tl.zeros((H_pad,), dtype=tl.float32)  # m initialised as torch.float32
        M_off = M_ptr + dst_idx * H + tl.arange(0, H_pad)
        tl.store(M_off, zeros, mask=H_mask)
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        OUT_off = OUT_ptr + dst_off
        tl.store(OUT_off, zeros, mask=H_C_mask)
        return

    q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    acc = tl.zeros((H_pad, C_pad), dtype=tl.float32)  # output accumulator, pending normalization by l_i
    l_i = tl.zeros((H_pad,), dtype=tl.float32)  # sum of attention weights
    m_i = tl.full((H_pad,), value=-float("inf"), dtype=tl.float32)  # running max for stability

    # helpers to avoid repeated computations/indexing:
    edge_ptr = E_ptr + neigh_start * H * C + H_C_off  # pointer to first edge_attr
    e_idx = neigh_start  # first edge index
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    # for _ in tl.range(num_edges, warp_specialize=True):
    for _ in range(num_edges):
        e = tl.load(edge_ptr, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        # src neighbor index: rowptr[e_idx]
        src_idx = tl.load(ROW_ptr + e_idx)

        src_off = src_idx * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        k_e = k + e
        v_e = v + e

        qk = tl.sum(q * k_e, axis=-1) * qk_scale  # Shape: [H]

        m_ij = tl.maximum(m_i, qk)  # new running max
        alpha_ij = tl.exp(qk - m_ij)  # attention weight for current edge
        correction = tl.exp(m_i - m_ij)  # correction factor for previous accumulations

        # update accumulators with correction
        acc = acc * correction[:, None]
        l_i = l_i * correction

        # add current contribution, update running max
        acc = acc + alpha_ij[:, None] * v_e
        l_i = l_i + alpha_ij
        m_i = m_ij

        # move to next edge
        edge_ptr += H * C
        e_idx += 1

    # final normalization: divide by sum of attention weights
    acc = acc / l_i[:, None]
    tl.store(
        OUT_ptr + dst_off,
        acc.to(out_dtype).reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )

    # store m_i + log(l_i) for backward
    m_start = dst_idx * H
    m_off = m_start + tl.arange(0, H_pad)

    m_i += tl.log(l_i)
    tl.store(M_ptr + m_off, m_i, mask=H_mask)


@triton.jit
def _gt_bwd_dst_pass(
    Q_ptr,
    K_ptr,
    V_ptr,
    E_ptr,
    OUT_ptr,  # saved forward outputs o_i
    M_ptr,  # saved m_i + ln l_i
    ROW_ptr,  # [M] (edge -> src)
    COLPTR_ptr,  # [N_dst + 1]
    D_OUT_ptr,  # [N_dst * H * C]
    D_Q_ptr,  # OUT
    D_ptr,  # [N_dst * H]
    N_dst,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    dst_idx = tl.program_id(0)
    if dst_idx >= N_dst:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    dst_off = dst_idx * H * C + H_C_off

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    if num_edges == 0:
        # store D_j = <d_out, out> = 0 and dQ = 0
        zeros = tl.zeros((H_pad,), dtype=tl.float32)
        tl.store(D_ptr + dst_idx * H + tl.arange(0, H_pad), zeros, mask=H_mask)
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_Q_ptr + dst_off, zeros, mask=H_C_mask)
        return

    d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    out = tl.load(OUT_ptr + dst_off, H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    # D_j = <d_out, out> for one-pass computation of dQ
    Dj = tl.sum(d_out * out, axis=-1)  # [H]

    q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    dq = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    edge_ptr = E_ptr + neigh_start * H * C + H_C_off  # pointer to first edge_attr
    e_idx = neigh_start  # first edge index
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    # for _ in tl.range(num_edges, warp_specialize=True):
    for _ in range(num_edges):
        e = tl.load(edge_ptr, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        src = tl.load(ROW_ptr + e_idx)
        src_off = src * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        ke = k + e
        # score and alpha using saved M
        m_j = tl.load(M_ptr + dst_idx * H + tl.arange(0, H_pad), mask=H_mask).to(tl.float32)
        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        alpha_ij = tl.exp(s_ij - m_j)

        v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        ve = v + e

        dalpha = tl.sum(d_out * ve, axis=-1)
        dS = alpha_ij * (dalpha - Dj)

        dq += dS[:, None] * ke * qk_scale

        # move to next edge
        edge_ptr += H * C
        e_idx += 1

    # store D_j and dQ
    tl.store(D_ptr + dst_idx * H + tl.arange(0, H_pad), Dj, mask=H_mask)
    tl.store(
        D_Q_ptr + dst_off,
        dq.to(out_dtype).reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )


@triton.jit
def _gt_bwd_src_pass(
    Q_ptr,
    K_ptr,
    V_ptr,
    E_ptr,
    ROWPTR_ptr,  # [N_src+1]
    EDGE_IDS_ptr,  # [M] edge id list grouped by src
    EDGE_DST_ptr,  # [M] dst node for each edge
    D_ptr,  # [N_dst * H] D_j from pass dst-pass
    M_ptr,  # [N_dst * H] saved m_j from fwd
    D_OUT_ptr,  # [N_dst * H * C]
    D_K_ptr,  # [N_src * H * C]
    D_V_ptr,  # [N_src * H * C]
    D_E_ptr,  # [M * H * C]
    N_src,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    src_idx = tl.program_id(0)
    if src_idx >= N_src:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    _, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    start = tl.load(ROWPTR_ptr + src_idx)
    end = tl.load(ROWPTR_ptr + src_idx + 1)
    num_edges = end - start

    if num_edges == 0:
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_K_ptr + src_idx * H * C + H_C_off, zeros, mask=H_C_mask)
        tl.store(D_V_ptr + src_idx * H * C + H_C_off, zeros, mask=H_C_mask)
        return

    # src-side k, v (shared for all edges)
    src_off = src_idx * H * C + H_C_off
    k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    accK = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    accV = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    # note that edges aren't necessarily contiguous in memory here, use EDGE_IDS_ptr
    for i in range(num_edges):
        # for i in tl.range(0, num_edges, warp_specialize=True):
        # indexing into edge list + corresponding dst node
        e_idx = tl.load(EDGE_IDS_ptr + start + i)
        dst = tl.load(EDGE_DST_ptr + e_idx)

        # get saved tensors for dst node
        dst_off = dst * H * C + H_C_off
        q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        m_j = tl.load(M_ptr + dst * H + tl.arange(0, H_pad)).to(tl.float32)
        Dj = tl.load(D_ptr + dst * H + tl.arange(0, H_pad)).to(tl.float32)

        e_off = e_idx * H * C + H_C_off
        e = tl.load(E_ptr + e_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        ke = k + e
        ve = v + e

        # some recomputations from dst-pass
        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        alpha_ij = tl.exp(s_ij - m_j)
        dalpha = tl.sum(d_out * ve, axis=-1)
        dS = alpha_ij * (dalpha - Dj)

        # per-edge k, v contributions, summing up to per-edge e contribution
        dV_edge = alpha_ij[:, None] * d_out
        dK_edge = dS[:, None] * q * qk_scale
        dE_edge = dV_edge + dK_edge

        tl.store(
            D_E_ptr + e_off,
            dE_edge.to(out_dtype).reshape(
                H_pad * C_pad,
            ),
            mask=H_C_mask,
        )

        accK += dK_edge
        accV += dV_edge

    # write final accumulated per-src grads
    tl.store(
        D_K_ptr + src_off,
        accK.to(out_dtype).reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )
    tl.store(
        D_V_ptr + src_off,
        accV.to(out_dtype).reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )


#########################################
# PyTorch Custom Operator for Triton GT #
#########################################
# These functions wrap the Triton kernels in PyTorch custom ops,
# so that they can be used in a PyTorch autograd graph and compiled with torch.compile.
# They include '_fake' versions which just do the relevant memory allocations
# and return empty tensors, for use in torch.compile tracing.
# The '_setup_context' function saves the necessary tensors for the backward pass.
# for more details on pytorch custom ops see https://docs.pytorch.org/tutorials/advanced/python_custom_ops_functional.html


@torch.library.custom_op("anemoi::graph_transformer_attention", mutates_args=(), device_types="cuda")
def graph_transformer_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    e: Tensor,
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Opaque custom op wrapping the Triton GraphTransformer attention.

    Returns
    -------
    out : Tensor
        Attention output cast back to ``q.dtype`` (the user-facing result).
    out_saved : Tensor
        Float32 attention output, kept for the backward pass.
    m : Tensor
        Float32 log-sum-exp normalizer, kept for the backward pass.
    """
    q, k, v, e = (x.contiguous() for x in (q, k, v, e))
    row, colptr = (x.contiguous() for x in (row, colptr))

    N_dst, H, C = q.shape
    out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)
    m = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

    _gt_fwd[(N_dst,)](q, k, v, e, m, row, colptr, out_saved, N_dst, H, C, tl.float32)

    out = out_saved.to(q.dtype)
    # Custom-op outputs must not alias one another; ``.to`` returns ``self`` when
    # ``q`` is already float32, so clone to keep ``out`` and ``out_saved`` distinct.
    if out is out_saved:
        out = out.clone()

    return out, out_saved, m


@graph_transformer_attention.register_fake
def _graph_transformer_attention_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    e: Tensor,
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    N_dst, H, C = q.shape
    out = torch.empty((N_dst, H, C), device=q.device, dtype=q.dtype)
    out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)
    m = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)
    return out, out_saved, m


# TODO(Jan): single bwd pass for non-bipartite graphs
@torch.library.custom_op("anemoi::graph_transformer_attention_backward", mutates_args=(), device_types="cuda")
def graph_transformer_attention_backward(
    d_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    e: Tensor,
    out_saved: Tensor,
    m: Tensor,
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Opaque custom op wrapping the Triton GraphTransformer backward kernels.

    Registered as its own custom op so that AOTAutograd does not trace into the
    raw Triton kernel launches when compiling the backward graph.
    """
    d_out = d_out.contiguous()

    N_dst, H, C = q.shape
    N_src = k.shape[0]

    def torch_dtype_to_triton(dtype):
        if dtype == torch.float16:
            return tl.float16
        elif dtype == torch.bfloat16:
            return tl.bfloat16
        elif dtype == torch.float32:
            return tl.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    grad_dtype = torch_dtype_to_triton(d_out.dtype)

    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)
    dE = torch.empty_like(e)
    D = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

    # Pass A: destination nodes (computes D and dQ)
    _gt_bwd_dst_pass[(N_dst,)](q, k, v, e, out_saved, m, row, colptr, d_out, dQ, D, N_dst, H, C, grad_dtype)

    # Pass B: source nodes (accumulate dK, dV, dE)
    _gt_bwd_src_pass[(N_src,)](q, k, v, e, rowptr, edge_ids, edge_dst, D, m, d_out, dK, dV, dE, N_src, H, C, grad_dtype)

    return dQ, dK, dV, dE


@graph_transformer_attention_backward.register_fake
def _graph_transformer_attention_backward_fake(
    d_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    e: Tensor,
    out_saved: Tensor,
    m: Tensor,
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(e),
    )


def _graph_transformer_attention_backward(ctx, d_out, _d_out_saved, _d_m):
    # Only the gradient w.r.t. the user-facing ``out`` is used; ``out_saved`` and
    # ``m`` are internal saved tensors that are not consumed downstream.
    q, k, v, e, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst = ctx.saved_tensors

    dQ, dK, dV, dE = graph_transformer_attention_backward(
        d_out, q, k, v, e, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst
    )

    # Gradients for (q, k, v, e, row, colptr, rowptr, edge_ids, edge_dst).
    return dQ, dK, dV, dE, None, None, None, None, None


def _graph_transformer_attention_setup_context(ctx, inputs, output):
    q, k, v, e, row, colptr, rowptr, edge_ids, edge_dst = inputs
    _out, out_saved, m = output

    # The forward op makes contiguous copies internally, but those are not the tensors
    # passed here (setup_context receives the original op inputs). Save contiguous
    # versions so the Triton backward kernels, which assume a contiguous layout, receive
    # contiguous inputs.
    q, k, v, e = (x.contiguous() for x in (q, k, v, e))
    row, colptr, rowptr, edge_ids, edge_dst = (x.contiguous() for x in (row, colptr, rowptr, edge_ids, edge_dst))

    ctx.save_for_backward(q, k, v, e, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst)


graph_transformer_attention.register_autograd(
    _graph_transformer_attention_backward,
    setup_context=_graph_transformer_attention_setup_context,
)


#######################
# Triton GT interface #
#######################


def graph_transformer_attention_conv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    edges: Tensor,
    csc: tuple[Tensor, Tensor],
    reverse: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """torch.compile-friendly GraphTransformer attention."""
    row, colptr = csc
    rowptr, edge_ids, edge_dst = reverse
    out, _out_saved, _m = graph_transformer_attention(query, key, value, edges, row, colptr, rowptr, edge_ids, edge_dst)
    return out


##########################################
# Fused edge-attribute embedding variant #
##########################################
# Computes the per-edge additive bias e = edge_attr @ W + b inside the kernels
# instead of consuming a materialized [M, H, C] tensor: the embedding is a K=3
# GEMM whose cost is purely writing (and re-reading) 2 KB per edge, and on large
# graphs it dominates both step time and activation memory. W is [edge_dim, H*C]
# (lin_edge.weight transposed) and stays resident in L2 (~16 KB).
#
# The backward needs no per-edge dE: dE_e = alpha_e * d_out_j + qk_scale * dS_e
# * q_j, so dW = sum_e attr_e (x) dE_e collapses to per-destination accumulators
# A_j = sum attr_e*alpha_e and B_j = sum attr_e*dS_e (the dst pass computes
# alpha and dS per edge anyway) followed by node-proportional einsums. For
# low-degree graphs the per-edge recomputation in the fully fused backward is
# slower than the unfused kernels, so below a mean-degree threshold the backward
# materializes e transiently and reuses the unfused backward op instead; the
# forward (and thus checkpoint recompute) never materializes e either way.

# Minimum mean destination degree (edges / dst nodes) for the fully fused
# backward; below it the hybrid backward is used. Crossover measured on A100.
FUSED_BWD_MIN_MEAN_DEGREE = float(os.environ.get("ANEMOI_GT_FUSED_BWD_MIN_MEAN_DEGREE", "6.0"))


@triton.jit
def _load_edge_embed_weights(
    W_ptr, B_ptr, ED: tl.constexpr, ED_pad: tl.constexpr, H: tl.constexpr, C: tl.constexpr, H_C_mask, H_C_off
):
    """Load W [ED, H*C] and b [H*C] into registers (b as zeros when B_ptr is null)."""
    d_off = tl.arange(0, ED_pad)
    d_mask = d_off < ED
    w = tl.load(W_ptr + d_off[:, None] * (H * C) + H_C_off[None, :], mask=d_mask[:, None] & H_C_mask, other=0.0).to(
        tl.float32
    )
    b = tl.load(B_ptr + H_C_off, mask=H_C_mask, other=0.0).to(tl.float32)
    return w, b


@triton.jit
def _fused_edge_bias(ATTR_ptr, e_idx, w, b, ED: tl.constexpr, ED_pad: tl.constexpr):
    """Per-edge bias e = attr @ W + b; returns (attr, e_flat)."""
    d_off = tl.arange(0, ED_pad)
    attr = tl.load(ATTR_ptr + e_idx * ED + d_off, mask=d_off < ED, other=0.0).to(tl.float32)
    return attr, tl.sum(attr[:, None] * w, axis=0) + b


@triton.jit
def _gt_fused_fwd(
    Q_ptr,  # [N_dst, H, C]
    K_ptr,  # [N_src, H, C]
    V_ptr,  # [N_src, H, C]
    ATTR_ptr,  # [M, ED]
    W_ptr,  # [ED, H*C]
    B_ptr,  # [H*C]
    M_ptr,
    ROW_ptr,
    COLPTR_ptr,
    OUT_ptr,
    N_dst,
    ED: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    dst_idx = tl.program_id(0)
    if dst_idx >= N_dst:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    ED_pad: tl.constexpr = triton.next_power_of_2(ED)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    dst_off = dst_idx * H * C + H_C_off

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    if num_edges == 0:
        zeros = tl.zeros((H_pad,), dtype=tl.float32)
        tl.store(M_ptr + dst_idx * H + tl.arange(0, H_pad), zeros, mask=H_mask)
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(OUT_ptr + dst_off, zeros, mask=H_C_mask)
        return

    w, b = _load_edge_embed_weights(W_ptr, B_ptr, ED, ED_pad, H, C, H_C_mask, H_C_off)
    q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    acc = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    l_i = tl.zeros((H_pad,), dtype=tl.float32)
    m_i = tl.full((H_pad,), value=-float("inf"), dtype=tl.float32)

    e_idx = neigh_start
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for _ in range(num_edges):
        attr_i, e_flat = _fused_edge_bias(ATTR_ptr, e_idx, w, b, ED, ED_pad)  # noqa: F841
        e = e_flat.reshape((H_pad, C_pad))

        src_idx = tl.load(ROW_ptr + e_idx)
        src_off = src_idx * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        k_e = k + e
        v_e = v + e

        qk = tl.sum(q * k_e, axis=-1) * qk_scale

        m_ij = tl.maximum(m_i, qk)
        alpha_ij = tl.exp(qk - m_ij)
        correction = tl.exp(m_i - m_ij)

        acc = acc * correction[:, None] + alpha_ij[:, None] * v_e
        l_i = l_i * correction + alpha_ij
        m_i = m_ij

        e_idx += 1

    acc = acc / l_i[:, None]
    tl.store(OUT_ptr + dst_off, acc.to(out_dtype).reshape((H_pad * C_pad,)), mask=H_C_mask)

    m_i += tl.log(l_i)
    tl.store(M_ptr + dst_idx * H + tl.arange(0, H_pad), m_i, mask=H_mask)


@triton.jit
def _gt_fused_bwd_dst_pass(
    Q_ptr,
    K_ptr,
    V_ptr,
    ATTR_ptr,
    W_ptr,
    B_ptr,
    OUT_ptr,
    M_ptr,
    ROW_ptr,
    COLPTR_ptr,
    D_OUT_ptr,
    D_Q_ptr,
    D_ptr,
    A_ptr,  # [N_dst, ED, H]  sum attr*alpha
    Bc_ptr,  # [N_dst, ED, H]  sum attr*dS
    AS_ptr,  # [N_dst, H]      sum alpha
    BS_ptr,  # [N_dst, H]      sum dS
    N_dst,
    ED: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    dst_idx = tl.program_id(0)
    if dst_idx >= N_dst:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    ED_pad: tl.constexpr = triton.next_power_of_2(ED)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    dst_off = dst_idx * H * C + H_C_off
    h_off = tl.arange(0, H_pad)
    d_off = tl.arange(0, ED_pad)
    dh_off = dst_idx * ED * H + d_off[:, None] * H + h_off[None, :]
    dh_mask = (d_off[:, None] < ED) & (h_off[None, :] < H)

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    if num_edges == 0:
        zeros_h = tl.zeros((H_pad,), dtype=tl.float32)
        tl.store(D_ptr + dst_idx * H + h_off, zeros_h, mask=H_mask)
        tl.store(D_Q_ptr + dst_off, tl.zeros((H_pad * C_pad,), dtype=out_dtype), mask=H_C_mask)
        tl.store(A_ptr + dh_off, tl.zeros((ED_pad, H_pad), dtype=tl.float32), mask=dh_mask)
        tl.store(Bc_ptr + dh_off, tl.zeros((ED_pad, H_pad), dtype=tl.float32), mask=dh_mask)
        tl.store(AS_ptr + dst_idx * H + h_off, zeros_h, mask=H_mask)
        tl.store(BS_ptr + dst_idx * H + h_off, zeros_h, mask=H_mask)
        return

    w, b = _load_edge_embed_weights(W_ptr, B_ptr, ED, ED_pad, H, C, H_C_mask, H_C_off)
    d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    out = tl.load(OUT_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    Dj = tl.sum(d_out * out, axis=-1)
    q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    m_j = tl.load(M_ptr + dst_idx * H + h_off, mask=H_mask).to(tl.float32)

    dq = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    accA = tl.zeros((ED_pad, H_pad), dtype=tl.float32)
    accB = tl.zeros((ED_pad, H_pad), dtype=tl.float32)
    asum = tl.zeros((H_pad,), dtype=tl.float32)
    bsum = tl.zeros((H_pad,), dtype=tl.float32)

    e_idx = neigh_start
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for _ in range(num_edges):
        attr, e_flat = _fused_edge_bias(ATTR_ptr, e_idx, w, b, ED, ED_pad)
        e = e_flat.reshape((H_pad, C_pad))

        src = tl.load(ROW_ptr + e_idx)
        src_off = src * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        ke = k + e
        ve = v + e
        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        alpha_ij = tl.exp(s_ij - m_j)
        dalpha = tl.sum(d_out * ve, axis=-1)
        dS = alpha_ij * (dalpha - Dj)

        dq += dS[:, None] * ke * qk_scale
        accA += attr[:, None] * alpha_ij[None, :]
        accB += attr[:, None] * dS[None, :]
        asum += alpha_ij
        bsum += dS

        e_idx += 1

    tl.store(D_ptr + dst_idx * H + h_off, Dj, mask=H_mask)
    tl.store(D_Q_ptr + dst_off, dq.to(out_dtype).reshape((H_pad * C_pad,)), mask=H_C_mask)
    tl.store(A_ptr + dh_off, accA, mask=dh_mask)
    tl.store(Bc_ptr + dh_off, accB, mask=dh_mask)
    tl.store(AS_ptr + dst_idx * H + h_off, asum, mask=H_mask)
    tl.store(BS_ptr + dst_idx * H + h_off, bsum, mask=H_mask)


@triton.jit
def _gt_fused_bwd_src_pass(
    Q_ptr,
    K_ptr,
    V_ptr,
    ATTR_ptr,
    W_ptr,
    B_ptr,
    ROWPTR_ptr,
    EDGE_IDS_ptr,
    EDGE_DST_ptr,
    D_ptr,
    M_ptr,
    D_OUT_ptr,
    D_K_ptr,
    D_V_ptr,
    N_src,
    ED: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    src_idx = tl.program_id(0)
    if src_idx >= N_src:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    ED_pad: tl.constexpr = triton.next_power_of_2(ED)
    _, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    start = tl.load(ROWPTR_ptr + src_idx)
    end = tl.load(ROWPTR_ptr + src_idx + 1)
    num_edges = end - start

    src_off = src_idx * H * C + H_C_off
    if num_edges == 0:
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_K_ptr + src_off, zeros, mask=H_C_mask)
        tl.store(D_V_ptr + src_off, zeros, mask=H_C_mask)
        return

    w, b = _load_edge_embed_weights(W_ptr, B_ptr, ED, ED_pad, H, C, H_C_mask, H_C_off)
    k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    accK = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    accV = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for i in range(num_edges):
        e_idx = tl.load(EDGE_IDS_ptr + start + i)
        dst = tl.load(EDGE_DST_ptr + e_idx)

        dst_off = dst * H * C + H_C_off
        q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        m_j = tl.load(M_ptr + dst * H + tl.arange(0, H_pad), mask=tl.arange(0, H_pad) < H).to(tl.float32)
        Dj = tl.load(D_ptr + dst * H + tl.arange(0, H_pad), mask=tl.arange(0, H_pad) < H).to(tl.float32)

        attr_i, e_flat = _fused_edge_bias(ATTR_ptr, e_idx, w, b, ED, ED_pad)  # noqa: F841
        e = e_flat.reshape((H_pad, C_pad))

        ke = k + e
        ve = v + e

        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        alpha_ij = tl.exp(s_ij - m_j)
        dalpha = tl.sum(d_out * ve, axis=-1)
        dS = alpha_ij * (dalpha - Dj)

        accV += alpha_ij[:, None] * d_out
        accK += dS[:, None] * q * qk_scale

    tl.store(D_K_ptr + src_off, accK.to(out_dtype).reshape((H_pad * C_pad,)), mask=H_C_mask)
    tl.store(D_V_ptr + src_off, accV.to(out_dtype).reshape((H_pad * C_pad,)), mask=H_C_mask)


@torch.library.custom_op("anemoi::graph_transformer_attention_fused", mutates_args=(), device_types="cuda")
def graph_transformer_attention_fused(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edge_attr: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Triton GT attention with the edge embedding computed in-kernel.

    ``edge_attr`` is the raw [M, edge_dim] attribute tensor (dst-sorted) and
    ``weight``/``bias`` are ``lin_edge`` parameters with ``weight`` transposed
    to [edge_dim, H*C]; the [M, H, C] embedded tensor is never materialized.
    """
    q, k, v, edge_attr, weight = (x.contiguous() for x in (q, k, v, edge_attr, weight))
    row, colptr = row.contiguous(), colptr.contiguous()

    N_dst, H, C = q.shape
    ED = edge_attr.shape[1]
    if bias is None:
        bias = torch.zeros(H * C, device=q.device, dtype=weight.dtype)
    bias = bias.contiguous()

    out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)
    m = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

    _gt_fused_fwd[(N_dst,)](q, k, v, edge_attr, weight, bias, m, row, colptr, out_saved, N_dst, ED, H, C, tl.float32)

    out = out_saved.to(q.dtype)
    if out is out_saved:
        out = out.clone()

    return out, out_saved, m


@graph_transformer_attention_fused.register_fake
def _graph_transformer_attention_fused_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edge_attr: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    N_dst, H, C = q.shape
    out = torch.empty((N_dst, H, C), device=q.device, dtype=q.dtype)
    out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)
    m = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)
    return out, out_saved, m


@torch.library.custom_op("anemoi::graph_transformer_attention_fused_backward", mutates_args=(), device_types="cuda")
def graph_transformer_attention_fused_backward(
    d_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edge_attr: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    out_saved: Tensor,
    m: Tensor,
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Backward for the fused op: returns (dQ, dK, dV, dW, db).

    Fully fused for mean dst degree >= FUSED_BWD_MIN_MEAN_DEGREE; below that,
    the per-edge recomputation loses to the unfused kernels, so e is
    materialized transiently and the unfused backward op is reused (with
    dW = attr^T dE and db = sum dE).
    """
    d_out = d_out.contiguous()
    N_dst, H, C = q.shape
    N_src = k.shape[0]
    M, ED = edge_attr.shape

    if M / max(N_dst, 1) < FUSED_BWD_MIN_MEAN_DEGREE:
        e = edge_attr.to(weight.dtype) @ weight
        if bias is not None:
            e = e + bias
        e = e.view(M, H, C)
        dQ, dK, dV, dE = graph_transformer_attention_backward(
            d_out, q, k, v, e, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst
        )
        dE = dE.reshape(M, H * C)
        dW = edge_attr.to(dE.dtype).t() @ dE
        db = dE.sum(dim=0)
        return dQ, dK, dV, dW.to(weight.dtype), db.to(weight.dtype)

    def torch_dtype_to_triton(dtype):
        if dtype == torch.float16:
            return tl.float16
        elif dtype == torch.bfloat16:
            return tl.bfloat16
        elif dtype == torch.float32:
            return tl.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    grad_dtype = torch_dtype_to_triton(d_out.dtype)

    bias_t = bias.contiguous() if bias is not None else torch.zeros(H * C, device=q.device, dtype=weight.dtype)

    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)
    D = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)
    A = torch.empty((N_dst, ED, H), device=q.device, dtype=torch.float32)
    Bc = torch.empty((N_dst, ED, H), device=q.device, dtype=torch.float32)
    asum = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)
    bsum = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

    _gt_fused_bwd_dst_pass[(N_dst,)](
        q,
        k,
        v,
        edge_attr,
        weight,
        bias_t,
        out_saved,
        m,
        row,
        colptr,
        d_out,
        dQ,
        D,
        A,
        Bc,
        asum,
        bsum,
        N_dst,
        ED,
        H,
        C,
        grad_dtype,
    )
    _gt_fused_bwd_src_pass[(N_src,)](
        q,
        k,
        v,
        edge_attr,
        weight,
        bias_t,
        rowptr,
        edge_ids,
        edge_dst,
        D,
        m,
        d_out,
        dK,
        dV,
        N_src,
        ED,
        H,
        C,
        grad_dtype,
    )

    # dW = sum_e attr_e (x) dE_e, contracted per destination node
    qk_scale = 1.0 / math.sqrt(C)
    d_out32 = d_out.float()
    q32 = q.float()
    dW = torch.einsum("ndh,nhc->dhc", A, d_out32) + qk_scale * torch.einsum("ndh,nhc->dhc", Bc, q32)
    db = torch.einsum("nh,nhc->hc", asum, d_out32) + qk_scale * torch.einsum("nh,nhc->hc", bsum, q32)
    return dQ, dK, dV, dW.reshape(ED, H * C).to(weight.dtype), db.reshape(H * C).to(weight.dtype)


@graph_transformer_attention_fused_backward.register_fake
def _graph_transformer_attention_fused_backward_fake(
    d_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    edge_attr: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    out_saved: Tensor,
    m: Tensor,
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_ids: Tensor,
    edge_dst: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(weight),
        torch.empty(weight.shape[1], device=weight.device, dtype=weight.dtype),
    )


def _graph_transformer_attention_fused_backward(ctx, d_out, _d_out_saved, _d_m):
    saved = list(ctx.saved_tensors)
    if not ctx.has_bias:
        saved.insert(5, None)
    q, k, v, edge_attr, weight, bias, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst = saved

    dQ, dK, dV, dW, db = graph_transformer_attention_fused_backward(
        d_out, q, k, v, edge_attr, weight, bias, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst
    )
    if bias is None:
        db = None
    # Gradients for (q, k, v, edge_attr, weight, bias, row, colptr, rowptr, edge_ids, edge_dst).
    return dQ, dK, dV, None, dW, db, None, None, None, None, None


def _graph_transformer_attention_fused_setup_context(ctx, inputs, output):
    q, k, v, edge_attr, weight, bias, row, colptr, rowptr, edge_ids, edge_dst = inputs
    _out, out_saved, m = output

    q, k, v, edge_attr, weight = (x.contiguous() for x in (q, k, v, edge_attr, weight))
    bias = bias.contiguous() if bias is not None else None
    row, colptr, rowptr, edge_ids, edge_dst = (x.contiguous() for x in (row, colptr, rowptr, edge_ids, edge_dst))

    tensors = [q, k, v, edge_attr, weight, bias, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst]
    ctx.has_bias = bias is not None
    ctx.save_for_backward(*[t for t in tensors if t is not None])


graph_transformer_attention_fused.register_autograd(
    _graph_transformer_attention_fused_backward,
    setup_context=_graph_transformer_attention_fused_setup_context,
)


def graph_transformer_attention_fused_conv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    edge_attr: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    csc: tuple[Tensor, Tensor],
    reverse: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """torch.compile-friendly GraphTransformer attention with fused edge embedding."""
    row, colptr = csc
    rowptr, edge_ids, edge_dst = reverse
    out, _out_saved, _m = graph_transformer_attention_fused(
        query, key, value, edge_attr, weight, bias, row, colptr, rowptr, edge_ids, edge_dst
    )
    return out
