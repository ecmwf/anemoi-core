# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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

# The forward and both backward kernels must compute each score q . (k + e) to the same bits. The
# backward subtracts the row maximum saved by the forward, so if the winning edge's recomputed score
# differs from it in the last bit, its probability is off by that rounding times the scale; with
# nearly one-hot attention and large scores this is far beyond float32 accuracy (dV errors hundreds
# of times those of plain float32 at scores around 1e4). The compiler merges multiplies and adds into
# single fused operations differently in each kernel, so all three are compiled without that merging.
_SAME_ROUNDING = dict(enable_fp_fusion=False)


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
    STATS_ptr,  # [N_dst, 2, H] row maximum of the unscaled scores q . (k + e), then inverse of the softmax sum
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
        zeros = tl.zeros((H_pad,), dtype=tl.float32)  # stats initialised as torch.float32
        stats_off = STATS_ptr + dst_idx * 2 * H + tl.arange(0, H_pad)
        tl.store(stats_off, zeros, mask=H_mask)
        tl.store(stats_off + H, zeros, mask=H_mask)
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
    # 1 / sqrt(C) / ln 2: scales the scores and turns exp(x) into 2^(x / ln 2)
    exp2_scale: tl.constexpr = 1.4426950408889634 / tl.sqrt(float(C))

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

        qk = tl.sum(q * k_e, axis=-1)  # Shape: [H]

        # The running maximum is subtracted from the raw score before it is scaled. Near the maximum
        # the difference is exact, so the rounding error in the exponent is relative to the gap from
        # the maximum rather than to the score itself. See KohakuBlueleaf, KohakuFA,
        # https://github.com/KohakuBlueleaf/KohakuFA (bug 2, scale before shift).
        m_ij = tl.maximum(m_i, qk)  # new running max
        alpha_ij = tl.exp2((qk - m_ij) * exp2_scale)  # attention weight for current edge
        correction = tl.exp2((m_i - m_ij) * exp2_scale)  # correction factor for previous accumulations

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

    # The row maximum and the inverse sum are saved as two numbers for the backward. Folded into one
    # number, m + log(l), the log(l) part would be rounded away once the scores are large, and the
    # probabilities recomputed in the backward would no longer sum to 1. See KohakuBlueleaf, KohakuFA,
    # https://github.com/KohakuBlueleaf/KohakuFA (bug 1, combined row statistic). They sit next to each
    # other, so the backward reads both for a destination node from one stretch of memory.
    stats_off = STATS_ptr + dst_idx * 2 * H + tl.arange(0, H_pad)
    tl.store(stats_off, m_i, mask=H_mask)
    tl.store(stats_off + H, 1.0 / l_i, mask=H_mask)


@triton.jit
def _gt_bwd_dst_pass(
    Q_ptr,
    K_ptr,
    V_ptr,
    E_ptr,
    OUT_ptr,  # saved forward outputs o_i
    STATS_ptr,  # saved row maximum of the unscaled scores and inverse softmax sum, [N_dst, 2, H]
    ROW_ptr,  # [M] (edge -> src)
    COLPTR_ptr,  # [N_dst + 1]
    D_OUT_ptr,  # [N_dst * H * C]
    D_Q_ptr,  # OUT
    D_ptr,  # [N_dst, 2, H] written here: D_j = <d_out, out>, then the correction lam_j (see below)
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
        # store D_j = <d_out, out> = 0, lam_j = 0 and dQ = 0
        zeros = tl.zeros((H_pad,), dtype=tl.float32)
        tl.store(D_ptr + dst_idx * 2 * H + tl.arange(0, H_pad), zeros, mask=H_mask)
        tl.store(D_ptr + dst_idx * 2 * H + H + tl.arange(0, H_pad), zeros, mask=H_mask)
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_Q_ptr + dst_off, zeros, mask=H_C_mask)
        return

    d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    out = tl.load(OUT_ptr + dst_off, H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    # D_j = <d_out, out> for one-pass computation of dQ
    Dj = tl.sum(d_out * out, axis=-1)  # [H]

    q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    dq = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    # The exact score gradients of a node sum to zero over its edges, because D_j equals the
    # attention-weighted sum of dalpha. Here D_j comes from the saved output, whose rounding leaves
    # it off by a little; every score gradient then carries that error times its weight, and dq
    # picks up the error times the weighted mean key, which is far larger than the true gradient
    # once attention is nearly one-hot. The sums of the score gradients, of the weights
    # and of the weighted keys are kept, and lam = the first over the second is the part to take
    # away, so that the corrected gradients sum to zero again. The source pass uses the same lam.
    # See Chen et al., "Broken symmetry in BF16 attention" (GProj), https://arxiv.org/abs/2609.34272.
    ds_sum = tl.zeros((H_pad,), dtype=tl.float32)
    weight_sum = tl.zeros((H_pad,), dtype=tl.float32)
    weight_k = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    m_j = tl.load(STATS_ptr + dst_idx * 2 * H + tl.arange(0, H_pad), mask=H_mask)
    inv_l_j = tl.load(STATS_ptr + dst_idx * 2 * H + H + tl.arange(0, H_pad), mask=H_mask)

    edge_ptr = E_ptr + neigh_start * H * C + H_C_off  # pointer to first edge_attr
    e_idx = neigh_start  # first edge index
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))
    exp2_scale: tl.constexpr = 1.4426950408889634 / tl.sqrt(float(C))  # qk_scale / ln 2, as exp(x) = 2^(x / ln 2)

    # for _ in tl.range(num_edges, warp_specialize=True):
    for _ in range(num_edges):
        e = tl.load(edge_ptr, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        src = tl.load(ROW_ptr + e_idx)
        src_off = src * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        ke = k + e
        # Attention weight before dividing by the softmax sum; the division and the score scale are
        # the same for every edge of this node, so they are applied to dq once after the loop.
        s_ij = tl.sum(q * ke, axis=-1)
        weight_ij = tl.exp2((s_ij - m_j) * exp2_scale)

        v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        ve = v + e

        dalpha = tl.sum(d_out * ve, axis=-1)
        ds_ij = weight_ij * (dalpha - Dj)
        dq += ds_ij[:, None] * ke
        ds_sum += ds_ij
        weight_sum += weight_ij
        weight_k += weight_ij[:, None] * ke

        # move to next edge
        edge_ptr += H * C
        e_idx += 1

    lam = ds_sum / weight_sum
    dq = (dq - lam[:, None] * weight_k) * (inv_l_j * qk_scale)[:, None]

    # store D_j, lam_j and dQ
    tl.store(D_ptr + dst_idx * 2 * H + tl.arange(0, H_pad), Dj, mask=H_mask)
    tl.store(D_ptr + dst_idx * 2 * H + H + tl.arange(0, H_pad), lam, mask=H_mask)
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
    D_ptr,  # [N_dst, 2, H] D_j and lam_j from the dst pass
    STATS_ptr,  # [N_dst * 2 * H] saved row maximum and inverse softmax sum from fwd
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
    exp2_scale: tl.constexpr = 1.4426950408889634 / tl.sqrt(float(C))  # qk_scale / ln 2, as exp(x) = 2^(x / ln 2)

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
        m_j = tl.load(STATS_ptr + dst * 2 * H + tl.arange(0, H_pad))
        inv_l_j = tl.load(STATS_ptr + dst * 2 * H + H + tl.arange(0, H_pad))
        Dj = tl.load(D_ptr + dst * 2 * H + tl.arange(0, H_pad))
        lam_j = tl.load(D_ptr + dst * 2 * H + H + tl.arange(0, H_pad))

        e_off = e_idx * H * C + H_C_off
        e = tl.load(E_ptr + e_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        ke = k + e
        ve = v + e

        # some recomputations from dst-pass
        s_ij = tl.sum(q * ke, axis=-1)
        alpha_ij = tl.exp2((s_ij - m_j) * exp2_scale) * inv_l_j
        dalpha = tl.sum(d_out * ve, axis=-1)
        # Score gradient with the dst pass's correction lam_j, so that it sums to zero there as well.
        dS = alpha_ij * ((dalpha - Dj) - lam_j)

        # per-edge k, v contributions, summing up to per-edge e contribution
        dV_edge = alpha_ij[:, None] * d_out
        dK_edge = (dS * qk_scale)[:, None] * q
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

    Parameters
    ----------
    q : Tensor
        Queries of the destination nodes, ``(N_dst, H, C)``.
    k : Tensor
        Keys of the source nodes, ``(N_src, H, C)``.
    v : Tensor
        Values of the source nodes, ``(N_src, H, C)``.
    e : Tensor
        Edge features in CSC order, ``(M, H, C)``, added to the keys and values of each edge.
    row : Tensor
        Source node of each edge, in CSC order.
    colptr : Tensor
        Start of each destination node's edges in CSC order, ``(N_dst + 1,)``.
    rowptr : Tensor
        Start of each source node's edges in ``edge_ids``, ``(N_src + 1,)``.
    edge_ids : Tensor
        Edges grouped by source node, as indices into the CSC order.
    edge_dst : Tensor
        Destination node of each edge, in CSC order.

    Returns
    -------
    out : Tensor
        Attention output cast back to ``q.dtype`` (the user-facing result).
    out_saved : Tensor
        Float32 attention output, kept for the backward pass.
    stats : Tensor
        Float32 row maximum of the unscaled scores and inverse of the softmax sum, shape
        ``(N_dst, 2, H)``, kept for the backward pass.
    """
    q, k, v, e = (x.contiguous() for x in (q, k, v, e))
    row, colptr = (x.contiguous() for x in (row, colptr))

    N_dst, H, C = q.shape
    out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)
    stats = torch.empty((N_dst, 2, H), device=q.device, dtype=torch.float32)

    _gt_fwd[(N_dst,)](q, k, v, e, stats, row, colptr, out_saved, N_dst, H, C, tl.float32, **_SAME_ROUNDING)

    out = out_saved.to(q.dtype)
    # Custom-op outputs must not alias one another; ``.to`` returns ``self`` when
    # ``q`` is already float32, so clone to keep ``out`` and ``out_saved`` distinct.
    if out is out_saved:
        out = out.clone()

    return out, out_saved, stats


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
    stats = torch.empty((N_dst, 2, H), device=q.device, dtype=torch.float32)
    return out, out_saved, stats


# TODO(Jan): single bwd pass for non-bipartite graphs
@torch.library.custom_op("anemoi::graph_transformer_attention_backward", mutates_args=(), device_types="cuda")
def graph_transformer_attention_backward(
    d_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    e: Tensor,
    out_saved: Tensor,
    stats: Tensor,
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
    D = torch.empty((N_dst, 2, H), device=q.device, dtype=torch.float32)

    # Pass A: destination nodes (computes D, the correction lam and dQ)
    _gt_bwd_dst_pass[(N_dst,)](
        q, k, v, e, out_saved, stats, row, colptr, d_out, dQ, D, N_dst, H, C, grad_dtype, **_SAME_ROUNDING
    )

    # Pass B: source nodes (accumulate dK, dV, dE)
    _gt_bwd_src_pass[(N_src,)](
        q,
        k,
        v,
        e,
        rowptr,
        edge_ids,
        edge_dst,
        D,
        stats,
        d_out,
        dK,
        dV,
        dE,
        N_src,
        H,
        C,
        grad_dtype,
        **_SAME_ROUNDING,
    )

    return dQ, dK, dV, dE


@graph_transformer_attention_backward.register_fake
def _graph_transformer_attention_backward_fake(
    d_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    e: Tensor,
    out_saved: Tensor,
    stats: Tensor,
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


def _graph_transformer_attention_backward(ctx, d_out, _d_out_saved, _d_stats):
    # Only the gradient w.r.t. the user-facing ``out`` is used; ``out_saved`` and
    # ``stats`` are internal saved tensors that are not consumed downstream.
    q, k, v, e, out_saved, stats, row, colptr, rowptr, edge_ids, edge_dst = ctx.saved_tensors

    dQ, dK, dV, dE = graph_transformer_attention_backward(
        d_out, q, k, v, e, out_saved, stats, row, colptr, rowptr, edge_ids, edge_dst
    )

    # Gradients for (q, k, v, e, row, colptr, rowptr, edge_ids, edge_dst).
    return dQ, dK, dV, dE, None, None, None, None, None


def _graph_transformer_attention_setup_context(ctx, inputs, output):
    q, k, v, e, row, colptr, rowptr, edge_ids, edge_dst = inputs
    _out, out_saved, stats = output

    # The forward op makes contiguous copies internally, but those are not the tensors
    # passed here (setup_context receives the original op inputs). Save contiguous
    # versions so the Triton backward kernels, which assume a contiguous layout, receive
    # contiguous inputs.
    q, k, v, e = (x.contiguous() for x in (q, k, v, e))
    row, colptr, rowptr, edge_ids, edge_dst = (x.contiguous() for x in (row, colptr, rowptr, edge_ids, edge_dst))

    ctx.save_for_backward(q, k, v, e, out_saved, stats, row, colptr, rowptr, edge_ids, edge_dst)


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
    out, _out_saved, _stats = graph_transformer_attention(
        query, key, value, edges, row, colptr, rowptr, edge_ids, edge_dst
    )
    return out
