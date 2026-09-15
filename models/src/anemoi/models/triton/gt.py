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
    D_E_ptr,  # OUT [M * H * C] per-edge dE; stores are contiguous in CSC order here
    ALPHA_ptr,  # OUT [M * H] fp32 per-edge alpha, written to CSR slots for the src pass
    DS_ptr,  # OUT [M * H] fp32 per-edge dS (alpha folded in), written to CSR slots
    CSRPOS_ptr,  # [M] CSC edge id -> CSR slot; makes the src pass's reads contiguous
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
        # no incident edges: dQ = 0, and there are no per-edge outputs to write
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_Q_ptr + dst_off, zeros, mask=H_C_mask)
        return

    d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    out = tl.load(OUT_ptr + dst_off, H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

    # D_j = <d_out, out> for one-pass computation of dQ
    Dj = tl.sum(d_out * out, axis=-1)  # [H]

    q = tl.load(Q_ptr + dst_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
    dq = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    # m_j is per-destination, so it is loaded once instead of on every edge
    m_j = tl.load(M_ptr + dst_idx * H + tl.arange(0, H_pad), mask=H_mask).to(tl.float32)

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
        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        alpha_ij = tl.exp(s_ij - m_j)

        v = tl.load(V_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))
        ve = v + e

        dalpha = tl.sum(d_out * ve, axis=-1)
        dS = alpha_ij * (dalpha - Dj)

        dq += dS[:, None] * ke * qk_scale

        # Hand (alpha, dS) over to the src pass: both are computed here anyway, so
        # the src pass then needs neither its k/v/e/m/D loads nor a second q.ke dot
        # product. They are scattered to CSR slots, which turns the src pass's
        # per-edge reads into a contiguous stream -- a scattered store is
        # fire-and-forget, whereas a scattered load would sit on the critical path
        # of the latency-bound src pass.
        pos = tl.load(CSRPOS_ptr + e_idx)
        ah_off = pos * H + tl.arange(0, H_pad)
        tl.store(ALPHA_ptr + ah_off, alpha_ij, mask=H_mask)
        tl.store(DS_ptr + ah_off, dS, mask=H_mask)

        # dE is written here too, where the per-edge stores are contiguous in CSC
        # order; the src pass would have had to scatter them.
        dV_edge = alpha_ij[:, None] * d_out
        dK_edge = dS[:, None] * q * qk_scale
        tl.store(
            D_E_ptr + e_idx * H * C + H_C_off,
            (dV_edge + dK_edge)
            .to(out_dtype)
            .reshape(
                H_pad * C_pad,
            ),
            mask=H_C_mask,
        )

        # move to next edge
        edge_ptr += H * C
        e_idx += 1

    # store dQ
    tl.store(
        D_Q_ptr + dst_off,
        dq.to(out_dtype).reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )


@triton.jit
def _gt_bwd_src_pass(
    Q_ptr,  # [N_dst * H * C]
    ROWPTR_ptr,  # [N_src+1]
    EDGE_DST_ptr,  # [M] dst node per CSR slot (contiguous reads)
    ALPHA_ptr,  # [M * H] fp32 per-edge alpha from the dst pass, CSR order
    DS_ptr,  # [M * H] fp32 per-edge dS (alpha folded in) from the dst pass, CSR order
    D_OUT_ptr,  # [N_dst * H * C]
    D_K_ptr,  # OUT [N_src * H * C]
    D_V_ptr,  # OUT [N_src * H * C]
    N_src,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Accumulate dK and dV per source node from the dst pass's per-edge (alpha, dS).

    Everything this pass needs is precomputed by ``_gt_bwd_dst_pass``, so it only
    gathers the dst-side ``q``/``d_out`` rows and streams ``alpha``/``dS`` in CSR
    order. ``dE`` is written by the dst pass, where those stores are contiguous.
    """
    src_idx = tl.program_id(0)
    if src_idx >= N_src:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    start = tl.load(ROWPTR_ptr + src_idx)
    end = tl.load(ROWPTR_ptr + src_idx + 1)
    num_edges = end - start

    src_off = src_idx * H * C + H_C_off

    if num_edges == 0:
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_K_ptr + src_off, zeros, mask=H_C_mask)
        tl.store(D_V_ptr + src_off, zeros, mask=H_C_mask)
        return

    accK = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    accV = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    # Tiled over the source node's edge list: this kernel launches few programs, each
    # running a long serial chain of indirect gathers (edge_dst -> q/d_out row), and
    # tiling lets BLOCK_E of those gathers be in flight at once. The forward and dst
    # passes are already bandwidth-saturated and gain nothing from the same treatment,
    # but this one runs well below the HBM floor.
    num_tiles = (num_edges + BLOCK_E - 1) // BLOCK_E
    for t in range(num_tiles):
        csr_slots = start + t * BLOCK_E + tl.arange(0, BLOCK_E)
        e_mask = csr_slots < end
        dst = tl.load(EDGE_DST_ptr + csr_slots, mask=e_mask, other=0)

        # [BLOCK_E, H_pad * C_pad] gathers of the dst-side rows. The H/C padding mask
        # only exists when H or C is not a power of two; combining it with a plain
        # `True` would not broadcast, hence the constexpr branches.
        row_offs = dst[:, None] * H * C + H_C_off[None, :]
        if H == H_pad and C == C_pad:
            row_mask = e_mask[:, None]
        else:
            row_mask = e_mask[:, None] & H_C_mask[None, :]

        ah_offs = csr_slots[:, None] * H + tl.arange(0, H_pad)[None, :]
        if H == H_pad:
            ah_mask = e_mask[:, None]
        else:
            ah_mask = e_mask[:, None] & H_mask[None, :]

        q = tl.load(Q_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32).reshape((BLOCK_E, H_pad, C_pad))
        d_out = tl.load(D_OUT_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32).reshape((BLOCK_E, H_pad, C_pad))

        # contiguous streaming reads (CSR order); masked-off lanes contribute zero
        alpha_ij = tl.load(ALPHA_ptr + ah_offs, mask=ah_mask, other=0.0)
        dS = tl.load(DS_ptr + ah_offs, mask=ah_mask, other=0.0)

        accK += tl.sum(tl.reshape(dS, (BLOCK_E, H_pad, 1)) * q, axis=0) * qk_scale
        accV += tl.sum(tl.reshape(alpha_ij, (BLOCK_E, H_pad, 1)) * d_out, axis=0)

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
    edge_dst_csr: Tensor,
    csr_pos: Tensor,
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
    edge_dst_csr: Tensor,
    csr_pos: Tensor,
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
    edge_dst_csr: Tensor,
    csr_pos: Tensor,
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

    # Per-edge (alpha, dS) handoff from the dst pass to the src pass. The dst pass
    # computes both anyway, which lets the src pass drop its k/v/e/m/D loads and its
    # recomputation of the scores. Kept in CSR order so the src pass reads them as a
    # contiguous stream. This costs 2 * M * H fp32, i.e. 2/C of the dE allocated above.
    num_edges = row.shape[0]
    alpha_e = torch.empty((num_edges, H), device=q.device, dtype=torch.float32)
    dS_e = torch.empty((num_edges, H), device=q.device, dtype=torch.float32)

    # Pass A: destination nodes (computes dQ and dE, and emits the per-edge handoff)
    _gt_bwd_dst_pass[(N_dst,)](
        q, k, v, e, out_saved, m, row, colptr, d_out, dQ, dE, alpha_e, dS_e, csr_pos, N_dst, H, C, grad_dtype
    )

    # Pass B: source nodes (accumulate dK, dV)
    # The tile width follows the mean source degree: high-degree sources are few
    # programs with long gather chains and profit from concurrent gathers, whereas at
    # degree ~1 the masked-off lanes are pure overhead.
    block_e = 4 if num_edges >= 4 * N_src else 1
    _gt_bwd_src_pass[(N_src,)](
        q, rowptr, edge_dst_csr, alpha_e, dS_e, d_out, dK, dV, N_src, H, C, grad_dtype, BLOCK_E=block_e
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
    m: Tensor,
    row: Tensor,
    colptr: Tensor,
    rowptr: Tensor,
    edge_dst_csr: Tensor,
    csr_pos: Tensor,
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
    q, k, v, e, out_saved, m, row, colptr, rowptr, edge_dst_csr, csr_pos = ctx.saved_tensors

    dQ, dK, dV, dE = graph_transformer_attention_backward(
        d_out, q, k, v, e, out_saved, m, row, colptr, rowptr, edge_dst_csr, csr_pos
    )

    # Gradients for (q, k, v, e, row, colptr, rowptr, edge_dst_csr, csr_pos).
    return dQ, dK, dV, dE, None, None, None, None, None


def _graph_transformer_attention_setup_context(ctx, inputs, output):
    q, k, v, e, row, colptr, rowptr, edge_dst_csr, csr_pos = inputs
    _out, out_saved, m = output

    # The forward op makes contiguous copies internally, but those are not the tensors
    # passed here (setup_context receives the original op inputs). Save contiguous
    # versions so the Triton backward kernels, which assume a contiguous layout, receive
    # contiguous inputs.
    q, k, v, e = (x.contiguous() for x in (q, k, v, e))
    row, colptr, rowptr, edge_dst_csr, csr_pos = (x.contiguous() for x in (row, colptr, rowptr, edge_dst_csr, csr_pos))

    ctx.save_for_backward(q, k, v, e, out_saved, m, row, colptr, rowptr, edge_dst_csr, csr_pos)


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
    reverse: tuple[Tensor, Tensor, Tensor, Tensor],
) -> Tensor:
    """torch.compile-friendly GraphTransformer attention."""
    row, colptr = csc
    # edge_ids is only needed to build the two CSR-ordered tensors below
    rowptr, _edge_ids, edge_dst_csr, csr_pos = reverse
    out, _out_saved, _m = graph_transformer_attention(
        query, key, value, edges, row, colptr, rowptr, edge_dst_csr, csr_pos
    )
    return out
