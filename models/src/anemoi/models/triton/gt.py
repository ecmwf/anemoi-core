# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

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
        zeros = tl.zeros((H_pad,), dtype=out_dtype)
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
    tl.store(D_ptr + dst_idx * H + tl.arange(0, H_pad), Dj.to(out_dtype), mask=H_mask)
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
    N_src: tl.constexpr,
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


# TODO(Jan): single bwd pass for non-bipartite graphs


class GraphTransformerFunction(torch.autograd.Function):
    """Custom autograd for GraphTransformer using Triton kernels."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError(
                "Error. The 'triton' backend was selected for the GraphTransformer but 'torch.cuda.is_available()' returned 'False'. The 'triton' backend is currently only supported on GPUs. To run on other device types, please select a different backend for the GraphTransformer in the models config. If you intend to run on GPUs, please ensure your torch install supports running on GPUs."
            )

    @staticmethod
    def forward(ctx, q, k, v, e, csc, reverse):
        """Args:
        q: [N_dst, H, C]
        k: [N_src, H, C]
        v: [N_src, H, C]
        e: [num_edges, H, C]
        csc: (row, colptr)
        reverse: (rowptr, edge_ids, edge_dst)
        """
        row, colptr = csc
        rowptr, edge_ids, edge_dst = reverse

        # Ensure contiguous memory layout for Triton
        q, k, v, e = [x.contiguous() for x in (q, k, v, e)]
        row, colptr, rowptr, edge_ids, edge_dst = [x.contiguous() for x in (row, colptr, rowptr, edge_ids, edge_dst)]

        N_dst, H, C = q.shape
        out = torch.empty_like(q)
        m = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

        def torch_dtype_to_triton(dtype):
            if dtype == torch.float16:
                return tl.float16
            elif dtype == torch.bfloat16:
                return tl.bfloat16
            elif dtype == torch.float32:
                return tl.float32
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        out_dtype = torch_dtype_to_triton(q.dtype)
        ctx.out_dtype = out_dtype

        _gt_fwd[(N_dst,)](q, k, v, e, m, row, colptr, out, N_dst, H, C, out_dtype)

        # Save tensors for backward
        ctx.save_for_backward(q, k, v, e, out, m, row, colptr, rowptr, edge_ids, edge_dst)
        return out

    @staticmethod
    def backward(ctx, d_out):
        d_out = d_out.contiguous()
        q, k, v, e, out, m, row, colptr, rowptr, edge_ids, edge_dst = ctx.saved_tensors

        N_dst, H, C = q.shape
        N_src = k.shape[0]

        # Allocate grads and intermediates
        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        dE = torch.empty_like(e)
        D = torch.empty((N_dst, H), device=q.device, dtype=q.dtype)

        # Pass A: destination nodes (computes D and dQ)
        _gt_bwd_dst_pass[(N_dst,)](q, k, v, e, out, m, row, colptr, d_out, dQ, D, N_dst, H, C, ctx.out_dtype)

        # Pass B: source nodes (accumulate dK, dV, dE)
        _gt_bwd_src_pass[(N_src,)](
            q, k, v, e, rowptr, edge_ids, edge_dst, D, m, d_out, dK, dV, dE, N_src, H, C, ctx.out_dtype
        )

        return dQ, dK, dV, dE, None, None
