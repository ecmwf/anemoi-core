# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import triton
import triton.language as tl


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

    dst_start = dst_idx * H * C
    dst_off = dst_start + tl.arange(0, H * C)

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    if num_edges == 0:
        zeros = tl.zeros((H,), dtype=tl.float32)  # m initialised as torch.float32
        tl.store(M_ptr + dst_idx * H + tl.arange(0, H), zeros)
        zeros = tl.zeros((H * C,), dtype=out_dtype)
        tl.store(OUT_ptr + dst_off, zeros)
        return

    q = tl.load(Q_ptr + dst_off).to(tl.float32).reshape((H, C))
    acc = tl.zeros((H, C), dtype=tl.float32)  # output accumulator, pending normalization by l_i
    l_i = tl.zeros((H,), dtype=tl.float32)  # sum of attention weights
    m_i = tl.full((H,), value=-float("inf"), dtype=tl.float32)  # running max for stability

    # helpers to avoid repeated computations/indexing:
    qk_scale = 1 / tl.sqrt(float(C))
    edge_ptr = E_ptr + neigh_start * H * C + tl.arange(0, H * C)  # pointer to first edge_attr
    e_idx = neigh_start  # first edge index

    for _ in range(num_edges):  # iterate over incident edges
        e = tl.load(edge_ptr).to(tl.float32).reshape((H, C))

        # src neighbor index: rowptr[e_idx]
        src_idx = tl.load(ROW_ptr + e_idx)
        src_off = src_idx * H * C + tl.arange(0, H * C)
        k = tl.load(K_ptr + src_off).to(tl.float32).reshape((H, C))
        v = tl.load(V_ptr + src_off).to(tl.float32).reshape((H, C))

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
            H * C,
        ),
    )

    # store m_i + log(l_i) for backward
    m_start = dst_idx * H
    m_off = m_start + tl.arange(0, H)

    m_i += tl.log(l_i)
    tl.store(M_ptr + m_off, m_i)


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

    qk_scale = 1.0 / tl.sqrt(float(C))
    dst_off = dst_idx * H * C + tl.arange(0, H * C)

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    if num_edges == 0:
        # store D_j = <d_out, out> = 0 and dQ = 0
        zeros = tl.zeros((H,), dtype=out_dtype)
        tl.store(D_ptr + dst_idx * H + tl.arange(0, H), zeros)
        zeros = tl.zeros((H * C,), dtype=out_dtype)
        tl.store(D_Q_ptr + dst_off, zeros)
        return

    d_out = tl.load(D_OUT_ptr + dst_off).to(tl.float32).reshape((H, C))
    out = tl.load(OUT_ptr + dst_off).to(tl.float32).reshape((H, C))

    # D_j = <d_out, out> for one-pass computation of dQ
    Dj = tl.sum(d_out * out, axis=-1)  # [H]

    q = tl.load(Q_ptr + dst_off).to(tl.float32).reshape((H, C))
    dq = tl.zeros((H, C), dtype=tl.float32)

    edge_ptr = E_ptr + neigh_start * H * C + tl.arange(0, H * C)  # pointer to first edge_attr
    e_idx = neigh_start  # first edge index

    for _ in range(num_edges):
        e = tl.load(edge_ptr).to(tl.float32).reshape((H, C))

        src = tl.load(ROW_ptr + e_idx)
        src_off = src * H * C + tl.arange(0, H * C)
        k = tl.load(K_ptr + src_off).to(tl.float32).reshape((H, C))

        ke = k + e
        # score and alpha using saved M
        m_j = tl.load(M_ptr + dst_idx * H + tl.arange(0, H)).to(tl.float32)
        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        alpha_ij = tl.exp(s_ij - m_j)

        v = tl.load(V_ptr + src_off).to(tl.float32).reshape((H, C))
        ve = v + e

        dalpha = tl.sum(d_out * ve, axis=-1)
        dS = alpha_ij * (dalpha - Dj)

        dq += dS[:, None] * ke * qk_scale

        # move to next edge
        edge_ptr += H * C
        e_idx += 1

    # store D_j and dQ
    tl.store(D_ptr + dst_idx * H + tl.arange(0, H), Dj.to(out_dtype))
    tl.store(
        D_Q_ptr + dst_off,
        dq.to(out_dtype).reshape(
            H * C,
        ),
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

    start = tl.load(ROWPTR_ptr + src_idx)
    end = tl.load(ROWPTR_ptr + src_idx + 1)
    num_edges = end - start

    if num_edges == 0:
        zeros = tl.zeros((H * C,), dtype=out_dtype)
        tl.store(D_K_ptr + src_idx * H * C + tl.arange(0, H * C), zeros)
        tl.store(D_V_ptr + src_idx * H * C + tl.arange(0, H * C), zeros)
        return

    # src-side k, v (shared for all edges)
    src_off = src_idx * H * C + tl.arange(0, H * C)
    k = tl.load(K_ptr + src_off).to(tl.float32).reshape((H, C))
    v = tl.load(V_ptr + src_off).to(tl.float32).reshape((H, C))

    accK = tl.zeros((H, C), dtype=tl.float32)
    accV = tl.zeros((H, C), dtype=tl.float32)

    # note that edges aren't necessarily contiguous in memory here, use EDGE_IDS_ptr
    for i in range(num_edges):
        # indexing into edge list + corresponding dst node
        e_idx = tl.load(EDGE_IDS_ptr + start + i)
        dst = tl.load(EDGE_DST_ptr + e_idx)

        # get saved tensors for dst node
        dst_off = dst * H * C + tl.arange(0, H * C)
        q = tl.load(Q_ptr + dst_off).to(tl.float32).reshape((H, C))
        d_out = tl.load(D_OUT_ptr + dst_off).to(tl.float32).reshape((H, C))
        m_j = tl.load(M_ptr + dst * H + tl.arange(0, H)).to(tl.float32)
        Dj = tl.load(D_ptr + dst * H + tl.arange(0, H)).to(tl.float32)

        e_off = e_idx * H * C + tl.arange(0, H * C)
        e = tl.load(E_ptr + e_off).to(tl.float32).reshape((H, C))

        ke = k + e
        ve = v + e

        # some recomputations from dst-pass
        s_ij = tl.sum(q * ke, axis=-1) * (1.0 / tl.sqrt(float(C)))
        alpha_ij = tl.exp(s_ij - m_j)
        dalpha = tl.sum(d_out * ve, axis=-1)
        dS = alpha_ij * (dalpha - Dj)

        # per-edge k, v contributions, summing up to per-edge e contribution
        dV_edge = alpha_ij[:, None] * d_out
        dK_edge = dS[:, None] * q * (1.0 / tl.sqrt(float(C)))
        dE_edge = dV_edge + dK_edge

        tl.store(
            D_E_ptr + e_off,
            dE_edge.to(out_dtype).reshape(
                H * C,
            ),
        )

        accK += dK_edge
        accV += dV_edge

    # write final accumulated per-src grads
    tl.store(
        D_K_ptr + src_off,
        accK.to(out_dtype).reshape(
            H * C,
        ),
    )
    tl.store(
        D_V_ptr + src_off,
        accV.to(out_dtype).reshape(
            H * C,
        ),
    )


# TODO(Jan): single bwd pass for non-bipartite graphs


class GraphTransformerFunction(torch.autograd.Function):
    """Custom autograd for GraphTransformer using Triton kernels."""

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
