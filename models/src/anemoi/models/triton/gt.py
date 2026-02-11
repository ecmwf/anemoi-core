# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

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


from anemoi.models.triton.norm import _layer_norm_bwd, _layer_norm_fwd, _rms_norm_bwd
from anemoi.models.triton.norm import _rms_norm_fwd
from anemoi.models.triton.utils import build_masks_and_offsets
from anemoi.models.triton.utils import torch_dtype_to_triton

LOGGER = logging.getLogger(__name__)


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
    qk_norm: tl.constexpr,  # int, 0: no normalisation, 1: rms normalisation, 2: layer normalisation
    elementwise_affine: tl.constexpr,  # bool, whether to apply learnable elementwise affine after normalisation (only relevant if qk_norm=True)
    w_q_norm_ptr,  # ptr [C] or None, depending on qk_norm and elementwise_affine
    w_k_norm_ptr,  # ptr [C] or None, depending on qk_norm and elementwise_affine
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

    if qk_norm > 0:
        C_pad_off = tl.arange(0, C_pad)
        if elementwise_affine:
            # w_k_norm and w_q_norm should be pointers to tensors
            assert w_k_norm_ptr is not None and w_q_norm_ptr is not None
            w_q_norm = tl.load(w_q_norm_ptr + C_pad_off, mask=(C_pad_off < C)).to(tl.float32)
            w_k_norm = tl.load(w_k_norm_ptr + C_pad_off, mask=(C_pad_off < C)).to(tl.float32)
        else:
            w_q_norm = None
            w_k_norm = None
        if qk_norm == 1:  # rms norm
            q = _rms_norm_fwd(q, w_q_norm, C, elementwise_affine)
        elif qk_norm == 2:  # layer norm
            q = _layer_norm_fwd(q, w_q_norm, C, elementwise_affine)

    acc = tl.zeros((H_pad, C_pad), dtype=tl.float32)  # output accumulator, pending normalization by l_i
    l_i = tl.zeros((H_pad,), dtype=tl.float32)  # sum of attention weights
    m_i = tl.full((H_pad,), value=-float("inf"), dtype=tl.float32)  # running max for stability

    # helpers to avoid repeated computations/indexing:
    edge_ptr = E_ptr + neigh_start * H * C + H_C_off  # pointer to first edge_attr
    e_idx = neigh_start  # first edge index

    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for _ in range(num_edges):
        e = tl.load(edge_ptr, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        # src neighbor index: rowptr[e_idx]
        src_idx = tl.load(ROW_ptr + e_idx)

        src_off = src_idx * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        if qk_norm > 0:
            if qk_norm == 1:  # rms norm
                k = _rms_norm_fwd(k, w_k_norm, C, elementwise_affine)
            elif qk_norm == 2:  # layer norm
                k = _layer_norm_fwd(k, w_k_norm, C, elementwise_affine)

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
    qk_norm: tl.constexpr,  # int, 0: no normalisation, 1: rms normalisation, 2: layer normalisation
    elementwise_affine: tl.constexpr,  # bool, whether to apply learnable elementwise affine after normalisation (only relevant if qk_norm=True)
    w_q_norm_ptr,  # ptr [C] or None, depending on qk_norm
    w_k_norm_ptr,  # ptr [C] or None, depending on qk_norm
    D_w_q_norm_partial_ptr,  # ptr [C] or None, depending on qk_norm
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

    if qk_norm > 0:
        # Q is saved unnormalised,
        # Since normalisation was done in the forward pass, we must renormalise it here
        # before we recompute elements of the attention forward pass
        q_unnorm = q  # need to save an unnormalised copy for the bwd pass later
        if elementwise_affine:
            # w_k_norm and w_q_norm should be pointers to tensors
            assert w_k_norm_ptr is not None and w_q_norm_ptr is not None
            C_pad_off = tl.arange(0, C_pad)
            w_q_norm = tl.load(w_q_norm_ptr + C_pad_off, mask=(C_pad_off < C)).to(tl.float32)
            # Load the weights for k_norm outside of the inner loop
            w_k_norm = tl.load(w_k_norm_ptr + C_pad_off, mask=(C_pad_off < C)).to(tl.float32)
        else:
            w_q_norm = None
            w_k_norm = None
        if qk_norm == 1:  # rms norm
            q = _rms_norm_fwd(q, w_q_norm, C, elementwise_affine)
        elif qk_norm == 2:  # layer norm
            q = _layer_norm_fwd(q, w_q_norm, C, elementwise_affine)

    dq = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    edge_ptr = E_ptr + neigh_start * H * C + H_C_off  # pointer to first edge_attr
    e_idx = neigh_start  # first edge index
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for _ in range(num_edges):
        e = tl.load(edge_ptr, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        src = tl.load(ROW_ptr + e_idx)
        src_off = src * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask).to(tl.float32).reshape((H_pad, C_pad))

        # Normalise k if required
        if qk_norm == 1:  # rms norm
            k = _rms_norm_fwd(k, w_k_norm, C, elementwise_affine)
        elif qk_norm == 2:  # layer norm
            k = _layer_norm_fwd(k, w_k_norm, C, elementwise_affine)

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

    if qk_norm > 0:
        # Compute the backward pass of RMS norm
        if qk_norm == 1:  # rms norm
            dq, dw_q_norm = _rms_norm_bwd(q_unnorm, w_q_norm, dq, C, elementwise_affine)
        elif qk_norm == 2:  # layer norm
            dq, dw_q_norm = _layer_norm_bwd(q_unnorm, w_q_norm, dq, C, elementwise_affine)
        if elementwise_affine:
            C_pad_off = tl.arange(0, C_pad)
            tl.store(D_w_q_norm_partial_ptr + C_pad_off, dw_q_norm, mask=(C_pad_off < C))

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
    qk_norm: tl.constexpr,  # int, 0: no normalisation, 1: rms normalisation, 2: layer norm
    elementwise_affine: tl.constexpr,  # bool, whether to apply learnable elementwise affine after normalisation (only relevant if qk_norm=True)
    w_q_norm_ptr,  # ptr [C] or None, depending on qk_norm and elementwise_affine
    w_k_norm_ptr,  # ptr [C] or None, depending on qk_norm and elementwise_affine
    D_w_k_norm_partial_ptr,  # ptr [C] or None, depending on qk_norm and elementwise_affine
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
    if qk_norm > 0:  # perform rms or layer normalisation on Q and K
        # K is saved unnormalised,
        # Since normalisation was done in the fwd pass, we must renormalise now
        k_unnorm = k  # must save copy of unnormalised value for bwd pass later
        if elementwise_affine:
            # w_k_norm and w_q_norm should be pointers to tensors
            assert w_k_norm_ptr is not None and w_q_norm_ptr is not None
            # Load the weights for q_norm outside of the inner loop
            C_pad_off = tl.arange(0, C_pad)
            w_q_norm = tl.load(w_q_norm_ptr + C_pad_off, mask=(C_pad_off < C)).to(tl.float32)
            w_k_norm = tl.load(w_k_norm_ptr + C_pad_off, mask=(C_pad_off < C)).to(tl.float32)
        else:
            w_q_norm = None
            w_k_norm = None
        if qk_norm == 1:  # rms norm
            k = _rms_norm_fwd(k, w_k_norm, C, elementwise_affine)
        elif qk_norm == 2:  # layer norm
            k = _layer_norm_fwd(k, w_k_norm, C, elementwise_affine)

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
        if qk_norm == 1:  # rms norm
            q = _rms_norm_fwd(q, w_q_norm, C, elementwise_affine)
        elif qk_norm == 2:  # layer norm
            q = _layer_norm_fwd(q, w_q_norm, C, elementwise_affine)
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

    if qk_norm > 0:
        # Compute the backward pass of RMS Norm
        if qk_norm == 1:  # rms norm
            accK, dw_k_norm = _rms_norm_bwd(k_unnorm, w_k_norm, accK, C, elementwise_affine)
        elif qk_norm == 2:  # layer norm
            accK, dw_k_norm = _layer_norm_bwd(k_unnorm, w_k_norm, accK, C, elementwise_affine)
        if elementwise_affine:
            C_pad_off = tl.arange(0, C_pad)
            tl.store(D_w_k_norm_partial_ptr + C_pad_off, dw_k_norm, mask=(C_pad_off < C))

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
    def forward(ctx, q, k, v, e, csc, reverse, qk_norm, w_qnorm, w_knorm):
        """Args:
        q: [N_dst, H, C]
        k: [N_src, H, C]
        v: [N_src, H, C]
        e: [num_edges, H, C]
        csc: (row, colptr)
        reverse: (rowptr, edge_ids, edge_dst)
        qk_norm: int, 0: no normalisation, 1: rms normalisation, 2: layer normalisation
        w_qnorm_ptr: [C] or None
        w_knorm_ptr: [C] or None
        """
        row, colptr = csc
        rowptr, edge_ids, edge_dst = reverse

        # Ensure contiguous memory layout for Triton
        q, k, v, e = [x.contiguous() for x in (q, k, v, e)]
        row, colptr, rowptr, edge_ids, edge_dst = [x.contiguous() for x in (row, colptr, rowptr, edge_ids, edge_dst)]

        N_dst, H, C = q.shape
        out = torch.empty_like(q)
        m = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

        out_dtype = torch_dtype_to_triton(q.dtype)
        ctx.out_dtype = out_dtype

        # Set up qk_normalisation
        ctx.qk_norm = qk_norm
        elementwise_affine = w_qnorm is not None and w_knorm is not None
        ctx.elementwise_affine = elementwise_affine

        _gt_fwd[(N_dst,)](
            q, k, v, e, m, row, colptr, out, N_dst, H, C, out_dtype, qk_norm, elementwise_affine, w_qnorm, w_knorm
        )

        # Save tensors for backward
        ctx.save_for_backward(q, k, v, e, out, m, row, colptr, rowptr, edge_ids, edge_dst, w_qnorm, w_knorm)
        return out

    @staticmethod
    def backward(ctx, d_out):
        d_out = d_out.contiguous()
        q, k, v, e, out, m, row, colptr, rowptr, edge_ids, edge_dst, w_qnorm, w_knorm = ctx.saved_tensors

        N_dst, H, C = q.shape
        N_src = k.shape[0]
        
        # Allocate grads and intermediates
        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        dE = torch.empty_like(e)

        dW_qnorm_partial = None
        dW_knorm_partial = None
        if ctx.elementwise_affine:
            assert (
                w_qnorm is not None and w_knorm is not None
            ), "Expected w_qnorm and w_knorm to be not None when elementwise_affine is True"
            dW_qnorm_partial = torch.zeros(
                (
                    N_dst,
                    C,
                ),
                device=d_out.device,
                dtype=torch.float32,
            )
            dW_knorm_partial = torch.zeros(
                (
                    N_src,
                    C,
                ),
                device=d_out.device,
                dtype=torch.float32,
            )

        D = torch.empty((N_dst, H), device=q.device, dtype=q.dtype)

        # Pass A: destination nodes (computes D and dQ)
        _gt_bwd_dst_pass[(N_dst,)](
            q,
            k,
            v,
            e,
            out,
            m,
            row,
            colptr,
            d_out,
            dQ,
            D,
            N_dst,
            H,
            C,
            ctx.out_dtype,
            ctx.qk_norm,
            ctx.elementwise_affine,
            w_qnorm,
            w_knorm,
            dW_qnorm_partial,
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
            m,
            d_out,
            dK,
            dV,
            dE,
            N_src,
            H,
            C,
            ctx.out_dtype,
            ctx.qk_norm,
            ctx.elementwise_affine,
            w_qnorm,
            w_knorm,
            dW_knorm_partial,
        )

        dW_qnorm = None
        dW_knorm = None
        if ctx.elementwise_affine:
            raise NotImplementedError(
                "elementwise_affine=True is not currently supported in the backward pass due to performance reasons. If you want elementwise affine normalisation, please open a ticket on the anemoi-core repository to request this feature."
            )

        return dQ, dK, dV, dE, None, None, None, dW_qnorm, dW_knorm


class GraphTransformer(torch.nn.Module):
    def __init__(self, dim: int, qk_norm: int = 0, elementwise_affine: bool = False):
        super().__init__()
        self.dim = dim
        self.qk_norm = qk_norm
        assert qk_norm in (0, 1, 2), "qk_norm must be 0 (no normalisation), 1 (RMS normalisation) or 2 (layer normalisation)"

        if qk_norm == 0 and elementwise_affine:
            raise ValueError(
                "elementwise_affine=True is not supported when qk_norm=False, since there is no normalisation. Please set elementwise_affine=False if you do not want to use normalisation, or set qk_norm=True if you want to use normalisation with learnable weights."
            )

        elif qk_norm > 0 and elementwise_affine:
            raise NotImplementedError(
                "elementwise_affine=True is not currently supported when qk_norm=True due to performance reasons. If you want elementwise affine normalisation, please open a ticket on the anemoi-core repository to request this feature."
            )

        if self.qk_norm > 0:
            LOGGER.info("Using fused QK norm in triton GT")
            
        if elementwise_affine:
            self.w_qnorm = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
            self.w_knorm = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
        else:
            self.register_parameter("w_qnorm", None)
            self.register_parameter("w_knorm", None)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edges_csc: Tensor,
        csc: tuple[Tensor, Tensor],
        reverse: tuple[Tensor, Tensor, Tensor],
    ) -> torch.Tensor:

        return GraphTransformerFunction.apply(
            query, key, value, edges_csc, csc, reverse, self.qk_norm, self.w_qnorm, self.w_knorm
        )
