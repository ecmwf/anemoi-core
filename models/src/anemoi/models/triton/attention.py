# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
#
# It is based on the fused attention example from the Triton-lang github repo (MIT license) (Credits: OpenAI kernel team)
# https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
#
# It has been extended to support sliding window attention.


import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def is_hip():
    return torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    desc_k,
    desc_v,  #
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    WINDOW: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    # range of values handled by this block
    if CAUSAL:

        # Attends within the following range
        # X - - - -
        # X X - - -
        # X X X - -
        # X X X X -
        # X X X X X

        # here we are trading code complexity for speed
        # in the official fused attn example from triton
        # they call the kernel once for off-band (e.g. full blocks) and once for
        # on-band (partial blocks) and only apply the mask in the partial block
        lo, hi = 0, (start_m + 1) * BLOCK_M
    elif WINDOW > 0:
        # Attends within the following range (Assuming W=1)
        # X X - - -
        # X X X - -
        # - X X X -
        # - - X X X
        # - - - X X

        lo = tl.maximum(0, (start_m * BLOCK_M) - WINDOW)
        hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M + WINDOW)

        # round up to lowest multiple if not even
        if lo % BLOCK_N != 0:
            lo = (lo // BLOCK_N) * BLOCK_N
        # round up to highest multiple if not even
        if hi % BLOCK_N != 0:
            hi = (hi // BLOCK_N) * BLOCK_N + BLOCK_N

        # this function doesnt convert to a multiple - it informs the compiler that the first number IS a multiple of the second
        lo = tl.multiple_of(lo, BLOCK_N)  
        hi = tl.multiple_of(hi, BLOCK_N)
    else:
        # Attends within the following range
        # X X X X X
        # X X X X X
        # X X X X X
        # X X X X X
        # X X X X X

        # here we are trading code complexity for speed
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)

        # Apply masking
        if WINDOW > 0:
            #TODO check if a speedup can be obtained by breaking window computation
            # into stages based on a tiles position (skip, partial or fullly attend)
            n_pos = start_n + offs_n[None, :]
            m_pos = offs_m[:, None]
            # Mask condition: keep if (q - window_size <= k <= q + window size)
            mask = (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW)
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        elif CAUSAL:
            # could be optimised
            # Only have to compute the mask when start >= start_m * BLOCK_M (boundary block)
            # But currently it doesnt compile with the extra condition
            # An alternate implemenation would be the triton-fused-attention technique
            # calling _attn_fwd_inner multiple times with different 'stage' parameters
            # corresponding to a blocks position in the seq_len^2 array
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        else:
            # global attention
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


def _generate_configs():
    if is_hip():
        NUM_STAGES_OPTIONS = [1]
    elif supports_host_descriptor():
        NUM_STAGES_OPTIONS = [2, 3, 4]
    else:
        NUM_STAGES_OPTIONS = [2, 3, 4]

    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook)
        for BM in [64, 128]
        for BN in [32, 64, 128]
        for s in NUM_STAGES_OPTIONS
        for w in [4, 8]
    ]

    if "PYTEST_VERSION" in os.environ:
        # Use a single config in testing for reproducibility
        configs = [
            triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
        ]
    return configs


def _keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def _prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=list(filter(_keep, _generate_configs())),
    key=["N_CTX", "HEAD_DIM", "warp_specialize"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    WINDOW: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    IS_HOPPER: tl.constexpr,  #
):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM]
    )
    desc_v = _maybe_make_tensor_desc(
        desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM]
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM]
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM]
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        desc_k,
        desc_v,  #
        offset_y,
        dtype,
        start_m,
        qk_scale,  #
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,  #
        CAUSAL,
        offs_m,
        offs_n,
        N_CTX,  #
        WINDOW,
        warp_specialize,
        IS_HOPPER,
    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(Out, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):  #  #  #  #
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(Out + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1

    # TODO change num_steps and start_m based on window, currently we have kernels which launch and do nothing in for loop

    # if CAUSAL:
    # This kernel is doing a causal attention calculation
    # We must determine where this block is in the global matrix
    # To determine if its treated
    # There are 3 options:
    #   fully masked out (curr_m < start_n) => skip block
    #   partially masked (start_n <= curr_m  <= start_n + BLOCK_N1) => apply MASK to block
    #   fully including (curr_m > start_n + BLOCK_N1) => include fully
    # TODO precompute list of size num_steps with [skip,skip,partial,full,full]

    # if WINDOW > 0:
    # CAUSAL=False #TODO move this much further up or replace with assert

    # fully masked out: curr_m < lo, curr_m > hi
    # partially masked: lo <= curr_m < lo + BLOCK_N1,  hi - BLOCK_N1 <= curr_m < hi
    # fully included: lo + BLOCK_N1 <= curr_m <= hi - BLOCK_N1

    # Calculate the upper and lower bounds when using sliding window
    if WINDOW > 0:
        # end_m = start_m + (step_m * num_steps)
        # we have a fixed block of n and we would normally iterate over an entire row on m
        # with sliding window, we iterate less, based on the postion of the fixed n block
        lo = tl.maximum(0, start_n - WINDOW)
        hi = tl.minimum(N_CTX, (start_n + BLOCK_N1) + WINDOW)

    skip_block = False
    partial_block = False

    for blk_idx in range(num_steps):

        if CAUSAL:
            if curr_m < start_n:
                skip_block = True
            elif curr_m >= start_n and curr_m <= start_n + BLOCK_N1:
                partial_block = True

        if WINDOW > 0:
            if curr_m < lo or curr_m > hi:
                skip_block = True
            # partially masked: lo <= curr_m < lo + BLOCK_N1,  hi - BLOCK_N1 <= curr_m < hi
            # if (curr_m >= lo and curr_m < lo + BLOCK_N1) or (curr_m >= hi - BLOCK_N1 and curr_m < hi):
            else:
                partial_block = True

        if not skip_block:
            qT = tl.load(qT_ptrs)
            # Load m before computing qk to reduce pipeline stall.
            offs_m = curr_m + tl.arange(0, BLOCK_M1)
            m = tl.load(M + offs_m)
            qkT = tl.dot(k, qT)
            pT = tl.math.exp2(qkT - m[None, :])
            # Autoregressive masking.
            if partial_block:
                if CAUSAL:
                    # fwd causal partial block (moving n, fixed m): offs_m[:, None] >= (start_n + offs_n[None, :])
                    mask = offs_m[None, :] >= offs_n[:, None]
                    pT = tl.where(mask, pT, 0.0)

                if WINDOW > 0:
                    n_pos = offs_n[:, None]
                    m_pos = offs_m[None, :]
                    # Mask condition: keep if (q - window_size <= k <= q + window size)
                    # fwd (moving n, fixed m opposit to here): (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW)
                    mask = (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW)
                    pT = tl.where(mask, pT, 0.0)

            do = tl.load(do_ptrs)
            # Compute dV.
            ppT = pT
            ppT = ppT.to(tl.float16)
            dv += tl.dot(ppT, do)
            # D (= delta) is pre-divided by ds_scale.
            Di = tl.load(D + offs_m)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
            dsT = pT * (dpT - Di[None, :])
            dsT = dsT.to(tl.float16)
            dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        # reset block mode
        skip_block = False
        partial_block = False
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    CAUSAL: tl.constexpr,  #
    WINDOW: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2

    # if CAUSAL:
    # This kernel is doing a causal attention calculation
    # We must determine where this block is in the global matrix
    # To determine if its treated
    # There are 3 options:
    #   fully masked out (curr_n < start_m) => skip block
    #   partially masked (start_m <= curr_n  <= start_m + BLOCK_M2) => apply MASK to block (start_m )
    #   fully including (curr_n > start_m + BLOCK_M2) => include fully
    # TODO precompute list of size num_steps with [skip,skip,partial,full,full]

    # if WINDOW > 0:
    # CAUSAL=False #TODO move this much further up or replace with assert

    # fully masked out: curr_n < lo, curr_n > hi
    # partially masked: lo <= curr_n < lo + BLOCK_M2,  hi - BLOCK_M2 <= curr_n < hi
    # fully included: lo + BLOCK_M2 <= curr_n <= hi - BLOCK_M2

    # Calculate the upper and lower bounds when using sliding window
    if WINDOW > 0:
        # end_m = start_m + (step_m * num_steps)
        # we have a fixed block of n and we would normally iterate over an entire row on m
        # with sliding window, we iterate less, based on the postion of the fixed n block
        lo = tl.maximum(0, start_m - WINDOW)
        hi = tl.minimum(N_CTX, (start_m + BLOCK_M2) + WINDOW)

    skip_block = False
    partial_block = False

    for blk_idx in range(num_steps):

        if CAUSAL:
            if curr_n > start_m + BLOCK_M2:
                skip_block = True
            elif curr_n >= start_m and curr_n <= start_m + BLOCK_M2:
                partial_block = True

        # copied directly from dkdv, but might have to change signs like was done for causal
        if WINDOW > 0:
            if curr_n < lo or curr_n > hi:
                skip_block = True
            # partially masked: lo <= curr_m < lo + BLOCK_M2,  hi - BLOCK_M2 <= curr_m < hi
            # if (curr_n >= lo and curr_n < lo + BLOCK_M2) or (curr_n >= hi - BLOCK_M2 and curr_n < hi):
            else:
                partial_block = True

        if not skip_block:
            kT = tl.load(kT_ptrs)
            vT = tl.load(vT_ptrs)
            qk = tl.dot(q, kT)
            p = tl.math.exp2(qk - m)
            # Autoregressive masking.
            if partial_block:

                # dkdv
                # if CAUSAL:
                #    mask = (offs_m[None, :] >= offs_n[:, None])

                # if WINDOW > 0:
                #    n_pos = offs_n[:, None]
                #    m_pos = offs_m[None, :]
                # Mask condition: keep if (q - window_size <= k <= q + window size)
                #    mask = (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW)

                offs_n = curr_n + tl.arange(0, BLOCK_N2)
                n_pos = offs_n[None, :]
                m_pos = offs_m[:, None]

                if CAUSAL:
                    mask = m_pos >= n_pos
                    p = tl.where(mask, p, 0.0)

                if WINDOW > 0:
                    # fwd (fixed m, moving n) (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW)
                    # mask = (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW) #original
                    mask = (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW)  # original
                    # mask = (n_pos > m_pos + WINDOW) & (n_pos < m_pos - WINDOW)
                    # mask = (m_pos >= n_pos + WINDOW) & (n)
                    p = tl.where(mask, p, 0.0)

            # Compute dP and dS.
            dp = tl.dot(do, vT).to(tl.float32)
            ds = p * (dp - Di[:, None])
            ds = ds.to(tl.float16)
            # Compute dQ.
            # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
            dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
        # reset block mode
        skip_block = False
        partial_block = False
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR #was used in original bwd pass when in partial causal block
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        CAUSAL=CAUSAL,  #
        WINDOW=WINDOW,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    # MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR # was used in original bwd pass when in partial causal block
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        start_n,
        num_steps,  #
        CAUSAL=CAUSAL,  #
        WINDOW=WINDOW,  #
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, window, sm_scale, warp_specialize=False):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        if window is None:
            window = 0
        assert not (causal and window > 0) #causal and window together not supported

        #TODO refactor hip and tensor descriptor into functions
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_v = TensorDescriptor(
                v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block
            )
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        #defines how memory is allocated by triton
        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")
        triton.set_allocator(alloc_fn)

        #defines how blocks in the q,k and v input matrices are distributed across SMs on a GPU
        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](
            sm_scale,
            M,  #
            q.shape[0],
            q.shape[1],  #
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            WINDOW=window,
            CAUSAL=causal,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.window = window
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors

        if not do.is_contiguous():
            do = do.contiguous()

        if do.shape == o.shape and do.stride() != o.stride():
            do = do.reshape(o.shape).contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do, delta, BATCH, N_HEAD, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #  #  #  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,  #
            WINDOW=ctx.window,
        )

        return dq, dk, dv, None, None, None, None
