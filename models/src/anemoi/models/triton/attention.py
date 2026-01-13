# (C) Copyright 2026 Anemoi contributors.
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
    BLOCK_N: tl.constexpr,  #
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    WINDOW: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    """Tiled calculation of the attention algorithm.

    Each program has loaded a BLOCK_M sized section of the context Q
    This is stored in shared memory throughout.
    It then loops over K and V in sizes of BLOCK_N until the entire
    BLOCK_M sized output is accumulated

    Optionally, causal or sliding window masking can be performed.

    """
    # range of values handled by this kernel
    if CAUSAL:
        # Attends within the following range
        # X - - - -
        # X X - - -
        # X X X - -
        # X X X X -
        # X X X X X

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

        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k) * qk_scale

        # apply masking
        if WINDOW > 0:
            m_pos = offs_m[:, None]
            n_pos = start_n + offs_n[None, :]
            # Mask condition: keep if (q - window_size <= k <= q + window size)
            mask = (n_pos <= m_pos + WINDOW) & (n_pos >= m_pos - WINDOW)
            qk = tl.where(mask, qk, -1.0e6)

        elif CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, -1.0e6)

        # global attention
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

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

        # Move to next block
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
):
    dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX

    # This confusing bit of code can be ignored
    # Depending on specific settings and what type of GPU is used,
    # (see _system_specific_settings())
    # a different wrapper class on top of tensors is used
    # This is done to support further use of hardware features like TMA
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
        BLOCK_N,  #
        CAUSAL,
        offs_m,
        offs_n,
        N_CTX,  #
        WINDOW,
        warp_specialize,
    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(Out, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):  #  #  #  #
    """Calculates a per-token scalar Delta needed for backwards softmax computation.

    Pre-computing and storing it here prevents repeated recomputation during inner backward computation.
    """
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
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    """Computes dK and dV with respect to dO.

    Each kernel loads a single BLOCK_N1-sized block of K and V into shared memory.
    It then iterates over a column of Q in blocks of BLOCK_M1.
    """
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1

    # This is a single stage kernel, which loops over columns of Q
    # When masking (causal or sliding-window) is used, it must determine where the
    # current block is in the global matrix to determine how it is masked
    # There are 2 options:
    #   fully masked out => skip block
    #   masked => apply MASK to block

    # Calculate the upper and lower bounds when using masking
    if WINDOW > 0:
        # Rather then iterating across the whole context, we iterate
        # Over a window around the position of the K and V blocks
        lo = tl.maximum(0, start_n - WINDOW)
        hi = tl.minimum(N_CTX, (start_n + BLOCK_N1) + WINDOW)
        kv_lower_bound = offs_n[:, None] - WINDOW
        kv_upper_bound = offs_n[:, None] + WINDOW
    elif CAUSAL:
        # lo, hi = start_n, N_CTX
        # start_n, rounded down to lowest multiple if not even
        lo, hi = (start_n // BLOCK_M1) * BLOCK_M1, N_CTX
        # this function doesnt convert to a multiple - it informs the compiler that the first number IS a multiple of the second
        lo = tl.multiple_of(lo, BLOCK_M1)
        hi = tl.multiple_of(hi, BLOCK_M1)

    else:
        lo: tl.constexpr = 0
        hi: tl.constexpr = N_CTX

    # skip up to 'lo'
    qT_ptrs += lo * stride_tok
    do_ptrs += lo * stride_tok

    # for _ in range(num_steps):
    for curr_m in tl.range(lo, hi, step_m, warp_specialize=warp_specialize):

        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        # Apply masking.
        if CAUSAL:
            mask = offs_m[None, :] >= offs_n[:, None]
            qkT = tl.where(mask, qkT, float("-inf"))

        if WINDOW > 0:
            m_pos = offs_m[None, :]
            # Mask condition: keep if (q - window_size <= k <= q + window size)
            mask = (kv_lower_bound <= m_pos) & (kv_upper_bound >= m_pos)
            qkT = tl.where(mask, qkT, float("-inf"))

        # Apply exponent after masking, improves numerical stability and accuracy
        pT = tl.math.exp2(qkT - m[None, :])

        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))

        # Increment pointers.
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
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
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    CAUSAL: tl.constexpr,  #
    WINDOW: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    """Computes dQ with respect to dO.

    Each kernel loads a single BLOCK_M2-sized block of Q into shared memory.
    It then iterates over columns of K and V in blocks of BLOCK_N2.
    """
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

    # This is a single stage kernel, which loops over columns of K and V
    # When masking (causal or sliding-window) is used, it must determine where the
    # current block is in the global matrix to determine how it is masked
    # There are 2 options:
    #   fully masked out => skip block
    #   masked => apply MASK to block

    # Calculate the upper and lower bounds when using masking
    if WINDOW > 0:
        lo = tl.maximum(0, start_m - WINDOW)
        hi = tl.minimum(N_CTX, (start_m + BLOCK_M2) + WINDOW)
        q_lower_bound = offs_m[:, None] - WINDOW
        q_upper_bound = offs_m[:, None] + WINDOW
    elif CAUSAL:
        # hi is block after start_m, rounded up to nearest multiple of step_n
        lo, hi = 0, ((start_m + BLOCK_M2) // BLOCK_N2) * BLOCK_N2
        # this function doesnt convert to a multiple - it informs the compiler that the first number IS a multiple of the second
        lo = tl.multiple_of(lo, BLOCK_N2)
        hi = tl.multiple_of(hi, BLOCK_N2)
    else:
        lo: tl.constexpr = 0
        hi: tl.constexpr = N_CTX

    # skip up to 'lo'
    kT_ptrs += lo * stride_tok
    vT_ptrs += lo * stride_tok

    for curr_n in tl.range(lo, hi, step_n, warp_specialize=warp_specialize):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        # Apply masking based on the position of the current K and V blocks.
        if CAUSAL:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        if WINDOW > 0:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_n[None, :] <= q_upper_bound) & (offs_n[None, :] >= q_lower_bound)
            qk = tl.where(mask, qk, float("-inf"))

        # Apply exponent after masking, improves numerical stability and accuracy
        p = tl.math.exp2(qk - m)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))

        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
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
    HEAD_DIM: tl.constexpr,  #
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    warp_specialize: tl.constexpr,
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

    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        CAUSAL=CAUSAL,  #
        WINDOW=WINDOW,  #
        warp_specialize=warp_specialize,
    )

    # Store computed dK and dV
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # Compute dQ
    start_m = pid * BLOCK_M2
    start_n = 0

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
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        start_n,
        CAUSAL=CAUSAL,  #
        WINDOW=WINDOW,  #
        warp_specialize=warp_specialize,
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


def _system_specific_settings(q, k, v, o, warp_specialize):
    """Provides addtional performance settings for specific systems.

    These performance settings come from the Triton fused attention example.
    """
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_V == HEAD_DIM_K

    extra_kern_args = {}
    # Tuning for AMD target
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    if is_blackwell() and warp_specialize:
        if HEAD_DIM_K == 128 and q.dtype == torch.float16:
            extra_kern_args["maxnreg"] = 168
        else:
            extra_kern_args["maxnreg"] = 80

    # Use device_descriptor for Hopper + warpspec.
    if supports_host_descriptor() and not (is_hopper() and warp_specialize):
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    else:
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

    # Defines how memory is allocated by triton
    # Certain GPUs (e.g. Nvidia Hoppers and Blackwells) support TMA (Tensor Memory Accelerator)
    # TMA is hardware support for async data movement between global and shared memory
    # Basically the relevant memory addresses are calculated by dedicated hardware instead of registers
    # This frees up threads and registers to do other computations
    # TMA requires global memory allocations, so we set the alloc_fn here
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    return desc_q, desc_k, desc_v, desc_o, extra_kern_args


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, window, sm_scale, warp_specialize=False):
        """This function implements a tiled version of the attention algorithm.

        Input matrices are in the shape [BATCH, N_HEAD, N_CTX, HEAD_DIM]

        """
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        if window is None:
            window = 0
        assert not (causal and window > 0), "causal and window not supported in combination"

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        desc_q, desc_k, desc_v, desc_o, extra_kern_args = _system_specific_settings(q, k, v, o, warp_specialize)

        # defines how blocks in the q,k and v input matrices are distributed across SMs on a GPU
        # (SMs are essentially processors on a GPU, with typically 1024 threads per SM)
        # Here a 2D grid is defined: NUM_CTX/BLOCK_M * (BATCH_SIZE * NUM_HEADS)
        # Meaning there is at least (BATCH_SIZE * NUM_HEADS) SMs
        # Depending on BLOCK_M, the context window might also be split across SMs
        # BLOCK_M is a hyperparameter which triton sets at runtime by running small performance tests
        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid

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
            **extra_kern_args,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.window = window
        ctx.warp_specialize = warp_specialize
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
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)

        # precompute 'delta' value needed for softmax computation
        _attn_bwd_preprocess[pre_grid](
            o, do, delta, BATCH, N_HEAD, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #  #  #  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)

        # Compute backward pass.
        # There is at least one kernel per BATCH * NUM_HEAD.
        # Depending on BLOCK_N1, the context dimension can be further split across kernels
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
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,  #
            WINDOW=ctx.window,
            # bwd kernels don't compile with warp_spec=true (GH200, triton 3.4)
            warp_specialize=False,
        )

        return dq, dk, dv, None, None, None, None
