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
import math
from triton.tools.tensor_descriptor import TensorDescriptor #TODO(cathal) doesnt seem to work for torch 2.7 and less, guard 

from anemoi.models.triton.utils import is_blackwell
from anemoi.models.triton.utils import is_hip
from anemoi.models.triton.utils import supports_host_descriptor
from anemoi.models.triton.utils import torch_dtype_to_triton

def set_allocator():
    """Defines how memory is allocated by triton.
    Certain GPUs (e.g. Nvidia Hoppers and Blackwells) support TMA (Tensor Memory Accelerator)
    TMA is hardware support for async data movement between global and shared memory
    Basically the relevant memory addresses are calculated by dedicated hardware instead of registers
    This frees up threads and registers to do other computations
    TMA requires global memory allocations, so we set the alloc_fn here
    """
    #TODO(cathal) Update to check if allocator has been set before attempting to set
    # Currently there isn't a stable way to do this via Triton
    
    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    
    triton.set_allocator(alloc_fn)
    
set_allocator()


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    desc_k,
    desc_v,  #
    iter_offset,
    dtype: tl.constexpr,
    start_fixed,
    qk_scale,  #
    BLOCK_FIXED: tl.constexpr,
    BLOCK_ITER: tl.constexpr,  #
    CAUSAL: tl.constexpr,
    offs_fixed: tl.constexpr,
    offs_iter: tl.constexpr,  #
    N_CTX: tl.constexpr,
    WINDOW: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """Tiled calculation of the attention algorithm.

    Each program has loaded a BLOCK_FIXED sized section of the context Q
    This is stored in shared memory throughout.
    It then loops over K and V in sizes of BLOCK_ITER until the entire
    BLOCK_FIXED sized output is accumulated

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

        lo, hi = 0, tl.minimum((start_fixed + 1) * BLOCK_FIXED, N_CTX)
        hi = tl.multiple_of(hi, BLOCK_FIXED)
        
        # round down to lowest multiple of BLOCK_ITER 
        lo = (lo // BLOCK_ITER) * BLOCK_ITER
    elif WINDOW > 0:
        # Attends within the following range (Assuming W=1)
        # X X - - -
        # X X X - -
        # - X X X -
        # - - X X X
        # - - - X X

        lo = tl.maximum(0, (start_fixed * BLOCK_FIXED) - WINDOW)
        hi = tl.minimum(N_CTX, (start_fixed + 1) * BLOCK_FIXED + WINDOW)

        # round down to lowest multiple if not even
        if lo % BLOCK_ITER != 0:
            lo = (lo // BLOCK_ITER) * BLOCK_ITER
        # round up to highest multiple if not even
        #if hi % BLOCK_ITER != 0:
        #    hi = (hi // BLOCK_ITER) * BLOCK_ITER + BLOCK_ITER
        
        # Apply bounds and inform compiler about multiples
        lo = tl.multiple_of(lo, BLOCK_ITER)
        # Only assert hi is a multiple if it wasn't clamped by N_CTX
        #if hi <= N_CTX:
        #    hi = tl.multiple_of(hi, BLOCK_ITER)
        #else:
        #    hi = tl.minimum(hi, N_CTX)
    else:
        # Attends within the following range
        # X X X X X
        # X X X X X
        # X X X X X
        # X X X X X
        # X X X X X

        lo, hi = 0, N_CTX
    
    #TODO(cathal) change based on dtype
    MINUS_INF : tl.constexpr = -1.0e8
    
    iter_offset = iter_offset + lo
    # loop over k, v and update accumulator
    for curr_iter in tl.range(lo, hi, BLOCK_ITER, warp_specialize=WARP_SPECIALIZE):
        curr_iter = tl.multiple_of(curr_iter, BLOCK_ITER)  # Tells compiler curr_iter is a multiple of BLOCK_ITER
        mask_iter = curr_iter + offs_iter
        
        # -- compute qk ----
        #k = desc_k.load([iter_offset, 0]).T
        k = desc_k.load([iter_offset, 0])#.T
        # transpose access to mask_iter bc k is transposed
        #k = tl.where(mask_iter[None, :] < N_CTX, k, 0.0)  # mask out-of-bounds k values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED
        k = tl.where(mask_iter[:, None] < N_CTX, k, 0.0)  # mask out-of-bounds k values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        
        # Apply out-of-bounds masking first
        qk = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, qk, MINUS_INF)

        # apply masking
        if WINDOW > 0:
            fixed_pos = offs_fixed[:, None]
            iter_pos = curr_iter + offs_iter[None, :]
            # Mask condition: keep if (q - window_size <= k <= q + window size)
            mask = (iter_pos <= fixed_pos + WINDOW) & (iter_pos >= fixed_pos - WINDOW)
            qk = tl.where(mask, qk, MINUS_INF)

        elif CAUSAL:
            fixed_pos = offs_fixed[:, None]
            iter_pos = curr_iter + offs_iter[None, :]
            mask = fixed_pos >= iter_pos
            qk = tl.where(mask, qk, MINUS_INF)

        # global attention
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])

        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij) 
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v = desc_v.load([iter_offset, 0])
        v = tl.where(mask_iter[:, None] < N_CTX, v, 0.0)  # mask out-of-bounds v values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)

        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        #l_i = l_i * alpha + l_ij # not sure if this is right
        #l_i_new = tl.exp2(lse_i - m_ij) + l_ij
        #lse_i = m_ij + tl.log(l_i_new)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Move to next block
        iter_offset += BLOCK_ITER
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    """Updates the tensor descriptor to use the block size determined by autotuning"""
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        # If the input is not a tensor descriptor, return
        return

    BLOCK_FIXED = nargs["BLOCK_FIXED"]
    BLOCK_ITER = nargs["BLOCK_ITER"]
    HEAD_DIM = nargs["HEAD_DIM"]
    # Must determine of this is operating on fwd or bwd pass
    if "desc_o" in nargs:  # FWD pass
        nargs["desc_q"].block_shape = [BLOCK_FIXED, HEAD_DIM]
        nargs["desc_v"].block_shape = [BLOCK_ITER, HEAD_DIM]
        nargs["desc_k"].block_shape = [BLOCK_ITER, HEAD_DIM]
        nargs["desc_o"].block_shape = [BLOCK_FIXED, HEAD_DIM]
    else:  # BWD pass

        # Must determine if this is for bwd_dkdv or _bwd_dq to
        # determine which tensors have fixed and iterating blocks

        # Determine kernel by checking kernel args
        if "desc_dq" in nargs:  # _bwd_dq
            nargs["desc_q"].block_shape = [BLOCK_FIXED, HEAD_DIM]
            nargs["desc_v"].block_shape = [BLOCK_ITER, HEAD_DIM]
            nargs["desc_k"].block_shape = [BLOCK_ITER, HEAD_DIM]
            nargs["desc_do"].block_shape = [BLOCK_FIXED, HEAD_DIM]
            nargs["desc_dq"].block_shape = [BLOCK_FIXED, HEAD_DIM]

        else:  # _bwd_dkdv
            nargs["desc_q"].block_shape = [BLOCK_ITER, HEAD_DIM]
            nargs["desc_v"].block_shape = [BLOCK_FIXED, HEAD_DIM]
            nargs["desc_k"].block_shape = [BLOCK_FIXED, HEAD_DIM]
            nargs["desc_do"].block_shape = [BLOCK_ITER, HEAD_DIM]
            nargs["desc_dv"].block_shape = [BLOCK_FIXED, HEAD_DIM]
            nargs["desc_dk"].block_shape = [BLOCK_FIXED, HEAD_DIM]


def _generate_configs(try_warp_spec=True):
    """Generates a list of runtime configs for triton autotuning.

    Returns a list of different performance-related hyperparams

    Triton will benchmark all the configs after initalising and will
    run with the best config.

    backward config is more constrained. warp-spec doesnt compile in mnay cases (triton v3.4 on GH200)
    so it is disabled by default in bwd
    """
    if is_hip():
        NUM_STAGES_OPTIONS = [1]
    else:
        NUM_STAGES_OPTIONS = [2, 3, 4]

    configs = [
        triton.Config(
            {"BLOCK_FIXED": BM, "BLOCK_ITER": BN, "WARP_SPECIALIZE": WS},
            num_stages=s,
            num_warps=w,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BM in [16, 32, 64, 128]
        for BN in [16, 32, 64, 128]
        for s in NUM_STAGES_OPTIONS
        for w in [4, 8]
        for WS in ([True, False] if try_warp_spec else [False])
    ]

    if "PYTEST_VERSION" in os.environ:
        # Use a single config in testing for reproducibility
        # Use BLOCK_FIXED=16 to avoid issues with small N_CTX in tests
        configs = [ 
            triton.Config(
                dict(BLOCK_FIXED=16, BLOCK_ITER=16, WARP_SPECIALIZE=False),
                num_stages=1,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            )
        ]
    return configs


def _prune_invalid_configs_fwd(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    HEAD_DIM = kwargs["HEAD_DIM"]

    # Filter out configs where BLOCK_FIXED > N_CTX
    # And configs where BLOCK_* does not divide evenly into N_CTX
    valid_configs=[conf for conf in configs if (conf.kwargs.get("BLOCK_FIXED", 0) <= N_CTX)]
        
    if len(valid_configs) == 0:
        raise ValueError(f"Autotuning found no valid runtime configurations for your setup (mode=fwd, {N_CTX=}, {HEAD_DIM=}). Please use an alternate attention backend or open a ticket on the anemoi-core repository describing your use case.")

    return valid_configs


def _prune_invalid_configs_bwd(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    HEAD_DIM = kwargs["HEAD_DIM"]
    
    valid_configs = [
        conf
        for conf in configs
        if (
            # Filter out configs where BLOCK_* > N_CTX
            conf.kwargs.get("BLOCK_FIXED", 0) <= N_CTX
            and conf.kwargs.get("BLOCK_ITER", 0) <= N_CTX
            # BLOCK_FIXED must divide evenly into BLOCK_ITER (static assert in _bwd_dvdk and _bwd_dq)
            and conf.kwargs.get("BLOCK_FIXED", 0) % conf.kwargs.get("BLOCK_ITER", 0) == 0
            # and configs where the block size does not divide evenly into the context
            and N_CTX % conf.kwargs.get("BLOCK_FIXED", 0) == 0 and N_CTX % conf.kwargs.get("BLOCK_ITER", 0) == 0
        )
    ]
    
    if len(valid_configs) == 0:
        raise ValueError(f"Autotuning found no valid configurations for your setup (mode=bwd, {N_CTX=}, {HEAD_DIM=}). Please use an alternate attention backend or open a ticket on the anemoi-core repository describing your use case.")

    return valid_configs


@triton.jit
def _maybe_make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=_generate_configs(),
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs_fwd},
    cache_results=True,
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    WINDOW: tl.constexpr,  #
    BLOCK_FIXED: tl.constexpr,  #
    BLOCK_ITER: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    dtype: tl.constexpr,
    n_ctx_rounded: tl.constexpr,
):
    tl.static_assert(BLOCK_ITER <= HEAD_DIM)
                     
    start_fixed = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # number of elements in all non-head-dim dimensions (use rounded value for actual tensor layout)
    y_dim = Z * H * n_ctx_rounded

    # If tensor_descriptors weren't created on host (in system_specific_settings())
    # Then they are created here
    desc_q = _maybe_make_tensor_descriptor(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_descriptor(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_ITER, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_descriptor(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_ITER, HEAD_DIM],
    )
    output_block_size = BLOCK_FIXED
    if (start_fixed + 1) * BLOCK_FIXED > N_CTX: 
        output_block_size = N_CTX - start_fixed * BLOCK_FIXED
    desc_o = _maybe_make_tensor_descriptor(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[output_block_size, HEAD_DIM],
    )
    
    # Offsets into the start of the tensor which this kernel will iterate over
    # Use n_ctx_rounded for calculating offsets since tensor is padded
    iter_offset = off_z * (n_ctx_rounded * H) + off_h * n_ctx_rounded
    # Offsets into the region of the tensor which this kernel will keep in memory
    fixed_offset = iter_offset + start_fixed * BLOCK_FIXED  # (used for Tensor Descriptors)
    # initialize offset pointers
    offs_fixed = start_fixed * BLOCK_FIXED + tl.arange(0, BLOCK_FIXED)  # (used for normal tensors)
    offs_iter = tl.arange(0, BLOCK_ITER)
    
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_FIXED], dtype=tl.float32) - float("inf") # list of maximum values, will be updated after each BLOCK_ITER. used to compute online softmax
    l_i = tl.zeros([BLOCK_FIXED], dtype=tl.float32) #- float("inf") #+ 1.0
    acc = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2) #hack to make calculating exponent faster, by merging 1/ln(2) now the cheaper exp2() fn can be called later instead of exp()
    # load q: it will stay in SRAM throughout
    q = desc_q.load([fixed_offset, 0])
    q = tl.where(offs_fixed[:, None] < N_CTX, q, 0.0)  # mask out-of-bounds q values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        desc_k,
        desc_v,  #
        iter_offset,
        dtype,
        start_fixed,
        qk_scale,  #
        BLOCK_FIXED,
        BLOCK_ITER,  #
        CAUSAL,
        offs_fixed,
        offs_iter,
        N_CTX,  #
        WINDOW,
        WARP_SPECIALIZE,
    )
    # epilogue
    m_i += tl.math.log2(l_i)
    #o_scale = tl.exp(m_i - l_i)
    # Guard against division by zero when entire rows are masked out
    #acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    acc = acc / l_i[:, None]
    #acc = acc * o_scale[:,None]
    #m_ptrs = M + off_hz * N_CTX + offs_fixed
    #ctx_rounded = N_CTX
    #if N_CTX % BLOCK_FIXED != 0:
    #    ctx_rounded = (N_CTX // BLOCK_FIXED + 1) * BLOCK_FIXED
    m_ptrs = M + off_hz * N_CTX + offs_fixed 
    #m_ptrs = M + off_hz * n_ctx_rounded + offs_fixed 
    tl.store(m_ptrs, m_i, mask=offs_fixed < N_CTX) # TODO(cathal) might be overwiriting block boundarys in m where offs_fixed goes beyond ctx and into the next
    #acc = tl.where(offs_fixed[:, None] < N_CTX, acc, 0.0)  # mask out-of-bounds output values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED
    desc_o.store([fixed_offset, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(Out, DO, Delta, N_CTX, PRE_BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):  #  #  #  #
    """Calculates a per-token scalar Delta needed for backwards softmax computation.

    Pre-computing and storing it here prevents repeated recomputation during inner backward computation.
    """
    off_m = tl.program_id(0) * PRE_BLOCK + tl.arange(0, PRE_BLOCK)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(Out + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=off_m[:, None] < N_CTX, other=0.0).to(tl.float32)
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :], mask=off_m[:, None] < N_CTX, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.autotune(
    # Autotuning is crucial to get good performance at larger head dims
    # For an o96 2048c configuration, got a 3x speedup from autotuning
    configs=_generate_configs(try_warp_spec=False),
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs_bwd},
    cache_results=True,
)
@triton.jit
def _attn_bwd_dkdv(
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_dk,
    desc_dv,
    M,
    L,
    D,
    sm_scale: tl.constexpr,
    # shared by Q/K/V/DO.
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
    BLOCK_FIXED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Computes dK and dV with respect to dO.

    Each kernel loads a single BLOCK_FIXED-sized block of K and V into shared memory.
    It then iterates over a column of Q in blocks of BLOCK_ITER.
    """

    # index into context
    pid = tl.program_id(0)
    start_fixed = pid * BLOCK_FIXED
    start_iter = 0

    # Index into head and batch
    off_hz = tl.program_id(2)
    off_chz = (off_hz * N_CTX).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    iter_offset = off_z * (N_CTX * H) + off_h * N_CTX
    fixed_offset = iter_offset + start_fixed

    # number of elements in all non-head-dim dimensions
    y_dim: tl.constexpr = Z * H * N_CTX
    desc_q = _maybe_make_tensor_descriptor(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_ITER, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_descriptor(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_descriptor(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )
    desc_do = _maybe_make_tensor_descriptor(
        desc_do,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_ITER, HEAD_DIM],
    )

    # When working with N_CTX values that are not divisible by BLOCK_FIXED, we need to adjust the block size of the output tensors in the last block, to avoid writing out-of-bounds values.

    output_block_size = BLOCK_FIXED
    if (start_fixed + 1) * BLOCK_FIXED > N_CTX: 
        output_block_size = N_CTX - start_fixed * BLOCK_FIXED
    desc_dk = _maybe_make_tensor_descriptor(
        desc_dk,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[output_block_size, HEAD_DIM],
    )
    desc_dv = _maybe_make_tensor_descriptor(
        desc_dv,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[output_block_size, HEAD_DIM],
    )

    # offset pointers for batch/head
    M += off_chz
    L += off_chz
    D += off_chz

    offs_fixed = start_fixed + tl.arange(0, BLOCK_FIXED)

    dv = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = desc_k.load([fixed_offset, 0])
    v = desc_v.load([fixed_offset, 0])

    # Create offset pointers for tensors
    offs_iter = start_iter + tl.arange(0, BLOCK_ITER)
    offs_fixed = start_fixed + tl.arange(0, BLOCK_FIXED)

    tl.static_assert(BLOCK_FIXED % BLOCK_ITER == 0)
    curr_iter = start_iter
    
    #TODO(cathal) change based on dtype
    #if dtype == tl.bfloat16:
        #MINUS_INF : tl.constexpr = -1.0e8
    #elif dtype == tl.float16:
    MINUS_INF : tl.constexpr = -1.0e8

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
        lo = tl.maximum(0, start_fixed - WINDOW) 
        hi = tl.minimum(N_CTX, (start_fixed + BLOCK_FIXED) + WINDOW)
        kv_lower_bound: tl.constexpr = offs_fixed[:, None] - WINDOW
        kv_upper_bound: tl.constexpr = offs_fixed[:, None] + WINDOW
        
        #align lo and hi to nearest BLOCK_ITER
        lo = (lo // BLOCK_ITER) * BLOCK_ITER
        hi = ((hi + BLOCK_ITER - 1) // BLOCK_ITER) * BLOCK_ITER
        lo = tl.multiple_of(lo, BLOCK_ITER)
        hi = tl.multiple_of(hi, BLOCK_ITER)
    elif CAUSAL:
        # lo, hi = start_n, N_CTX
        # start_fixed, rounded down to lowest multiple if not even
        lo = (start_fixed // BLOCK_ITER) * BLOCK_ITER
        hi = N_CTX
        # this function doesnt convert to a multiple - it informs the compiler that the first number IS a multiple of the second
        lo = tl.multiple_of(lo, BLOCK_ITER)
        hi = tl.multiple_of(hi, BLOCK_ITER)

    else:
        lo: tl.constexpr = 0
        hi: tl.constexpr = N_CTX

    # skip up to 'lo'
    iter_offset += lo

    offs_iter = iter_offset + tl.arange(0, BLOCK_ITER)
    
    for curr_iter in tl.range(lo, hi, BLOCK_ITER, warp_specialize=WARP_SPECIALIZE):

        mask_iter = curr_iter + offs_iter
        qT = desc_q.load([iter_offset, 0]).T
        qT = tl.where(mask_iter[None, :] < N_CTX, qT, 0.0)  # mask out-of-bounds q values to 0, so they dont contribute to output. This is needed when N_CTX is not divisible by BLOCK_FIXED

        # Load m before computing qk to reduce pipeline stall.
        
        m = tl.load(M + offs_iter, mask=mask_iter < N_CTX, other=0.0)  # mask out-of-bounds m values, since they will not contribute to output when N_CTX is not divisible by BLOCK_FIXED
        qkT = tl.dot(k, qT)
        qkT = tl.where((mask_iter[:, None] < N_CTX), qkT, MINUS_INF)
        # Apply masking.
        if CAUSAL:
            mask = offs_iter[None, :] >= offs_fixed[:, None]
            qkT = tl.where(mask, qkT, MINUS_INF)

        if WINDOW > 0:
            iter_pos = offs_iter[None, :]
            # Mask condition: keep if (q - window_size <= k <= q + window size)
            mask = (kv_lower_bound <= iter_pos) & (kv_upper_bound >= iter_pos)
            qkT = tl.where(mask, qkT, MINUS_INF)
            
        # Apply exponent after masking, improves numerical stability and accuracy
        pT = tl.math.exp2(qkT - m[None, :])
        
        do = desc_do.load([iter_offset, 0])
        # Compute dV.
        ppT = pT.to(dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_iter, mask=mask_iter < N_CTX, other=0.0)  # mask out-of-bounds D values to 0, so they dont contribute to output. This is needed when N_CTX is not divisible by BLOCK_FIXED
        Di = tl.where(mask_iter < N_CTX, Di, 0.0)  # mask out-of-bounds D values to 0, so they dont contribute to output. This is needed when N_CTX is not divisible by BLOCK_FIXED
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(dtype)
        dk += tl.dot(dsT, tl.trans(qT))

        # Move to next iter block
        iter_offset += BLOCK_ITER

    # Store computed dK and dV
    # when N_CTX is not divisible by BLOCK_FIXED, we may have computed values for out-of-bounds positions, 
    # In this case, we set the block size of desc_d[vk] to be smaller in the tail case, such that the store only writes the in-bounds values.
    desc_dv.store([fixed_offset, 0], dv.to(dtype))
    dk *= sm_scale 
    desc_dk.store([fixed_offset, 0], dk.to(dtype))


@triton.autotune(
    # Autotuning is crucial to get good performance at larger head dims
    # For an o96 2048c configuration, got a 3x speedup from autotuning
    configs=_generate_configs(try_warp_spec=False),
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs_bwd},
    cache_results=True,
)
@triton.jit
def _attn_bwd_dq(
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_dq,
    M,
    D,
    # shared by Q/K/V/DO.
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
    BLOCK_FIXED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Computes dQ with respect to dO.

    Each kernel loads a single BLOCK_FIXED-sized block of Q into shared memory.
    It then iterates over columns of K and V in blocks of BLOCK_ITER.
    """

    #LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    LN2: tl.constexpr = 0.6931471805599453  # = ln(2)

    # index into context
    pid = tl.program_id(0)
    start_fixed = pid * BLOCK_FIXED
    start_iter = 0

    # Index into head and batch
    off_hz = tl.program_id(2)
    off_chz = (off_hz * N_CTX).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    iter_offset = off_z * (N_CTX * H) + off_h * N_CTX
    fixed_offset = iter_offset + start_fixed

    # How many elements in the first 3 dimensaions
    y_dim = Z * H * N_CTX
    # Build tensor descriptors if they havent already been built in _system_specific_settings()
    desc_q = _maybe_make_tensor_descriptor(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )
    desc_v = _maybe_make_tensor_descriptor(
        desc_v,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_ITER, HEAD_DIM],
    )
    desc_k = _maybe_make_tensor_descriptor(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_ITER, HEAD_DIM],
    )
    desc_do = _maybe_make_tensor_descriptor(
        desc_do,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )
    # When working with N_CTX values that are not divisible by BLOCK_FIXED, we need to adjust the block size of the output tensors in the last block, to avoid writing out-of-bounds values.

    output_block_size = BLOCK_FIXED
    if (start_fixed + 1) * BLOCK_FIXED > N_CTX: 
        output_block_size = N_CTX - start_fixed * BLOCK_FIXED
    desc_dq = _maybe_make_tensor_descriptor(
        desc_dq,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[output_block_size, HEAD_DIM],
    )

    # offset pointers for batch/head
    M += off_chz
    D += off_chz

    # generate offset array for fixed block
    offs_fixed = start_fixed + tl.arange(0, BLOCK_FIXED)

    q = desc_q.load([fixed_offset, 0])
    dq = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)
    do = desc_do.load([fixed_offset, 0])

    m = tl.load(M + offs_fixed)
    m = m[:, None]

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_fixed)
    tl.static_assert(BLOCK_FIXED % BLOCK_ITER == 0)
    curr_iter = start_iter
    
    #TODO(cathal) change based on dtype
    MINUS_INF : tl.constexpr = -1.0e8

    # This is a single stage kernel, which loops over columns of K and V
    # When masking (causal or sliding-window) is used, it must determine where the
    # current block is in the global matrix to determine how it is masked
    # There are 2 options:
    #   fully masked out => skip block
    #   masked => apply MASK to block

    # Calculate the upper and lower bounds when using masking
    if WINDOW > 0:
        lo= tl.maximum(0, start_fixed - WINDOW)
        hi= tl.minimum(N_CTX, (start_fixed + BLOCK_FIXED) + WINDOW)
        q_lower_bound: tl.constexpr = offs_fixed[:, None] - WINDOW
        q_upper_bound: tl.constexpr = offs_fixed[:, None] + WINDOW
        
        #align lo and hi to nearest BLOCK_ITER
        lo = (lo // BLOCK_ITER) * BLOCK_ITER
        hi = ((hi + BLOCK_ITER - 1) // BLOCK_ITER) * BLOCK_ITER
        lo = tl.multiple_of(lo, BLOCK_ITER)
        hi = tl.multiple_of(hi, BLOCK_ITER)
    elif CAUSAL:
        # hi is block after start_m, rounded up to nearest multiple of step_n
        lo = 0
        hi = (start_fixed + BLOCK_FIXED)
        
        # round hi up to nearest multiple of BLOCK_ITER
        hi = ((hi + BLOCK_ITER - 1) // BLOCK_ITER) * BLOCK_ITER
        # this function doesnt convert to a multiple - it informs the compiler that the first number IS a multiple of the second
        lo = tl.multiple_of(lo, BLOCK_ITER)
        hi = tl.multiple_of(hi, BLOCK_ITER)
    else:
        lo: tl.constexpr = 0
        hi: tl.constexpr = N_CTX

    # skip up to 'lo'
    iter_offset += lo
    offs_iter = iter_offset + tl.arange(0, BLOCK_ITER)

    for curr_iter in tl.range(lo, hi, BLOCK_ITER, warp_specialize=WARP_SPECIALIZE):
        curr_iter = tl.multiple_of(curr_iter, BLOCK_ITER)  # Tells compiler curr_iter is a multiple of BLOCK_ITER
        mask_iter = curr_iter + offs_iter

        kT = desc_k.load([iter_offset, 0]).T
        vT = desc_v.load([iter_offset, 0]).T
        kT = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, kT, 0.0)  # mask out-of-bounds k values to 0, so they dont contribute to output when N_CTX is not divisible by BLOCK_FIXED
        vT = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, vT, 0.0)  # mask out-of-bounds v values to 0, so they dont contribute to output when N_CTX is not divisible by BLOCK_FIXED
        qk = tl.dot(q, kT)
        qk = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, qk, MINUS_INF)  # mask out-of-bounds qk values, since they will not contribute to output when N_CTX is not divisible by BLOCK_FIXED
        # Apply masking based on the position of the current K and V blocks.
        if CAUSAL:
            iter_pos = offs_iter[None, :]
            fixed_pos = offs_fixed[:, None]
            mask = fixed_pos >= iter_pos
            qk = tl.where(mask, qk, MINUS_INF)

        if WINDOW > 0:
            iter_pos = offs_iter[None, :]
            mask = (iter_pos <= q_upper_bound) & (iter_pos >= q_lower_bound)
            qk = tl.where(mask, qk, MINUS_INF)

        # Apply exponent after masking, improves numerical stability and accuracy
        p = tl.math.exp2(qk - m)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))

        # move to the next iter_block
        iter_offset += BLOCK_ITER

    # Write back dQ.
    dq *= LN2
    # to avoid writing out of bounds when N_CTX is not divisible by BLOCK_FIXED, the block size of desc_dq is set to be smaller in the last block, so we only write the in-bounds values
    desc_dq.store([fixed_offset, 0], dq.to(dtype))


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

    # If host descriptor isnt supported, the descriptors will be created on device at the start of the kernels
    if supports_host_descriptor():
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]

        dummy_block = [
            1,
            1,
        ]  # After autotuning, block_shape will be overwritten with the optimal block size in host_descriptor_pre_hook()
        desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    else:
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

    return desc_q, desc_k, desc_v, desc_o, extra_kern_args


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, window, sm_scale):
        """This function implements a tiled version of the attention algorithm.

        Input matrices are in the shape [BATCH, N_HEAD, N_CTX, HEAD_DIM]

        """

        # Ensure inputs are contiguous, important when working with pointers later
        assert (
            q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        ), "Error. Input tensors to TritonAttention must be contiguous"

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)

        if window is None:
            window = 0
        assert not (causal and window > 0), "causal and window not supported in combination"

        # Pad tensors to avoid out-of-bounds reads when N_CTX is not a multiple of BLOCK_ITER
        # This ensures tensor descriptors don't read past the end of the flattened array
        n_ctx = q.shape[2]
        n_ctx_rounded = math.ceil(n_ctx / 16) * 16 #TODO(cathal) use block size from autotuning results instead of hardcoding 16 here
        
        # Pad q, k, v if necessary
        #TODO(cathal) need a better way to do this
        if n_ctx_rounded != n_ctx:
            pad_size = n_ctx_rounded - n_ctx
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_size), value=0.0)
            k = torch.nn.functional.pad(k, (0, 0, 0, pad_size), value=0.0)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_size), value=0.0)
            o = torch.nn.functional.pad(o, (0, 0, 0, pad_size), value=0.0)
        
        M = torch.empty((q.shape[0], q.shape[1], n_ctx_rounded), device=q.device, dtype=torch.float32)

        desc_q, desc_k, desc_v, desc_o, extra_kern_args = _system_specific_settings(q, k, v, o, True)

        # defines how blocks in the q,k and v input matrices are distributed across SMs on a GPU
        # (SMs are essentially processors on a GPU, with typically 1024 threads per SM)
        # Here a 2D grid is defined: NUM_CTX/BLOCK_ * (BATCH_SIZE * NUM_HEADS)
        # Meaning there is at least (BATCH_SIZE * NUM_HEADS) SMs
        # Depending on BLOCK_FIXED, the context window might also be split across SMs
        # BLOCK_FIXED is a hyperparameter which triton sets at runtime by running small performance tests
        def grid(META):
            return (triton.cdiv(n_ctx_rounded, META["BLOCK_FIXED"]), q.shape[0] * q.shape[1], 1)

        _attn_fwd[grid](
            sm_scale,
            M, 
            #L,
            q.shape[0],
            q.shape[1],  #
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            N_CTX=n_ctx,  # Use original context length for masking
            HEAD_DIM=HEAD_DIM_K,  #
            WINDOW=window,
            CAUSAL=causal,  #
            dtype=torch_dtype_to_triton(q.dtype),
            n_ctx_rounded=n_ctx_rounded,
            **extra_kern_args,
        )

        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.window = window
        ctx.n_ctx = n_ctx
        
        # Remove padding from tensors
        if n_ctx_rounded != n_ctx:
            o = o[:, :, :n_ctx, :]
            q = q[:, :, :n_ctx, :]
            k = k[:, :, :n_ctx, :]
            v = v[:, :, :n_ctx, :]

        ctx.save_for_backward(q, k, v, o, M)
        
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M= ctx.saved_tensors


        if do.shape == o.shape and do.stride() != o.stride():
            do = do.reshape(o.shape)

        do = do.contiguous()

        dq = torch.empty_like(q).contiguous()
        dk = torch.empty_like(k).contiguous()
        dv = torch.empty_like(v).contiguous()
        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape

        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        k = k * (ctx.sm_scale * RCP_LN2) # TODO move to kernel
        PRE_BLOCK = 16
        pre_grid = (triton.cdiv(N_CTX, PRE_BLOCK), BATCH * N_HEAD)
        delta = torch.empty_like(M)

        # Pad tensors to avoid out-of-bounds reads when N_CTX is not a multiple of BLOCK_ITER
        # This ensures tensor descriptors don't read past the end of the flattened array
        n_ctx = q.shape[2]
        n_ctx_rounded = math.ceil(n_ctx / 16) * 16 #TODO(cathal) use block size from autotuning results instead of hardcoding 16 here
        
        # Pad q, k, v if necessary
        #TODO(cathal) need a better way to do this
        if n_ctx_rounded != n_ctx:
            pad_size = n_ctx_rounded - n_ctx
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_size), value=0.0)
            k = torch.nn.functional.pad(k, (0, 0, 0, pad_size), value=0.0)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_size), value=0.0)
            o = torch.nn.functional.pad(o, (0, 0, 0, pad_size), value=0.0)

        desc_q, desc_k, desc_v, desc_o, extra_kern_args = _system_specific_settings(q, k, v, o, True)
        desc_dq, desc_dk, desc_dv, desc_do, extra_kern_args = _system_specific_settings(dq, dk, dv, do, True)

        # precompute 'delta' value needed for softmax computation
        _attn_bwd_preprocess[pre_grid](o, do, delta, N_CTX, PRE_BLOCK=PRE_BLOCK, HEAD_DIM=HEAD_DIM)

        # defines how blocks in the q,k and v input matrices are distributed across SMs on a GPU
        # (SMs are essentially processors on a GPU, with typically 1024 threads per SM)
        # There is at least one kernel per BATCH * NUM_HEAD.
        # Depending on BLOCK_FIXED, the context dimension can be further split across kernels
        def grid_dkdv(META):
            return (triton.cdiv(N_CTX, META["BLOCK_FIXED"]), 1, BATCH * N_HEAD)

        def grid_dq(META):
            return (triton.cdiv(N_CTX, META["BLOCK_FIXED"]), 1, BATCH * N_HEAD)

        # Compute dK and dV
        _attn_bwd_dkdv[grid_dkdv](
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            desc_dk,
            desc_dv,
            M,
            L,
            delta,  #
            ctx.sm_scale,
            Z=BATCH,
            H=N_HEAD,
            N_CTX=N_CTX,  #
            HEAD_DIM=HEAD_DIM,  #
            CAUSAL=ctx.causal,  #
            WINDOW=ctx.window,  #
            dtype=torch_dtype_to_triton(q.dtype),
            **extra_kern_args,
        )

        # Compute dQ
        _attn_bwd_dq[grid_dq](
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            desc_dq,
            M,
            delta,  #
            Z=BATCH,
            H=N_HEAD,
            N_CTX=N_CTX,  #
            HEAD_DIM=HEAD_DIM,  #
            CAUSAL=ctx.causal,  #
            WINDOW=ctx.window,  #
            dtype=torch_dtype_to_triton(q.dtype),
            **extra_kern_args,
        )

        return dq, dk, dv, None, None, None, None
