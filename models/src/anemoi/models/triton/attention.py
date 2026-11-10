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


import math
import os

import torch
from packaging import version

from anemoi.models.triton.utils import is_blackwell
from anemoi.models.triton.utils import is_hip
from anemoi.models.triton.utils import supports_host_descriptor
from anemoi.models.triton.utils import torch_dtype_to_triton

try:
    import triton
    import triton.language as tl  # noqa: E402
except ImportError as e:
    raise ImportError(
        "The 'triton_attention' backend was selected but Triton is not installed. To use this backend please install Triton. Otherwise, select a different attention_implementation in the models config."
    ) from e

TENSOR_DESCRIPTOR_SUPPORTED = version.parse(triton.__version__) >= version.parse("2.7")
if TENSOR_DESCRIPTOR_SUPPORTED:
    from triton.tools.tensor_descriptor import TensorDescriptor


def set_allocator():
    """Defines how memory is allocated by triton.
    Certain GPUs (e.g. Nvidia Hoppers and Blackwells) support TMA (Tensor Memory Accelerator)
    TMA is hardware support for async data movement between global and shared memory
    Basically the relevant memory addresses are calculated by dedicated hardware instead of registers
    This frees up threads and registers to do other computations
    TMA requires global memory allocations, so we set the alloc_fn here
    """
    # Currently there isn't a stable way to check if the allocator has not been set via Triton

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)


set_allocator()


@triton.jit
def _window_bounds(
    start,  # first position of the fixed block
    N_CTX: tl.constexpr,
    BLOCK_FIXED: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
    WINDOW: tl.constexpr,
    UNEVEN_CTX: tl.constexpr,
):
    """Range of iter blocks that the fixed block [start, start + BLOCK_FIXED) sees through the sliding window.

    Returns lo <= lo_in <= hi_in <= hi, all multiples of BLOCK_ITER. [lo, hi) holds every iter block with at
    least one pair inside the window. [lo_in, hi_in) holds the iter blocks whose every pair is inside the window
    and inside the context, so they need no masking.
    """
    # Attends within the following range (Assuming W=1)
    # X X - - -
    # X X X - -
    # - X X X -
    # - - X X X
    # - - - X X
    lo = tl.maximum(0, start - WINDOW)
    hi = tl.minimum(N_CTX, start + BLOCK_FIXED + WINDOW)
    # round lo down and hi up to the nearest multiple of BLOCK_ITER
    lo = (lo // BLOCK_ITER) * BLOCK_ITER
    hi = ((hi + BLOCK_ITER - 1) // BLOCK_ITER) * BLOCK_ITER

    # An iter block [c, c + BLOCK_ITER) is fully inside the window of every fixed position when
    # c >= start + BLOCK_FIXED - 1 - WINDOW and c + BLOCK_ITER - 1 <= start + WINDOW.
    lo_in = ((tl.maximum(start + BLOCK_FIXED - 1 - WINDOW, 0) + BLOCK_ITER - 1) // BLOCK_ITER) * BLOCK_ITER
    hi_in = ((start + WINDOW + 1) // BLOCK_ITER) * BLOCK_ITER
    if UNEVEN_CTX:
        # the last, partly filled block always takes the masked path
        hi_in = tl.minimum(hi_in, (N_CTX // BLOCK_ITER) * BLOCK_ITER)
    lo_in = tl.minimum(tl.maximum(lo_in, lo), hi)
    hi_in = tl.maximum(tl.minimum(hi_in, hi), lo_in)
    return (
        tl.multiple_of(lo, BLOCK_ITER),
        tl.multiple_of(lo_in, BLOCK_ITER),
        tl.multiple_of(hi_in, BLOCK_ITER),
        tl.multiple_of(hi, BLOCK_ITER),
    )


@triton.jit
def _attn_fwd_inner(
    acc,  # accumulator in smem for the output of this block, with shape [BLOCK_FIXED, HEAD_DIM]
    l_i,  # smem buffer for the denominator of the softmax, with shape [BLOCK_FIXED]
    m_i,  # smem buffer for the maxes used in the softmax, with shape [BLOCK_FIXED]
    q,  # the block of Q loaded into shared memory, with shape [BLOCK_FIXED, HEAD_DIM]
    desc_k,  # tensor descriptor for K
    desc_v,  # tensor descriptor for V
    iter_offset,  # the offset into K and V of the first key of this batch and head
    lo,  # first key position this call iterates over, a multiple of BLOCK_ITER
    hi,  # end of the key positions this call iterates over
    dtype: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,  # precision of the matrix products for float32 inputs
    qk_scale,  # scaling factor for the QK^T operation, contains the 1/log(2) factor for faster exponentant calculation
    BLOCK_FIXED: tl.constexpr,  # The size of BLOCK_FIXED, which determines how much of Q is loaded into shared memory and how much of the output is calculated by each block, determined by autotuning
    BLOCK_ITER: tl.constexpr,  # The size of BLOCK_ITER, which determines how much of K and V is iterated over in each block, determined by autotuning
    CAUSAL: tl.constexpr,  # whether to apply causal masking.
    offs_fixed: tl.constexpr,  # offset pointers for the block of Q and O which this kernel is responsible for, used for masking
    offs_iter: tl.constexpr,  # offset pointers for the block of K and V which this kernel is responsible for, used for masking
    N_CTX: tl.constexpr,  # the context length.
    WINDOW: tl.constexpr,  # The sliding window size for attention. If 0, no sliding window masking is applied.
    WARP_SPECIALIZE: tl.constexpr,  # Whether or not warp specialization should be used.
    UNEVEN_CTX: tl.constexpr,  # bool, true if N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER. padding, dynamic tensor descriptor block sizes and masked loads will be used to handle the uneven context
    MASKED: tl.constexpr,  # False when every pair in [lo, hi) is inside the window and the context, so no mask is applied
):
    """Tiled calculation of the attention algorithm. Inner loop.

    Outside this function, each program has loaded a BLOCK_FIXED sized section of the context Q
    This is stored in shared memory throughout.
    It then loops over K and V in sizes of BLOCK_ITER from lo to hi, updating the
    BLOCK_FIXED sized output in shared memory.
    Outside this function, output is stored back to global memory.

    Optionally, causal or sliding window masking can be performed.
    Uneven context handling: When N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER, padding and masked loads are used to handle the "tail" of the context which doesnt fit into a full block.

    """
    # Score given to hidden keys. It must lie below every score q . k can take, or a hidden key wins
    # the row maximum over keys whose scores are all lower (scores of -1e8 are reached once the
    # logits are large). It stays finite, so a row that has seen only hidden keys so far gives
    # (s - m) = 0 rather than NaN; those keys drop out once a visible key sets the maximum.
    MINUS_INF: tl.constexpr = float(-1.0e38)

    # Compute the starting offset of K and V
    iter_offset = iter_offset + lo

    # loop over k, v and update accumulator
    for curr_iter in tl.range(lo, hi, BLOCK_ITER, warp_specialize=WARP_SPECIALIZE):
        curr_iter = tl.multiple_of(curr_iter, BLOCK_ITER)  # Tells compiler curr_iter is a multiple of BLOCK_ITER
        tail_iter_block = curr_iter + BLOCK_ITER > N_CTX

        # -- compute qk  (load Kt, optionally mask it, load compute QKt, optionally mask it)----
        k = desc_k.load([iter_offset, 0]).T
        if MASKED and UNEVEN_CTX and tail_iter_block:
            k = tl.where(
                (curr_iter + offs_iter)[None, :] < N_CTX, k, 0.0
            )  # mask out-of-bounds k values to 0, so they dont contribute to output. This is needed when N_CTX is not divisible by BLOCK_FIXED

        qk = tl.dot(q, k, input_precision=INPUT_PRECISION)

        if MASKED:
            if UNEVEN_CTX and tail_iter_block:
                qk = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, qk, MINUS_INF)

            # apply Causal or Window masking if needed.
            if WINDOW >= 0:
                fixed_pos = offs_fixed[:, None]
                iter_pos = (curr_iter + offs_iter)[None, :]
                # Mask condition: keep if (q - window_size <= k <= q + window size)
                mask = (iter_pos <= fixed_pos + WINDOW) & (iter_pos >= fixed_pos - WINDOW)
                qk = tl.where(mask, qk, MINUS_INF)

            elif CAUSAL:
                fixed_pos = offs_fixed[:, None]
                iter_pos = (curr_iter + offs_iter)[None, :]
                mask = fixed_pos >= iter_pos
                qk = tl.where(mask, qk, MINUS_INF)

        # The row maximum is kept in unscaled logits and subtracted before scaling, so the largest
        # score of a row maps to exactly 0. Scaling first and subtracting a separately rounded,
        # scaled maximum lets the compiler fuse the two into one multiply-add, which leaves the
        # largest score slightly off 0; at large logits that error reaches the saved output and,
        # through delta, every gradient. See Chen et al., "Broken symmetry in BF16 attention",
        # https://arxiv.org/abs/2609.34272, section 3.3.
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2((qk - m_ij[:, None]) * qk_scale)

        # -- compute correction factor --
        alpha = tl.math.exp2((m_i - m_ij) * qk_scale)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v = desc_v.load([iter_offset, 0])
        if MASKED and UNEVEN_CTX and tail_iter_block:
            v = tl.where(
                (curr_iter + offs_iter)[:, None] < N_CTX, v, 0.0
            )  # mask out-of-bounds v values to 0, so they dont contribute to output. This is needed when N_CTX is not divisible by BLOCK_ITER
        p = p.to(dtype)

        # compute the final dot product and update accumulator
        acc = tl.dot(p, v, acc, input_precision=INPUT_PRECISION)

        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Move to next block
        iter_offset += BLOCK_ITER

    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    """Updates the tensor descriptor to use the block size determined by autotuning.

    Tensor descriptors are alternatives to raw pointers which encode information about the shape, stride and block size of the tensors.
    They allow the compiler to make use of hardware features like TMA (Tensor Memory Accelerator) on supported hardware (e.g. Nvidia Hoppers and Blackwells).

    Tensor descriptors are initially created with a dummy block size, as the optimal block size is not known until autotuning.
    This pre-hook updates the block size of the tensor descriptors to match the block size determined by autotuning for this kernel configuration.
    """

    if not isinstance(nargs["desc_q"], TensorDescriptor):
        # If the input is not a tensor descriptor, return
        # This can happen when the hardware does not support host descriptors, and raw pointers are passed to the kernel instead.
        # In this case, the kernel will create tensor descriptors later from the raw pointers.
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
    """
    if is_hip():
        NUM_STAGES_OPTIONS = [1]
    else:
        NUM_STAGES_OPTIONS = [1, 2, 3, 4]

    # TODO(cathal) reduce number of configurations or add heuristics to select configurations based on hardware properties, to reduce autotuning time.
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
        configs = [
            triton.Config(
                dict(BLOCK_FIXED=32, BLOCK_ITER=16, WARP_SPECIALIZE=False),
                num_stages=1,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            )
        ]
    return configs


@triton.jit
def _maybe_make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=_generate_configs(try_warp_spec=False),
    key=["N_CTX", "HEAD_DIM", "INPUT_PRECISION"],
    cache_results=False,
)
@triton.jit
def _attn_fwd(
    sm_scale,  # softmax scaling factor.
    M,  # pointer to row-wise softmax max values (m_i) in unscaled logits, shape [Z*H*n_ctx_rounded]
    INV_L,  # pointer to row-wise inverse softmax sum (1/l_i), shape [Z*H*n_ctx_rounded]
    Z,  # batch size
    H,  # num heads
    # tensor descriptors or raw pointers, depending on hardware support for host descriptors. If raw pointers are passed, they will be converted to tensor descriptors in the kernel with block sizes determined by autotuning
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    # raw pointers for output when using uneven ctx, to handle the dynamic block sizes and masked stores required for uneven ctx.
    o,
    N_CTX: tl.constexpr,  # context length
    HEAD_DIM: tl.constexpr,  # head dimension
    WINDOW: tl.constexpr,  # sliding window size. If negative, no sliding window masking is applied. Only values within the window around the BLOCK_FIXED section of Q loaded by the kernel will be attended to.
    BLOCK_FIXED: tl.constexpr,  # size of the block of Q loaded into shared memory
    BLOCK_ITER: tl.constexpr,  # size of the block of K and V iterated over
    CAUSAL: tl.constexpr,  # whether to apply causal masking
    WARP_SPECIALIZE: tl.constexpr,  # whether to use warp specialization
    dtype: tl.constexpr,  # output dtype
    INPUT_PRECISION: tl.constexpr,  # precision of the matrix products for float32 inputs, see _dot_input_precision
    n_ctx_rounded: tl.constexpr,  # the next multiple of BLOCK_FIXED and BLOCK_ITER above N_CTX, used for calculating offsets and strides when UNEVEN_CTX is false
    UNEVEN_CTX: tl.constexpr,  # bool, true if N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER. padding, dynamic tensor descriptor block sizes and masked loads will be used to handle the uneven context
):
    """Tiled forward pass of attention.

    Each program loads a BLOCK_FIXED sized section of the context Q into shared memory,
    and iterates over K and V in blocks of BLOCK_ITER (inside the kernel _attn_fwd_inner)
    until the entire BLOCK_FIXED sized output O is accumulated.

    Uneven context handling: When N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER, padding and masked loads are used to handle the "tail" of the context which doesnt fit into a full block.
    """

    # ***** 1) determine which section of the output this program is responsible for *****

    # This kernel is executed by multiple SMs in parallel, each responsible for calculating a different section of the output O
    # The following lines allow the kernel to determine which section of the output it is responsible for, so it can load the correct section of Q and iterate over the correct sections of K and V
    start_fixed = tl.program_id(
        0
    )  # which BLOCK_FIXED sized section of the output this program is responsible, determining the subset of Q and O which is loaded and stored by this program
    off_hz = tl.program_id(1)  # Which head and batch this program is partly responsible for.
    off_z = off_hz // H
    off_h = off_hz % H

    # ****** 2) create tensor descriptors for the input and output tensors, with block sizes determined by autotuning *****

    # number of elements in all non-head-dim dimensions (use actual N_CTX for q,k,v,o layout)
    y_dim = Z * H * N_CTX

    # tensor descriptors might have already been created on host (in system_specific_settings())
    # If host descriptors aren't supported on the hardware, then tensor descripotrs are created now
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

    # For output: only create tensor descriptor when NOT in tail case
    # (tail case uses raw pointer arithmetic to handle dynamic block size)
    tail_case = ((start_fixed + 1) * BLOCK_FIXED) > N_CTX
    desc_o = _maybe_make_tensor_descriptor(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )

    # ****** 3) calculate offsets into the input and output tensors which this kernel is responsible for *****

    # Offsets into the start of the K and V tensors which this kernel will iterate over )
    iter_offset = off_z * H * N_CTX + off_h * N_CTX
    # Offsets into the middle of the Q tensor which this kernel will keep in memory, and the O tensor which it will write to.
    fixed_offset = iter_offset + start_fixed * BLOCK_FIXED

    # initialize offset pointers (used for normal tensors)
    offs_fixed = start_fixed * BLOCK_FIXED + tl.arange(0, BLOCK_FIXED)
    offs_iter = tl.arange(0, BLOCK_ITER)

    # ****** 4) allocate shared memory buffers for output and load fixed subset of Q into shared memory *****

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_FIXED], dtype=tl.float32) - float(
        "inf"
    )  # list of maximum values, will be updated after each BLOCK_ITER. used to compute online softmax
    l_i = tl.zeros([BLOCK_FIXED], dtype=tl.float32)
    acc = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)  # output accumulator

    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2) #hack to make calculating exponent faster, by merging 1/ln(2) now the cheaper exp2() fn can be called later instead of exp()

    # load q: it will stay in SRAM throughout
    q = desc_q.load([fixed_offset, 0])

    if UNEVEN_CTX and tail_case:
        # mask out-of-bounds q values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED
        q = tl.where(offs_fixed[:, None] < N_CTX, q, 0.0)

    # ****** 5) main loop, iterating over blocks of K and V, and updating the output accumulator, m_i and l_i for each block *****
    if WINDOW >= 0:
        # Key blocks inside the window of every query of this block need no mask, so only the blocks
        # at the two edges of the window go through the masked loop.
        lo, lo_in, hi_in, hi = _window_bounds(
            start_fixed * BLOCK_FIXED, N_CTX, BLOCK_FIXED, BLOCK_ITER, WINDOW, UNEVEN_CTX
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            iter_offset,
            lo,
            lo_in,
            dtype,
            INPUT_PRECISION,
            qk_scale,
            BLOCK_FIXED,
            BLOCK_ITER,
            CAUSAL,
            offs_fixed,
            offs_iter,
            N_CTX,
            WINDOW,
            WARP_SPECIALIZE,
            UNEVEN_CTX,
            True,
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            iter_offset,
            lo_in,
            hi_in,
            dtype,
            INPUT_PRECISION,
            qk_scale,
            BLOCK_FIXED,
            BLOCK_ITER,
            CAUSAL,
            offs_fixed,
            offs_iter,
            N_CTX,
            WINDOW,
            WARP_SPECIALIZE,
            UNEVEN_CTX,
            False,
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            iter_offset,
            hi_in,
            hi,
            dtype,
            INPUT_PRECISION,
            qk_scale,
            BLOCK_FIXED,
            BLOCK_ITER,
            CAUSAL,
            offs_fixed,
            offs_iter,
            N_CTX,
            WINDOW,
            WARP_SPECIALIZE,
            UNEVEN_CTX,
            True,
        )
    else:
        if CAUSAL:
            # Attends within the following range
            # X - - - -
            # X X - - -
            # X X X - -
            # X X X X -
            # X X X X X
            lo, hi = 0, tl.minimum((start_fixed + 1) * BLOCK_FIXED, N_CTX)
            hi = tl.multiple_of(hi, BLOCK_FIXED)
        else:
            # Attends within the following range
            # X X X X X
            # X X X X X
            # X X X X X
            # X X X X X
            # X X X X X
            lo, hi = 0, N_CTX
        if not UNEVEN_CTX:
            hi = tl.multiple_of(hi, BLOCK_ITER)
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            iter_offset,
            lo,
            hi,
            dtype,
            INPUT_PRECISION,
            qk_scale,
            BLOCK_FIXED,
            BLOCK_ITER,
            CAUSAL,
            offs_fixed,
            offs_iter,
            N_CTX,
            WINDOW,
            WARP_SPECIALIZE,
            UNEVEN_CTX,
            True,
        )
    # epilogue
    # Save m_max (in unscaled logits) and inv_l separately so the backward can compute
    # p = exp2((qk - m_max) * scale) * inv_l without the fp precision loss
    # that can occur when forming the combined M = m + log2(l).
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    m_ptrs = M + off_hz * n_ctx_rounded + offs_fixed  # Use n_ctx_rounded since M is allocated with that dimension
    inv_l_ptrs = INV_L + off_hz * n_ctx_rounded + offs_fixed

    # ***** 6) store output *****

    # store output values, handling uneven ctx by masking out-of-bounds values and adjusting block size of output when necessary
    if UNEVEN_CTX and tail_case:
        # mask the store to m so that we dont write out-of-bounds values when N_CTX is not divisible by BLOCK_FIXED.
        tl.store(m_ptrs, m_i, mask=offs_fixed < N_CTX)
        tl.store(inv_l_ptrs, inv_l_i, mask=offs_fixed < N_CTX)
        # need to write a smaller block size when using uneven ctx to avoid writing into the next SMs region
        # o is a tensor descriptor which doesnt support different block sizes, so access o as a regular pointer with 2D indexing
        offs_d = tl.arange(0, HEAD_DIM)
        o_ptrs = o + off_hz.to(tl.int64) * N_CTX * HEAD_DIM + offs_fixed[:, None] * HEAD_DIM + offs_d[None, :]
        tl.store(o_ptrs, acc.to(dtype), mask=offs_fixed[:, None] < N_CTX)
    else:
        tl.store(m_ptrs, m_i)
        tl.store(inv_l_ptrs, inv_l_i)
        desc_o.store([fixed_offset, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_dkdv_inner(
    dk,  # gradient accumulator for dK, [BLOCK_FIXED, HEAD_DIM]
    dv,  # gradient accumulator for dV, [BLOCK_FIXED, HEAD_DIM]
    k,  # the fixed block of K
    v,  # the fixed block of V
    desc_q,
    desc_do,
    M,  # row-wise softmax max values of this batch and head
    INV_L,  # row-wise inverse softmax sums of this batch and head
    D,  # delta values of this batch and head
    LAM,  # per-query correction of ds of this batch and head, written by _attn_bwd_dq
    iter_offset,  # the offset into Q and dO of the first query of this batch and head
    lo,  # first query position this call iterates over, a multiple of BLOCK_ITER
    hi,  # end of the query positions this call iterates over
    qk_scale,
    offs_fixed,  # positions of the fixed block of K and V
    N_CTX: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    dtype: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,  # precision of the matrix products for float32 inputs
    UNEVEN_CTX: tl.constexpr,
    MASKED: tl.constexpr,  # False when every pair in [lo, hi) is inside the window and the context, so no mask is applied
):
    """Inner loop of _attn_bwd_dkdv over the blocks of Q between lo and hi."""
    MINUS_INF: tl.constexpr = float(-1.0e38)  # score of hidden keys, see _attn_fwd_inner
    offs_iter = tl.arange(0, BLOCK_ITER)

    # skip up to 'lo'
    iter_offset += lo

    for curr_iter in tl.range(lo, hi, BLOCK_ITER, warp_specialize=WARP_SPECIALIZE):
        curr_iter = tl.multiple_of(curr_iter, BLOCK_ITER)  # Tells compiler curr_iter is a multiple of BLOCK_ITER
        curr_offs = curr_iter + offs_iter
        qT = desc_q.load([iter_offset, 0]).T
        tail_iter_block = MASKED and UNEVEN_CTX and (curr_iter + BLOCK_ITER > N_CTX)
        if tail_iter_block:
            # mask out-of-bounds q values to 0, so they dont contribute to output. This is needed when N_CTX is not divisible by BLOCK_FIXED
            qT = tl.where(curr_offs[None, :] < N_CTX, qT, 0.0)

        # Load m and inv_l before computing qk to reduce pipeline stall.
        if tail_iter_block:
            m = tl.load(M + curr_offs, mask=curr_offs < N_CTX, other=0.0)
            inv_l = tl.load(INV_L + curr_offs, mask=curr_offs < N_CTX, other=0.0)
        else:
            m = tl.load(M + curr_offs)
            inv_l = tl.load(INV_L + curr_offs)
        qkT = tl.dot(k, qT, input_precision=INPUT_PRECISION)

        # Apply masking.
        if MASKED:
            if tail_iter_block:
                qkT = tl.where((curr_offs[None, :] < N_CTX), qkT, MINUS_INF)
            if CAUSAL:
                mask = curr_offs[None, :] >= offs_fixed[:, None]
                qkT = tl.where(mask, qkT, MINUS_INF)
            if WINDOW >= 0:
                iter_pos = curr_offs[None, :]
                # Mask condition: keep if (q - window_size <= k <= q + window size)
                mask = (offs_fixed[:, None] - WINDOW <= iter_pos) & (offs_fixed[:, None] + WINDOW >= iter_pos)
                qkT = tl.where(mask, qkT, MINUS_INF)

        # Apply exponent after masking, then multiply by inv_l to get the
        # normalised softmax probability.  Keeping m_max and inv_l separate
        # (rather than a combined M = m + log2(l))
        # avoids fp precision loss when m and log2(l) differ greatly.
        pT = tl.math.exp2((qkT - m[None, :]) * qk_scale) * inv_l[None, :]

        do = desc_do.load([iter_offset, 0])
        if tail_iter_block:
            # mask out-of-bounds do values to 0, so they dont contribute to output. This is needed when N_CTX is not divisible by BLOCK_FIXED
            do = tl.where(curr_offs[:, None] < N_CTX, do, 0.0)
        # Compute dV.
        dv += tl.dot(pT.to(dtype), do, input_precision=INPUT_PRECISION)
        if tail_iter_block:
            Di = tl.load(D + curr_offs, mask=curr_offs < N_CTX, other=0.0)
            lam = tl.load(LAM + curr_offs, mask=curr_offs < N_CTX, other=0.0)
        else:
            Di = tl.load(D + curr_offs)
            lam = tl.load(LAM + curr_offs)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do), input_precision=INPUT_PRECISION).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk += tl.dot(dsT.to(dtype), tl.trans(qT), input_precision=INPUT_PRECISION)
        # dk from ds - lam * p, with lam from the dQ kernel, so the rounded ds rows sum to zero here too.
        # The correction stays a product of its own: lam * p is far smaller than ds, and rounding
        # ds - lam * p to 16 bits would lose most of it.
        dk += tl.dot((pT * -lam[None, :]).to(dtype), tl.trans(qT), input_precision=INPUT_PRECISION)

        # Move to next iter block
        iter_offset += BLOCK_ITER

    return dk, dv


@triton.autotune(
    # Autotuning is crucial to get good performance at larger head dims
    # For an o96 2048c configuration, got a 3x speedup from autotuning
    configs=_generate_configs(try_warp_spec=False),
    key=["N_CTX", "HEAD_DIM", "INPUT_PRECISION"],
    cache_results=False,
)
@triton.jit
def _attn_bwd_dkdv(
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_dk,
    desc_dv,
    # raw pointers for dk and dv when using uneven ctx, to handle the dynamic block sizes and masked stores required for uneven ctx
    dk_ptr,
    dv_ptr,
    M,  # pointer to row-wise softmax max values (m_i) saved by forward
    INV_L,  # pointer to row-wise inverse softmax sum (1/l_i) saved by forward
    D,  # pointer to delta values written by _attn_bwd_dq
    LAM,  # pointer to the per-query correction of ds written by _attn_bwd_dq, see there
    sm_scale: tl.constexpr,  # softmax scaling factor
    # shared by Q/K/V/DO.
    Z: tl.constexpr,  # batch size
    H: tl.constexpr,  # num heads
    N_CTX: tl.constexpr,  # context length
    BLOCK_ITER: tl.constexpr,  # size of the block of Q iterated over
    BLOCK_FIXED: tl.constexpr,  # size of the block of K and V loaded into shared memory
    HEAD_DIM: tl.constexpr,  # head dimension
    CAUSAL: tl.constexpr,  # whether to apply causal masking
    WINDOW: tl.constexpr,  # sliding window size. If negative, no sliding window masking is applied
    WARP_SPECIALIZE: tl.constexpr,  # whether to use warp specialization
    dtype: tl.constexpr,  # output dtype
    INPUT_PRECISION: tl.constexpr,  # precision of the matrix products for float32 inputs, see _dot_input_precision
    UNEVEN_CTX: tl.constexpr,  # bool, true if N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER
    n_ctx_rounded: tl.constexpr,  # the next multiple of BLOCK_FIXED and BLOCK_ITER above N_CTX, used for M/L/D indexing
):
    """Tiled backward pass for dK and dV.

    Each kernel loads a single BLOCK_FIXED-sized block of K and V into shared memory,
    and iterates over columns of Q in blocks of BLOCK_ITER, accumulating gradients dK and dV.

    Uneven context handling: When N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER, padding and masked loads are used to handle the "tail" of the context.
    """
    RCP_LN2: tl.constexpr = (
        1.44269504  # 1/log(2), used to merge with sm_scale to make calculating exponent faster by using exp2() later
    )

    # ***** 1) determine which section of the gradients this program is responsible for *****

    # This kernel is executed by multiple SMs in parallel, each responsible for calculating gradients for a different BLOCK_FIXED-sized section of K and V
    pid = tl.program_id(0)
    start_fixed = (
        pid * BLOCK_FIXED
    )  # which BLOCK_FIXED sized section of K and V this program loads and computes gradients for

    # Index into head and batch
    off_hz = tl.program_id(2)  # Which head and batch this program is responsible for
    off_chz = (off_hz * n_ctx_rounded).to(tl.int64)  # Use n_ctx_rounded since M/L/D are allocated with that dimension
    off_z = off_hz // H
    off_h = off_hz % H

    # ***** 2) calculate offsets into the input and output tensors which this kernel is responsible for *****

    iter_offset = (
        off_z * H * N_CTX + off_h * N_CTX
    )  # Offsets into the start of the Q tensor which this kernel will iterate over
    fixed_offset = (
        iter_offset + start_fixed
    )  # Offsets into the middle of the K/V tensors which this kernel will keep in memory, and the dK/dV tensors which it will write to

    # ***** 3) create tensor descriptors for the input and output tensors, with block sizes determined by autotuning *****

    # number of elements in all non-head-dim dimensions (use actual N_CTX for q,k,v,do layout)
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

    desc_dk = _maybe_make_tensor_descriptor(
        desc_dk,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )
    desc_dv = _maybe_make_tensor_descriptor(
        desc_dv,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )

    # ***** 4) allocate shared memory buffers for gradients and load fixed subset of K and V into shared memory *****

    # offset pointers for batch/head into M, D and LAM arrays
    M += off_chz
    INV_L += off_chz
    D += off_chz
    LAM += off_chz

    # start_fixed is an element offset here (pid * BLOCK_FIXED), not a block index as in _attn_fwd
    tail_fixed_block = (start_fixed + BLOCK_FIXED) > N_CTX

    offs_fixed = start_fixed + tl.arange(0, BLOCK_FIXED)

    dv = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)  # gradient accumulator for dV
    dk = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)  # gradient accumulator for dK

    # load K and V: they stay in SRAM throughout the inner loop.
    k = desc_k.load([fixed_offset, 0])
    # Match forward: scale the FP32 dot product, not the low-precision keys.
    # Rounding K before the dot changes the logits relative to the saved max;
    # exp2 amplifies that mismatch dramatically for sharp attention.
    qk_scale = tl.full((), sm_scale, tl.float32) * RCP_LN2
    v = desc_v.load([fixed_offset, 0])
    if UNEVEN_CTX and tail_fixed_block:
        # mask out-of-bounds k and v values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED/BLOCK_ITER
        k = tl.where(offs_fixed[:, None] < N_CTX, k, 0.0)
        v = tl.where(offs_fixed[:, None] < N_CTX, v, 0.0)

    # This is a single stage kernel, which loops over columns of Q
    # When masking (causal or sliding-window) is used, it must determine where the
    # current block is in the global matrix to determine how it is masked
    # There are 3 options:
    #   fully masked out => skip block
    #   partly masked => apply MASK to block
    #   fully inside the window => no MASK needed

    # ***** 5) main loop, iterating over blocks of Q, and updating the gradient accumulators dK and dV for each block *****

    if WINDOW >= 0:
        # Rather then iterating across the whole context, we iterate over a window around the
        # position of the K and V blocks; query blocks inside the window of every key need no mask.
        lo, lo_in, hi_in, hi = _window_bounds(start_fixed, N_CTX, BLOCK_FIXED, BLOCK_ITER, WINDOW, UNEVEN_CTX)
        dk, dv = _attn_bwd_dkdv_inner(
            dk,
            dv,
            k,
            v,
            desc_q,
            desc_do,
            M,
            INV_L,
            D,
            LAM,
            iter_offset,
            lo,
            lo_in,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            True,
        )
        dk, dv = _attn_bwd_dkdv_inner(
            dk,
            dv,
            k,
            v,
            desc_q,
            desc_do,
            M,
            INV_L,
            D,
            LAM,
            iter_offset,
            lo_in,
            hi_in,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            False,
        )
        dk, dv = _attn_bwd_dkdv_inner(
            dk,
            dv,
            k,
            v,
            desc_q,
            desc_do,
            M,
            INV_L,
            D,
            LAM,
            iter_offset,
            hi_in,
            hi,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            True,
        )
    else:
        if CAUSAL:
            # start_fixed, rounded down to lowest multiple if not even
            lo = (start_fixed // BLOCK_ITER) * BLOCK_ITER
            # this function doesnt convert to a multiple - it informs the compiler that the first number IS a multiple of the second
            lo = tl.multiple_of(lo, BLOCK_ITER)
        else:
            lo = 0
        dk, dv = _attn_bwd_dkdv_inner(
            dk,
            dv,
            k,
            v,
            desc_q,
            desc_do,
            M,
            INV_L,
            D,
            LAM,
            iter_offset,
            lo,
            N_CTX,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            True,
        )

    # ***** 6) store gradients dK and dV *****

    # Store computed dK and dV
    # when N_CTX is not divisible by BLOCK_FIXED, we may have computed values for out-of-bounds positions,
    # In this case, we set the block size of desc_d[vk] to be smaller in the tail case, such that the store only writes the in-bounds values.

    if UNEVEN_CTX and tail_fixed_block:
        # need to write a smaller block size when using uneven ctx to avoid writing into the next SMs region
        # to do this, access dk and dv as regular pointers with 2D indexing
        out_ptrs = (
            off_hz.to(tl.int64) * N_CTX * HEAD_DIM + offs_fixed[:, None] * HEAD_DIM + (tl.arange(0, HEAD_DIM))[None, :]
        )
        dv_ptrs = dv_ptr + out_ptrs
        dk_ptrs = dk_ptr + out_ptrs

        tl.store(dv_ptrs, dv.to(dtype), mask=offs_fixed[:, None] < N_CTX)
        dk *= sm_scale
        tl.store(dk_ptrs, dk.to(dtype), mask=offs_fixed[:, None] < N_CTX)
    else:
        desc_dv.store([fixed_offset, 0], dv.to(dtype))
        dk *= sm_scale
        desc_dk.store([fixed_offset, 0], dk.to(dtype))


@triton.jit
def _attn_bwd_dq_inner(
    dq,  # gradient accumulator for dQ, [BLOCK_FIXED, HEAD_DIM]
    p_k,  # accumulator for the rounded probabilities times K, [BLOCK_FIXED, HEAD_DIM]
    ds_sum,  # sums of the rounded ds of each query, [BLOCK_FIXED]
    p_sum,  # sums of the rounded probabilities of each query, [BLOCK_FIXED]
    q,  # the fixed block of Q
    do,  # the fixed block of dO
    m,  # row-wise softmax max values of the fixed block, [BLOCK_FIXED, 1]
    inv_l,  # row-wise inverse softmax sums of the fixed block
    Di,  # delta values of the fixed block
    desc_k,
    desc_v,
    iter_offset,  # the offset into K and V of the first key of this batch and head
    lo,  # first key position this call iterates over, a multiple of BLOCK_ITER
    hi,  # end of the key positions this call iterates over
    qk_scale,
    offs_fixed,  # positions of the fixed block of Q
    N_CTX: tl.constexpr,
    BLOCK_ITER: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    dtype: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,  # precision of the matrix products for float32 inputs
    UNEVEN_CTX: tl.constexpr,
    MASKED: tl.constexpr,  # False when every pair in [lo, hi) is inside the window and the context, so no mask is applied
):
    """Inner loop of _attn_bwd_dq over the blocks of K and V between lo and hi."""
    MINUS_INF: tl.constexpr = float(-1.0e38)  # score of hidden keys, see _attn_fwd_inner
    offs_iter = tl.arange(0, BLOCK_ITER)

    # skip up to 'lo'
    iter_offset += lo

    for curr_iter in tl.range(lo, hi, BLOCK_ITER, warp_specialize=WARP_SPECIALIZE):
        curr_iter = tl.multiple_of(curr_iter, BLOCK_ITER)  # Tells compiler curr_iter is a multiple of BLOCK_ITER
        tail_iter_block = MASKED and UNEVEN_CTX and ((curr_iter + BLOCK_ITER) > N_CTX)

        kT = desc_k.load([iter_offset, 0]).T
        vT = desc_v.load([iter_offset, 0]).T
        if tail_iter_block:
            # mask out-of-bounds k and q values to 0, so they dont contribute to output when N_CTX is not divisible by BLOCK_FIXED
            kT = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, kT, 0.0)
            vT = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, vT, 0.0)

        qk = tl.dot(q, kT, input_precision=INPUT_PRECISION)

        # apply masking
        if MASKED:
            if tail_iter_block:
                qk = tl.where((curr_iter + offs_iter)[None, :] < N_CTX, qk, MINUS_INF)
            if CAUSAL:
                iter_pos = (curr_iter + offs_iter)[None, :]
                fixed_pos = offs_fixed[:, None]
                mask = fixed_pos >= iter_pos
                qk = tl.where(mask, qk, MINUS_INF)
            if WINDOW >= 0:
                iter_pos = (curr_iter + offs_iter)[None, :]
                mask = (iter_pos <= offs_fixed[:, None] + WINDOW) & (iter_pos >= offs_fixed[:, None] - WINDOW)
                qk = tl.where(mask, qk, MINUS_INF)

        # Apply exponent after masking, then multiply by inv_l for the
        # normalised softmax probability (see _attn_bwd_dkdv for rationale).
        # Reconstruct probabilities with the same FP32 scaling as forward.
        p = tl.math.exp2((qk - m) * qk_scale) * inv_l[:, None]
        # Compute dP and dS.
        # NOTE: dp - Di still suffers from cancellation when the softmax is
        # very sharp (v[j*] ≈ out[i]) because both are O(1) scalars and their
        # difference is computed after the dot-product accumulation.  The
        # element-wise fix used in the graph-transformer kernel (gt.py) cannot
        # be applied here: v[j] lives in the iter tile while out[i] lives in
        # the fixed tile, so dot(do[i], v[j]-out[i]) cannot be expressed as a
        # single tl.dot without restructuring to a non-tiled per-query loop.
        dp = tl.dot(do, vT, input_precision=INPUT_PRECISION).to(tl.float32)
        ds = p * (dp - Di[:, None])
        # Compute dQ.
        # K is unscaled; apply the softmax chain-rule scale in the epilogue.
        ds = ds.to(dtype)
        p = p.to(dtype)
        ds_sum += tl.sum(ds.to(tl.float32), 1)
        p_sum += tl.sum(p.to(tl.float32), 1)
        dq += tl.dot(ds, tl.trans(kT), input_precision=INPUT_PRECISION)
        p_k += tl.dot(p, tl.trans(kT), input_precision=INPUT_PRECISION)

        # move to the next iter_block
        iter_offset += BLOCK_ITER

    return dq, p_k, ds_sum, p_sum


@triton.autotune(
    # Autotuning is crucial to get good performance at larger head dims
    # For an o96 2048c configuration, got a 3x speedup from autotuning
    configs=_generate_configs(try_warp_spec=False),
    key=["N_CTX", "HEAD_DIM", "INPUT_PRECISION"],
    cache_results=False,
)
@triton.jit
def _attn_bwd_dq(
    desc_q,
    desc_k,
    desc_v,
    desc_do,
    desc_dq,
    dq_ptr,  # raw pointer for dq when using uneven ctx, to handle the dynamic block sizes and masked stores required for uneven ctx
    Out,  # raw pointer to the output saved by forward
    M,  # pointer to row-wise softmax max values (m_i) saved by forward
    INV_L,  # pointer to row-wise inverse softmax sum (1/l_i) saved by forward
    D,  # written here: delta values, rowsum(O * dO) per query, read by _attn_bwd_dkdv
    LAM,  # written here: sum of the rounded ds over sum of the rounded probabilities per query, see below
    sm_scale: tl.constexpr,  # softmax scaling factor
    # shared by Q/K/V/DO.
    Z: tl.constexpr,  # batch size
    H: tl.constexpr,  # num heads
    N_CTX: tl.constexpr,  # context length
    BLOCK_ITER: tl.constexpr,  # size of the block of K and V iterated over
    BLOCK_FIXED: tl.constexpr,  # size of the block of Q loaded into shared memory
    HEAD_DIM: tl.constexpr,  # head dimension
    CAUSAL: tl.constexpr,  # whether to apply causal masking
    WINDOW: tl.constexpr,  # sliding window size. If negative, no sliding window masking is applied
    WARP_SPECIALIZE: tl.constexpr,  # whether to use warp specialization
    dtype: tl.constexpr,  # output dtype
    INPUT_PRECISION: tl.constexpr,  # precision of the matrix products for float32 inputs, see _dot_input_precision
    n_ctx_rounded: tl.constexpr,  # the next multiple of BLOCK_FIXED and BLOCK_ITER above N_CTX, used for M/D indexing
    UNEVEN_CTX: tl.constexpr,  # bool, true if N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER
):
    """Tiled backward pass for dQ.

    Each kernel loads a single BLOCK_FIXED-sized block of Q and dO into shared memory,
    and iterates over columns of K and V in blocks of BLOCK_ITER, accumulating gradients dQ.

    Uneven context handling: When N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER, padding and masked loads are used to handle the "tail" of the context.
    """

    RCP_LN2: tl.constexpr = 1.44269504  # = 1/ln(2), matching the forward and dK/dV kernels
    qk_scale = tl.full((), sm_scale, tl.float32) * RCP_LN2

    # ***** 1) determine which section of the gradients this program is responsible for *****

    # This kernel is executed by multiple SMs in parallel, each responsible for calculating gradients for a different BLOCK_FIXED-sized section of Q
    pid = tl.program_id(0)
    start_fixed = (
        pid * BLOCK_FIXED
    )  # which BLOCK_FIXED sized section of Q this program loads and computes gradients for

    # Index into head and batch
    off_hz = tl.program_id(2)  # Which head and batch this program is responsible for
    off_chz = off_hz * n_ctx_rounded
    off_z = off_hz // H
    off_h = off_hz % H

    # ***** 2) calculate offsets into the input and output tensors which this kernel is responsible for *****

    iter_offset = (
        off_z * (N_CTX * H) + off_h * N_CTX
    )  # Offsets into the start of the K and V tensors which this kernel will iterate over
    fixed_offset = (
        iter_offset + start_fixed
    )  # Offsets into the middle of the Q tensor which this kernel will keep in memory, and the dQ tensor which it will write to

    # ***** 3) create tensor descriptors for the input and output tensors, with block sizes determined by autotuning *****

    # How many elements in the first 3 dimensions (use actual N_CTX for q,k,v,do layout)
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
    desc_dq = _maybe_make_tensor_descriptor(
        desc_dq,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_FIXED, HEAD_DIM],
    )

    # ***** 4) allocate shared memory buffers for gradients and load fixed subset of Q and dO into shared memory *****

    # offset pointers for batch/head into M, D and LAM arrays
    M += off_chz
    INV_L += off_chz
    D += off_chz
    LAM += off_chz

    # generate offset array for fixed block
    offs_fixed = start_fixed + tl.arange(0, BLOCK_FIXED)
    tail_fixed_block = (start_fixed + BLOCK_FIXED) > N_CTX

    q = desc_q.load([fixed_offset, 0])
    dq = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)  # gradient accumulator for dQ
    # The exact ds of a query sums to zero over its keys, which keeps dq blind to where the keys sit
    # as a group. Rounding ds to the input precision before the product with k leaves a small sum,
    # and dq picks up that sum times the mean key, which outweighs the true gradient when the keys
    # are far from the origin compared with their spread. The sums of the rounded ds and of the
    # rounded probabilities are kept, and lam = their ratio, so that ds - lam * p sums to zero; dq
    # and dk use that. It also removes the sum that delta, built from the rounded saved output,
    # leaves in ds. See Chen et al., "Broken symmetry in BF16 attention" (GProj),
    # https://arxiv.org/abs/2609.34272.
    p_k = tl.zeros([BLOCK_FIXED, HEAD_DIM], dtype=tl.float32)
    ds_sum = tl.zeros([BLOCK_FIXED], dtype=tl.float32)
    p_sum = tl.zeros([BLOCK_FIXED], dtype=tl.float32)
    do = desc_do.load([fixed_offset, 0])
    if UNEVEN_CTX and tail_fixed_block:
        # mask out-of-bounds q values to 0, so they dont contribute to output. This can happen when N_CTX is not divisible by BLOCK_FIXED
        q = tl.where(offs_fixed[:, None] < N_CTX, q, 0.0)
        do = tl.where(
            offs_fixed[:, None] < N_CTX, do, 0.0
        )  # mask out-of-bounds do values to 0, so they dont contribute

    if UNEVEN_CTX and tail_fixed_block:
        m = tl.load(M + offs_fixed, mask=offs_fixed < N_CTX, other=0.0)  # Add masking to prevent loading garbage values
        inv_l = tl.load(INV_L + offs_fixed, mask=offs_fixed < N_CTX, other=0.0)
    else:
        m = tl.load(M + offs_fixed)
        inv_l = tl.load(INV_L + offs_fixed)
    m = m[:, None]

    # delta = rowsum(O * dO), the per-query scalar of the softmax backward. It is worked out here from
    # the saved output of this block and stored for the dK/dV kernel.
    o = tl.load(
        Out + off_hz.to(tl.int64) * N_CTX * HEAD_DIM + offs_fixed[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :],
        mask=offs_fixed[:, None] < N_CTX,
        other=0.0,
    )
    Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(D + offs_fixed, Di, mask=offs_fixed < N_CTX)

    # This is a single stage kernel, which loops over columns of K and V
    # When masking (causal or sliding-window) is used, it must determine where the
    # current block is in the global matrix to determine how it is masked
    # There are 3 options:
    #   fully masked out => skip block
    #   partly masked => apply MASK to block
    #   fully inside the window => no MASK needed

    # ***** 5) main loop, iterating over blocks of K and V, and updating the gradient accumulator dQ for each block *****

    if WINDOW >= 0:
        lo, lo_in, hi_in, hi = _window_bounds(start_fixed, N_CTX, BLOCK_FIXED, BLOCK_ITER, WINDOW, UNEVEN_CTX)
        dq, p_k, ds_sum, p_sum = _attn_bwd_dq_inner(
            dq,
            p_k,
            ds_sum,
            p_sum,
            q,
            do,
            m,
            inv_l,
            Di,
            desc_k,
            desc_v,
            iter_offset,
            lo,
            lo_in,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            True,
        )
        dq, p_k, ds_sum, p_sum = _attn_bwd_dq_inner(
            dq,
            p_k,
            ds_sum,
            p_sum,
            q,
            do,
            m,
            inv_l,
            Di,
            desc_k,
            desc_v,
            iter_offset,
            lo_in,
            hi_in,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            False,
        )
        dq, p_k, ds_sum, p_sum = _attn_bwd_dq_inner(
            dq,
            p_k,
            ds_sum,
            p_sum,
            q,
            do,
            m,
            inv_l,
            Di,
            desc_k,
            desc_v,
            iter_offset,
            hi_in,
            hi,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            True,
        )
    else:
        if CAUSAL:
            # hi is the block end, rounded up to the nearest multiple of BLOCK_ITER
            hi = ((start_fixed + BLOCK_FIXED + BLOCK_ITER - 1) // BLOCK_ITER) * BLOCK_ITER
            # this function doesnt convert to a multiple - it informs the compiler that the first number IS a multiple of the second
            hi = tl.multiple_of(hi, BLOCK_ITER)
        else:
            hi = N_CTX
        dq, p_k, ds_sum, p_sum = _attn_bwd_dq_inner(
            dq,
            p_k,
            ds_sum,
            p_sum,
            q,
            do,
            m,
            inv_l,
            Di,
            desc_k,
            desc_v,
            iter_offset,
            0,
            hi,
            qk_scale,
            offs_fixed,
            N_CTX,
            BLOCK_ITER,
            CAUSAL,
            WINDOW,
            WARP_SPECIALIZE,
            dtype,
            INPUT_PRECISION,
            UNEVEN_CTX,
            True,
        )

    # ***** 6) store gradient dQ *****

    lam = tl.where(p_sum > 0, ds_sum / p_sum, 0.0)
    tl.store(LAM + offs_fixed, lam, mask=offs_fixed < N_CTX)
    dq = (dq - lam[:, None] * p_k) * sm_scale
    # to avoid writing out of bounds when N_CTX is not divisible by BLOCK_FIXED, the block size of desc_dq is set to be smaller in the last block, so we only write the in-bounds values
    if UNEVEN_CTX and tail_fixed_block:
        dq_ptrs = (
            dq_ptr
            + off_hz.to(tl.int64) * N_CTX * HEAD_DIM
            + offs_fixed[:, None] * HEAD_DIM
            + tl.arange(0, HEAD_DIM)[None, :]
        )
        tl.store(dq_ptrs, dq.to(dtype), mask=offs_fixed[:, None] < N_CTX)
    else:
        desc_dq.store([fixed_offset, 0], dq.to(dtype))


def _system_specific_settings(q, k, v, o, warp_specialize):
    """Provides addtional performance settings for specific systems.

    These performance settings come from the Triton fused attention example.
    """
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V, HEAD_DIM_O = q.shape[-1], k.shape[-1], v.shape[-1], o.shape[-1]
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

        dummy_block = [
            1,
            1,
        ]  # After autotuning, block_shape will be overwritten with the optimal block size in host_descriptor_pre_hook()
        desc_q = TensorDescriptor(
            q,
            shape=[q.shape[0] * q.shape[1] * q.shape[2], HEAD_DIM_Q],
            strides=[HEAD_DIM_Q, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[v.shape[0] * v.shape[1] * v.shape[2], HEAD_DIM_V],
            strides=[HEAD_DIM_V, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[k.shape[0] * k.shape[1] * k.shape[2], HEAD_DIM_K],
            strides=[HEAD_DIM_K, 1],
            block_shape=dummy_block,
        )

        desc_o = TensorDescriptor(
            o,
            shape=[o.shape[0] * o.shape[1] * o.shape[2], HEAD_DIM_O],
            strides=[HEAD_DIM_O, 1],
            block_shape=dummy_block,
        )
    else:
        if not TENSOR_DESCRIPTOR_SUPPORTED:
            raise NotImplementedError(
                "TritonAttention requires either tensor descriptors or host descriptors, which are not supported on this system/triton version combination. Please consider updating your triton version to use TritonAttention. Alternatively, you can open a ticket on the anemoi-core repository to request support for your system."
            )
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

    return desc_q, desc_k, desc_v, desc_o, extra_kern_args


def _dot_input_precision() -> str:
    """Precision of the kernels' matrix products for float32 inputs, following PyTorch's float32 matmul setting.

    "ieee" (full float32) unless PyTorch allows TF32 for matmuls, then "tf32". Inputs in 16 bits are
    not affected. AMD GPUs run TF32 products only on some models, so there the products always run in
    full float32.
    """
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        allow_tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32"
    else:
        allow_tf32 = torch.get_float32_matmul_precision() != "highest"
    return "tf32" if allow_tf32 and not is_hip() else "ieee"


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, window, sm_scale):
        """This function implements a tiled version of the attention algorithm.
        Causal and sliding window masking are supported. Arbitrary context lengths are supported.
        Currently the following Head dimensions are supported: 16, 32, 64, 128, 256.
        Sequence lengths are assumed to match for Q, K and V tensors.

        Input matrices are in the shape [BATCH, N_HEAD, N_CTX, HEAD_DIM]

        Inputs may be float16, bfloat16 or float32. With float32 inputs the matrix products follow
        PyTorch's float32 matmul precision (torch.backends.cuda.matmul.fp32_precision, or
        torch.set_float32_matmul_precision): full float32 by default, TF32 when PyTorch allows it,
        which is faster but rounds the inputs to a 10-bit mantissa. The setting in force at the
        forward is also used for the backward. float16 and bfloat16 inputs are not affected.
        """

        # Ensure inputs are contiguous, important when working with pointers later
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)

        if window is None:
            window = -1  # If window is not specified, set to -1 to indicate no window masking
        assert not (causal and window >= 0), "causal and window not supported in combination"
        assert q.ndim == k.ndim == v.ndim == 4, "TritonAttention expects [batch, head, sequence, dim] tensors"
        assert (
            q.shape[:3] == k.shape[:3] == v.shape[:3]
        ), "TritonAttention requires q, k and v to share batch, head, and sequence dimensions"
        assert q.dtype == k.dtype == v.dtype, "TritonAttention requires q, k and v to use the same dtype"
        assert q.device == k.device == v.device, "TritonAttention requires q, k and v to be on the same device"

        n_ctx = q.shape[2]
        MAX_BLOCK_SIZE = 128  # not possible to infer BLOCK_SIZE determined by autotuning at this point, so we use the max possible block size to ensure we never read out of bounds.
        n_ctx_rounded = math.ceil(n_ctx / MAX_BLOCK_SIZE) * MAX_BLOCK_SIZE

        # Pad M tensor to avoid out-of-bounds reads when N_CTX is not a multiple of BLOCK_FIXED.
        uneven_ctx = n_ctx_rounded != n_ctx
        M = torch.empty((q.shape[0], q.shape[1], n_ctx_rounded), device=q.device, dtype=torch.float32)
        # inv_l (1/l_i per token) is kept apart from M rather than folded into
        # M = m + log2(l).  See _attn_fwd epilogue for rationale.
        inv_l = torch.empty((q.shape[0], q.shape[1], n_ctx_rounded), device=q.device, dtype=torch.float32)

        # Convert tensors from raw pointers to (host) tensor descriptors if the system supports it,
        # Tensor descriptors encode the shape, stride and block shape and pass this information to the compiler, allowing further optimisations and use of hardware features like TMA.
        # and get any additional system-specific kernel arguments
        desc_q, desc_k, desc_v, desc_o, extra_kern_args = _system_specific_settings(q, k, v, o, False)
        input_precision = _dot_input_precision()

        # defines how blocks in the q,k and v input matrices are distributed across SMs on a GPU
        # (SMs are essentially processors on a GPU, with typically 1024 threads per SM)
        # Here a 2D grid is defined: NUM_CTX/BLOCK_ * (BATCH_SIZE * NUM_HEADS)
        # Meaning there is at least (BATCH_SIZE * NUM_HEADS) SMs
        # Depending on BLOCK_FIXED, the context window might also be split across SMs
        # BLOCK_FIXED is a hyperparameter which triton sets at runtime by running small performance tests
        def grid(META):
            return (triton.cdiv(n_ctx, META["BLOCK_FIXED"]), q.shape[0] * q.shape[1], 1)

        _attn_fwd[grid](
            sm_scale,  # scaling factor applied to softmax
            M,  # row-wise softmax max values
            inv_l,  # row-wise inverse softmax sum
            q.shape[0],  # number of batches
            q.shape[1],  # number of heads
            # tensor descriptors for inputs and outputs
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            o,  # raw output tensor (required for uneven ctx case to handle tail case where a smaller block size is needed to avoid overwriting neighbours data)
            N_CTX=n_ctx,  # context length,
            HEAD_DIM=HEAD_DIM_K,  # head dimension
            WINDOW=window,  # window length for sliding window attention. If negative, no sliding window masking is applied
            CAUSAL=causal,  # whether to apply causal masking
            dtype=torch_dtype_to_triton(q.dtype),
            INPUT_PRECISION=input_precision,
            n_ctx_rounded=n_ctx_rounded,  # rounded context length used for indexing in M and D tensors to avoid out-of-bounds when N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER
            UNEVEN_CTX=uneven_ctx,  # bool, whether N_CTX is not divisible by BLOCK_FIXED or BLOCK_ITER, used to handle edge case with masked loads and stores
            **extra_kern_args,  # any additional system-specific kernel arguments
        )

        # save tensors and parameters needed for backward pass in context object
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.window = window
        ctx.n_ctx = n_ctx
        ctx.input_precision = input_precision  # the backward uses the forward's precision

        ctx.save_for_backward(q, k, v, o, M, inv_l)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, inv_l = ctx.saved_tensors

        if do.shape == o.shape and do.stride() != o.stride():
            do = do.reshape(o.shape)

        do = do.contiguous()

        dq = torch.empty_like(q).contiguous()
        dk = torch.empty_like(k).contiguous()
        dv = torch.empty_like(v).contiguous()
        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape

        delta = torch.empty_like(M)
        lam = torch.empty_like(M)

        n_ctx = q.shape[2]
        MAX_BLOCK_SIZE = 128
        # not possible to infer BLOCK_SIZE determined by autotuning at this point, so we use the max possible block size to ensure we never read out of bounds.
        n_ctx_rounded = math.ceil(n_ctx / MAX_BLOCK_SIZE) * MAX_BLOCK_SIZE
        uneven_ctx = n_ctx_rounded != n_ctx

        desc_q, desc_k, desc_v, desc_o, extra_kern_args = _system_specific_settings(q, k, v, o, False)
        desc_dq, desc_dk, desc_dv, desc_do, extra_kern_args = _system_specific_settings(dq, dk, dv, do, False)

        # defines how blocks in the q,k and v input matrices are distributed across SMs on a GPU
        # (SMs are essentially processors on a GPU, with typically 1024 threads per SM)
        # There is at least one kernel per BATCH * NUM_HEAD.
        # Depending on BLOCK_FIXED, the context dimension can be further split across kernels
        def grid_dkdv(META):
            return (triton.cdiv(n_ctx, META["BLOCK_FIXED"]), 1, BATCH * N_HEAD)

        def grid_dq(META):
            return (triton.cdiv(n_ctx, META["BLOCK_FIXED"]), 1, BATCH * N_HEAD)

        # for some reason, when using device-side tensor descriptors, the allocator must be set explictly before the backward pass, otherwise triton complains no allocator has been set
        if not supports_host_descriptor():
            set_allocator()

        # Compute dQ first: it also writes delta and lam, which the dK/dV kernel reads.
        _attn_bwd_dq[grid_dq](
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            desc_dq,
            # need to pass raw pointer in uneven ctx case
            dq,
            o,
            M,
            inv_l,
            delta,  #
            lam,
            ctx.sm_scale,
            Z=BATCH,
            H=N_HEAD,
            N_CTX=N_CTX,  #
            HEAD_DIM=HEAD_DIM,  #
            CAUSAL=ctx.causal,  #
            WINDOW=ctx.window,  #
            dtype=torch_dtype_to_triton(q.dtype),
            INPUT_PRECISION=ctx.input_precision,
            n_ctx_rounded=n_ctx_rounded,
            UNEVEN_CTX=uneven_ctx,
            **extra_kern_args,
        )

        # Compute dK and dV
        _attn_bwd_dkdv[grid_dkdv](
            desc_q,
            desc_k,
            desc_v,
            desc_do,
            desc_dk,
            desc_dv,
            # need to pass raw pointer in uneven ctx case
            dk,
            dv,
            M,
            inv_l,
            delta,  #
            lam,
            ctx.sm_scale,
            Z=BATCH,
            H=N_HEAD,
            N_CTX=N_CTX,  #
            HEAD_DIM=HEAD_DIM,  #
            CAUSAL=ctx.causal,  #
            WINDOW=ctx.window,  #
            dtype=torch_dtype_to_triton(q.dtype),
            INPUT_PRECISION=ctx.input_precision,
            n_ctx_rounded=n_ctx_rounded,
            UNEVEN_CTX=uneven_ctx,
            **extra_kern_args,
        )

        return dq, dk, dv, None, None, None
