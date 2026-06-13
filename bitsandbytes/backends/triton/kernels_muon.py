# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Fused Triton kernels for the elementwise phases of the Muon8bit / Muon4bit step.

8-bit kernel
------------
Replaces the eager sequence (dequantize_blockwise -> m.mul_(beta).add_(g) ->
u = g + beta*m -> stacked.copy_(u) -> quantize_blockwise), i.e. ~6 passes over
the data, with a single pass:

    m  = dequant(state1, absmax1)        # blockwise dynamic 8-bit
    m  = beta * m + g
    u  = g + beta * m   (nesterov)  |  m
    state1, absmax1 <- requant(m)        # in place
    u_out <- u (cast to u_out dtype, normally bf16 NS input buffer)

Quantization mirrors bitsandbytes.backends.triton.kernels_8bit_quant
(bisection over the sorted 256-entry dynamic code, nearest-level rounding),
with a guard for all-zero blocks.

4-bit NF4 kernel
----------------
Same single-pass design for 4-bit NormalFloat (NF4) quantized momentum:

    m  = dequant_nf4(state1, absmax1)    # packed uint8, two 4-bit codes per byte
    m  = beta * m + g
    u  = g + beta * m   (nesterov)  |  m
    state1, absmax1 <- requant_nf4(m)    # in place
    u_out <- u

NF4 uses a 16-entry table (quantiles of N(0,1)) with a hardcoded decision tree
for both dequantization and quantization, matching the existing Triton kernels
in bitsandbytes.backends.triton.kernels_4bit.
"""
from __future__ import annotations

import torch

import triton
import triton.language as tl

from bitsandbytes.backends.triton.kernels_8bit_quant import dequant_8bit_blockwise_kernel_util

_BLOCKSIZE = 256


@triton.jit
def _muon_momentum_8bit_fused_kernel(
    g_ptr,
    state1_ptr,
    absmax_ptr,
    qmap_ptr,
    u_ptr,
    beta,
    n_elements,
    n_blocks,
    NESTEROV: tl.constexpr,
    CODE_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    N_PER_TH: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start_idx = pid * N_PER_TH
    offsets = block_start_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N * N_PER_TH)
    mask = offsets < n_elements

    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    m = dequant_8bit_blockwise_kernel_util(state1_ptr, offsets, qmap_ptr, absmax_ptr, mask, BLOCK_SIZE_N)

    # Momentum update: m = beta*m + g; NS input u = g + beta*m (nesterov) or m
    m = m * beta + g
    if NESTEROV:
        u = g + beta * m
    else:
        u = m
    tl.store(u_ptr + offsets, u.to(u_ptr.dtype.element_ty), mask=mask)

    # Requantize m blockwise (nearest level in the sorted dynamic code).
    m_blocks = tl.reshape(m, (N_PER_TH, BLOCK_SIZE_N))
    absmax_new = tl.max(tl.abs(m_blocks), axis=1)
    # Guard all-zero blocks (e.g. zero gradients): normalized value is 0,
    # which quantizes to the code level nearest 0; stored absmax stays 0.
    m_norm = m_blocks / tl.where(absmax_new == 0.0, 1.0, absmax_new)[:, None]
    m_norm = tl.clamp(m_norm, -1.0, 1.0)

    lower = tl.zeros((N_PER_TH, BLOCK_SIZE_N), dtype=tl.int32)
    upper = tl.full((N_PER_TH, BLOCK_SIZE_N), CODE_SIZE - 1, dtype=tl.int32)
    for _ in range(8):  # ceil(log2(CODE_SIZE)) with CODE_SIZE = 256
        pivot = (lower + upper) // 2
        pivot_val = tl.load(qmap_ptr + pivot)
        is_higher = m_norm > pivot_val
        lower = tl.where(is_higher, pivot, lower)
        upper = tl.where(is_higher, upper, pivot)

    lower_val = tl.load(qmap_ptr + lower)
    upper_val = tl.load(qmap_ptr + upper)
    codes = tl.where(
        tl.abs(m_norm - lower_val) <= tl.abs(m_norm - upper_val), lower, upper
    ).to(tl.uint8)

    tl.store(state1_ptr + offsets, tl.reshape(codes, (BLOCK_SIZE_N * N_PER_TH,)), mask=mask)
    absmax_offsets = block_start_idx + tl.arange(0, N_PER_TH)
    tl.store(absmax_ptr + absmax_offsets, absmax_new, mask=absmax_offsets < n_blocks)


def muon_momentum_8bit_fused(
    grad: torch.Tensor,
    state1: torch.Tensor,
    absmax1: torch.Tensor,
    qmap: torch.Tensor,
    u_out: torch.Tensor,
    beta: float,
    nesterov: bool,
) -> None:
    """Single-pass momentum update on 8-bit state, writing the NS input to u_out.

    state1 and absmax1 are updated in place. All tensors must be contiguous
    and on the same CUDA device; u_out must have grad's shape (any float dtype).
    """
    n = grad.numel()
    n_per_th = 8
    grid = (triton.cdiv(n, _BLOCKSIZE * n_per_th),)
    _muon_momentum_8bit_fused_kernel[grid](
        grad,
        state1,
        absmax1,
        qmap,
        u_out,
        beta,
        n,
        absmax1.numel(),
        NESTEROV=nesterov,
        CODE_SIZE=256,
        BLOCK_SIZE_N=_BLOCKSIZE,
        N_PER_TH=n_per_th,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# 4-bit NF4 fused momentum kernel
# ---------------------------------------------------------------------------
_4BIT_BLOCKSIZE = 64  # default quantisation blocksize for 4-bit


@triton.jit
def _dequantize_nf4(val):
    """Map a 4-bit NF4 code (integer 0-15) to its normalised float value.

    Matches the lookup table in bitsandbytes.functional.get_4bit_type("nf4"):
      code  0 → -1.0,   1 → -0.6962,  2 → -0.5251,  3 → -0.3949
      code  4 → -0.2844, 5 → -0.1848,  6 → -0.0911,  7 →  0.0
      code  8 →  0.0796,  9 →  0.1609, 10 →  0.2461, 11 →  0.3379
      code 12 →  0.4407, 13 →  0.5626, 14 →  0.7230, 15 →  1.0
    """
    cond0 = (val & 0b1000) == 0b1000
    cond1 = (val & 0b0100) == 0b0100
    cond2 = (val & 0b0010) == 0b0010
    cond3 = (val & 0b0001) == 0b0001

    branch_pos = tl.where(
        cond1,
        tl.where(
            cond2,
            tl.where(cond3, 1.0, 0.7229568362236023),
            tl.where(cond3, 0.5626170039176941, 0.44070982933044434),
        ),
        tl.where(
            cond2,
            tl.where(cond3, 0.33791524171829224, 0.24611230194568634),
            tl.where(cond3, 0.16093020141124725, 0.07958029955625534),
        ),
    )
    branch_neg = tl.where(
        cond1,
        tl.where(
            cond2,
            tl.where(cond3, 0.0, -0.09105003625154495),
            tl.where(cond3, -0.18477343022823334, -0.28444138169288635),
        ),
        tl.where(
            cond2,
            tl.where(cond3, -0.39491748809814453, -0.5250730514526367),
            tl.where(cond3, -0.6961928009986877, -1.0),
        ),
    )
    return tl.where(cond0, branch_pos, branch_neg)


@triton.jit
def _quantize_nf4(x):
    """Map a normalised float in [-1, 1] to the nearest NF4 code (0-15).

    Thresholds are midpoints between adjacent NF4 quantisation levels.
    Matches the decision tree in kernels_4bit.quantize_nf4_blockwise_kernel.
    """
    return tl.where(
        x > 0.03979014977812767,
        tl.where(
            x > 0.3893125355243683,
            tl.where(
                x > 0.6427869200706482,
                tl.where(x > 0.8614784181118011, 15, 14),
                tl.where(x > 0.5016634166240692, 13, 12),
            ),
            tl.where(
                x > 0.2035212516784668,
                tl.where(x > 0.2920137718319893, 11, 10),
                tl.where(x > 0.1202552504837513, 9, 8),
            ),
        ),
        tl.where(
            x > -0.33967943489551544,
            tl.where(
                x > -0.13791173323988914,
                tl.where(x > -0.045525018125772476, 7, 6),
                tl.where(x > -0.23460740596055984, 5, 4),
            ),
            tl.where(
                x > -0.6106329262256622,
                tl.where(x > -0.4599952697753906, 3, 2),
                tl.where(x > -0.8480964004993439, 1, 0),
            ),
        ),
    )


@triton.jit
def _muon_momentum_4bit_fused_kernel(
    g_ptr,
    state1_ptr,   # packed uint8, shape (n_paired,) = (ceil(n_elements / 2),)
    absmax_ptr,   # float32, shape (n_blocks,)  = (ceil(n_elements / BLOCKSIZE),)
    u_ptr,        # output, any float dtype, shape (n_elements,)
    beta,
    n_elements,   # total number of momentum elements
    n_blocks,     # ceil(n_elements / BLOCKSIZE)
    n_paired,     # ceil(n_elements / 2)
    NESTEROV: tl.constexpr,
    BLOCKSIZE: tl.constexpr,   # NF4 quantisation blocksize (e.g. 64)
    N_PER_TH: tl.constexpr,   # number of quant blocks handled per program
):
    """Single-pass NF4 momentum update kernel.

    Each program covers N_PER_TH consecutive quantisation blocks:
      - Loads N_PER_TH*BLOCKSIZE gradient elements.
      - Loads N_PER_TH*BLOCKSIZE/2 packed state bytes and N_PER_TH absmax values.
      - Dequantises, updates momentum, writes NS input u, requantises in place.

    Packing convention (matching quantize_4bit / kernels_4bit):
      packed_byte[j] = (code_for_element_2j << 4) | code_for_element_2j+1
    """
    ELEMS_PER_PROG: tl.constexpr = N_PER_TH * BLOCKSIZE
    PAIRS_PER_PROG: tl.constexpr = N_PER_TH * BLOCKSIZE // 2

    pid = tl.program_id(axis=0)
    blk_start = pid * N_PER_TH          # index of first quant block this program owns
    elem_start = blk_start * BLOCKSIZE  # index of first element
    byte_start = blk_start * BLOCKSIZE // 2  # index of first packed byte

    # ---- Load gradient -------------------------------------------------------
    elem_offsets = elem_start + tl.arange(0, ELEMS_PER_PROG)
    elem_mask = elem_offsets < n_elements
    g_flat = tl.load(g_ptr + elem_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    g_2d = tl.reshape(g_flat, (N_PER_TH, BLOCKSIZE))

    # ---- Load packed NF4 state -----------------------------------------------
    byte_offsets = byte_start + tl.arange(0, PAIRS_PER_PROG)
    byte_mask = byte_offsets < n_paired
    packed = tl.load(state1_ptr + byte_offsets, mask=byte_mask, other=0).to(tl.uint8)
    packed_2d = tl.reshape(packed, (N_PER_TH, BLOCKSIZE // 2))

    # ---- Load absmax (one per quant block) -----------------------------------
    absmax_offsets = blk_start + tl.arange(0, N_PER_TH)
    absmax = tl.load(absmax_ptr + absmax_offsets, mask=absmax_offsets < n_blocks, other=0.0)

    # ---- Dequantise ----------------------------------------------------------
    # Packing: high nibble (bits 7:4) = first element of pair,
    #          low  nibble (bits 3:0) = second element of pair.
    code_hi = (packed_2d >> 4) & 0xF   # (N_PER_TH, BLOCKSIZE//2) — first of each pair
    code_lo = packed_2d & 0xF           # (N_PER_TH, BLOCKSIZE//2) — second of each pair

    val_hi = _dequantize_nf4(code_hi) * absmax[:, None]   # (N_PER_TH, BLOCKSIZE//2)
    val_lo = _dequantize_nf4(code_lo) * absmax[:, None]   # (N_PER_TH, BLOCKSIZE//2)

    # tl.interleave along the last axis:
    # result[b, 2*j] = val_hi[b,j],  result[b, 2*j+1] = val_lo[b,j]
    m_2d = tl.interleave(val_hi, val_lo)  # (N_PER_TH, BLOCKSIZE)

    # ---- Momentum update -----------------------------------------------------
    m_2d = beta * m_2d + g_2d
    if NESTEROV:
        u_2d = g_2d + beta * m_2d
    else:
        u_2d = m_2d

    # ---- Write NS input u_out ------------------------------------------------
    u_flat = tl.reshape(u_2d, (ELEMS_PER_PROG,))
    tl.store(u_ptr + elem_offsets, u_flat.to(u_ptr.dtype.element_ty), mask=elem_mask)

    # ---- Requantise ----------------------------------------------------------
    # New per-block absmax
    absmax_new = tl.max(tl.abs(m_2d), axis=1)  # (N_PER_TH,)

    # Normalise into [-1, 1]
    m_norm = m_2d / tl.where(absmax_new == 0.0, 1.0, absmax_new)[:, None]
    m_norm = tl.clamp(m_norm, -1.0, 1.0)

    # Quantise via NF4 decision tree → (N_PER_TH, BLOCKSIZE) uint8 codes
    code_all = _quantize_nf4(m_norm).to(tl.uint8)

    # Pack pairs: reshape to (N_PER_TH, BLOCKSIZE//2, 2), split along last axis
    code_3d = tl.reshape(code_all, (N_PER_TH, BLOCKSIZE // 2, 2))
    left_codes, right_codes = code_3d.split()   # each (N_PER_TH, BLOCKSIZE//2)
    packed_new = (left_codes << 4) | (right_codes & 0xF)
    packed_flat = tl.reshape(packed_new, (PAIRS_PER_PROG,))

    tl.store(state1_ptr + byte_offsets, packed_flat, mask=byte_mask)
    tl.store(absmax_ptr + absmax_offsets, absmax_new, mask=absmax_offsets < n_blocks)


# ---------------------------------------------------------------------------
# NVFP4 (e2m1) fused momentum kernel
# ---------------------------------------------------------------------------
# NVIDIA FP4 (e2m1, bias=1) bit layout per 4-bit code:
#   bit 3 = sign  (1 → negative)
#   bits 2-0 = magnitude code 0-7:
#     0 →  0      1 →  0.5    2 →  1.0    3 →  1.5
#     4 →  2.0    5 →  3.0    6 →  4.0    7 →  6.0
# Normalized by max (6.0): {0, 1/12, 1/6, 1/4, 1/3, 1/2, 2/3, 1}
#
# Unlike NF4, NVFP4 is sign-magnitude, making the trees simpler and symmetric.
# Quantize midpoints (absolute, in [-1,1] domain): 0.0417, 0.125, 0.2083,
# 0.2917, 0.4167, 0.5833, 0.8333.  (NVFP4 magnitudes / 6: 0→0, 1→1/12,
# 2→1/6, 3→1/4, 4→1/3, 5→1/2, 6→2/3, 7→1.)


@triton.jit
def _dequantize_nvfp4(val):
    """Map a 4-bit NVFP4 code (integer 0-15) to its normalised float in [-1, 1].

    Bit layout: bit3=sign, bits2:0=magnitude (0-7).
    Magnitudes (normalised by max=6): {0, 1/12, 1/6, 1/4, 1/3, 1/2, 2/3, 1}.
    """
    sign = tl.where((val & 0b1000) != 0, -1.0, 1.0)
    mag = val & 0b0111  # 0-7

    # 3-level binary tree over magnitude codes
    mag_val = tl.where(
        mag >= 4,
        tl.where(
            mag >= 6,
            tl.where(mag >= 7, 1.0, 4.0 / 6.0),          # 7 → 1.0,  6 → 2/3
            tl.where(mag >= 5, 3.0 / 6.0, 2.0 / 6.0),    # 5 → 0.5,  4 → 1/3
        ),
        tl.where(
            mag >= 2,
            tl.where(mag >= 3, 1.5 / 6.0, 1.0 / 6.0),   # 3 → 1/4,  2 → 1/6
            tl.where(mag >= 1, 0.5 / 6.0, 0.0),          # 1 → 1/12, 0 → 0
        ),
    )
    return sign * mag_val


@triton.jit
def _quantize_nvfp4(x):
    """Map a normalised float in [-1, 1] to the nearest NVFP4 code (0-15).

    Sign-magnitude: code = sign_bit (8 if negative) | magnitude_code (0-7).
    Thresholds are midpoints between adjacent magnitude levels in [-1, 1]:
      {0.0417, 0.125, 0.2083, 0.2917, 0.4167, 0.5833, 0.8333}
    """
    sign_bit = tl.where(x < 0.0, 8, 0)
    x_abs = tl.abs(x)

    mag = tl.where(
        x_abs >= 4.0 / 6.0 + 1.0 / 12.0,   # ≥ 0.8333 → 7
        7,
        tl.where(
            x_abs >= 3.5 / 6.0,              # ≥ 0.5833 → 6
            6,
            tl.where(
                x_abs >= 2.5 / 6.0,          # ≥ 0.4167 → 5
                5,
                tl.where(
                    x_abs >= 1.75 / 6.0,     # ≥ 0.2917 → 4
                    4,
                    tl.where(
                        x_abs >= 1.25 / 6.0, # ≥ 0.2083 → 3
                        3,
                        tl.where(
                            x_abs >= 0.75 / 6.0,  # ≥ 0.125 → 2
                            2,
                            tl.where(
                                x_abs >= 0.25 / 6.0,  # ≥ 0.0417 → 1
                                1,
                                0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return sign_bit | mag


@triton.jit
def _muon_momentum_nvfp4_fused_kernel(
    g_ptr,
    state1_ptr,   # packed uint8, shape (n_paired,) = (ceil(n_elements / 2),)
    absmax_ptr,   # float32, shape (n_blocks,) = (ceil(n_elements / BLOCKSIZE),)
    u_ptr,        # output float (any dtype), shape (n_elements,)
    beta,
    n_elements,
    n_blocks,
    n_paired,
    NESTEROV: tl.constexpr,
    BLOCKSIZE: tl.constexpr,   # NVFP4 quantisation blocksize (e.g. 64)
    N_PER_TH: tl.constexpr,   # quant blocks per program
):
    """Single-pass NVFP4 momentum update.  Identical structure to the NF4
    kernel but uses _dequantize_nvfp4 / _quantize_nvfp4 decision trees.

    Packing convention (same as NF4 / quantize_4bit):
      packed_byte[j] = (code_for_element_2j << 4) | code_for_element_2j+1
    """
    ELEMS_PER_PROG: tl.constexpr = N_PER_TH * BLOCKSIZE
    PAIRS_PER_PROG: tl.constexpr = N_PER_TH * BLOCKSIZE // 2

    pid = tl.program_id(axis=0)
    blk_start = pid * N_PER_TH
    elem_start = blk_start * BLOCKSIZE
    byte_start = blk_start * BLOCKSIZE // 2

    elem_offsets = elem_start + tl.arange(0, ELEMS_PER_PROG)
    elem_mask = elem_offsets < n_elements
    g_flat = tl.load(g_ptr + elem_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    g_2d = tl.reshape(g_flat, (N_PER_TH, BLOCKSIZE))

    byte_offsets = byte_start + tl.arange(0, PAIRS_PER_PROG)
    byte_mask = byte_offsets < n_paired
    packed = tl.load(state1_ptr + byte_offsets, mask=byte_mask, other=0).to(tl.uint8)
    packed_2d = tl.reshape(packed, (N_PER_TH, BLOCKSIZE // 2))

    absmax_offsets = blk_start + tl.arange(0, N_PER_TH)
    absmax = tl.load(absmax_ptr + absmax_offsets, mask=absmax_offsets < n_blocks, other=0.0)

    # Dequantise (same nibble convention as NF4 kernel)
    code_hi = (packed_2d >> 4) & 0xF
    code_lo = packed_2d & 0xF
    val_hi = _dequantize_nvfp4(code_hi) * absmax[:, None]
    val_lo = _dequantize_nvfp4(code_lo) * absmax[:, None]
    m_2d = tl.interleave(val_hi, val_lo)  # (N_PER_TH, BLOCKSIZE)

    # Momentum update
    m_2d = beta * m_2d + g_2d
    if NESTEROV:
        u_2d = g_2d + beta * m_2d
    else:
        u_2d = m_2d

    u_flat = tl.reshape(u_2d, (ELEMS_PER_PROG,))
    tl.store(u_ptr + elem_offsets, u_flat.to(u_ptr.dtype.element_ty), mask=elem_mask)

    # Requantise
    absmax_new = tl.max(tl.abs(m_2d), axis=1)
    m_norm = m_2d / tl.where(absmax_new == 0.0, 1.0, absmax_new)[:, None]
    m_norm = tl.clamp(m_norm, -1.0, 1.0)

    code_all = _quantize_nvfp4(m_norm).to(tl.uint8)
    code_3d = tl.reshape(code_all, (N_PER_TH, BLOCKSIZE // 2, 2))
    left_codes, right_codes = code_3d.split()
    packed_new = (left_codes << 4) | (right_codes & 0xF)
    packed_flat = tl.reshape(packed_new, (PAIRS_PER_PROG,))

    tl.store(state1_ptr + byte_offsets, packed_flat, mask=byte_mask)
    tl.store(absmax_ptr + absmax_offsets, absmax_new, mask=absmax_offsets < n_blocks)


def muon_momentum_nvfp4_fused(
    grad: torch.Tensor,
    state1: torch.Tensor,
    absmax1: torch.Tensor,
    u_out: torch.Tensor,
    beta: float,
    nesterov: bool,
    blocksize: int = _4BIT_BLOCKSIZE,
) -> None:
    """Single-pass NVFP4 (e2m1) momentum update, writing the NS input to u_out.

    state1 (packed NVFP4 uint8, ceil(n/2) bytes) and absmax1 are updated
    in place.  All tensors must be contiguous and on the same CUDA device.
    u_out must have grad's element count and any float dtype (normally bf16).
    """
    n = grad.numel()
    n_per_th = 4
    n_paired = (n + 1) // 2
    n_blocks = (n + blocksize - 1) // blocksize
    grid = (triton.cdiv(n, blocksize * n_per_th),)
    _muon_momentum_nvfp4_fused_kernel[grid](
        grad.contiguous(),
        state1,
        absmax1,
        u_out,
        beta,
        n,
        n_blocks,
        n_paired,
        NESTEROV=nesterov,
        BLOCKSIZE=blocksize,
        N_PER_TH=n_per_th,
        num_warps=4,
    )


def muon_momentum_4bit_fused(
    grad: torch.Tensor,
    state1: torch.Tensor,
    absmax1: torch.Tensor,
    u_out: torch.Tensor,
    beta: float,
    nesterov: bool,
    blocksize: int = _4BIT_BLOCKSIZE,
) -> None:
    """Single-pass NF4 momentum update, writing the NS input to u_out.

    state1 (packed NF4 uint8, ceil(n/2) bytes) and absmax1 are updated
    in place.  All tensors must be contiguous and on the same CUDA device.
    u_out must have grad's element count and any float dtype (normally bf16).

    Only NF4 is supported by this kernel; fall back to the eager path for FP4.
    """
    n = grad.numel()
    n_per_th = 4  # quant blocks processed per Triton program
    n_paired = (n + 1) // 2
    n_blocks = (n + blocksize - 1) // blocksize
    grid = (triton.cdiv(n, blocksize * n_per_th),)
    _muon_momentum_4bit_fused_kernel[grid](
        grad.contiguous(),
        state1,
        absmax1,
        u_out,
        beta,
        n,
        n_blocks,
        n_paired,
        NESTEROV=nesterov,
        BLOCKSIZE=blocksize,
        N_PER_TH=n_per_th,
        num_warps=4,
    )
