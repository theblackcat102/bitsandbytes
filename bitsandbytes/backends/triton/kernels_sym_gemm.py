# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Batched symmetric GEMM in Triton for Gram Newton-Schulz on GPUs without
CuTeDSL symmetric-kernel support (e.g. sm_120 workstation Blackwell).

Computes out = beta * C + alpha * (A @ B) for outputs that are known to be
symmetric (Gram matrices and products of polynomials of the same symmetric
matrix). Only the lower-triangular tiles are computed with tl.dot; each
off-diagonal tile is stored twice (as-is and transposed), so ~half the matmul
work of a full GEMM is skipped. C, when given, must itself be symmetric.

Exposes a backend namespace matching gram_newton_schulz's _TORCH_BACKEND
interface (sym_mm / sym_baddbmm / mm / mm_add), so it can be attached to a
GramNewtonSchulz instance as `_kernel_backend`. Non-symmetric products
(mm / mm_add, i.e. the final Q @ X) stay on cuBLAS.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=_configs(), key=["M", "K"])
@triton.jit
def _sym_bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_ob,
    stride_om,
    stride_on,
    alpha,
    beta,
    HAS_C: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(axis=1)
    t = tl.program_id(axis=0)

    # Invert the triangular tile index: largest i with i*(i+1)/2 <= t, then
    # j = t - i*(i+1)/2, giving the (row=i, col=j) tile with i >= j.
    i = ((tl.sqrt(8.0 * t.to(tl.float32) + 1.0) - 1.0) / 2.0).to(tl.int32)
    i = tl.where((i + 1) * (i + 2) // 2 <= t, i + 1, i)  # fix fp rounding
    i = tl.where(i * (i + 1) // 2 > t, i - 1, i)
    j = t - i * (i + 1) // 2

    rm = i * BLOCK + tl.arange(0, BLOCK)
    rn = j * BLOCK + tl.arange(0, BLOCK)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + batch * stride_ab + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + batch * stride_bb + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < M), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc *= alpha
    out_mask = (rm[:, None] < M) & (rn[None, :] < M)
    if HAS_C:
        c = tl.load(
            c_ptr + batch * stride_cb + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            mask=out_mask,
            other=0.0,
        )
        acc += beta * c.to(tl.float32)

    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + batch * stride_ob + rm[:, None] * stride_om + rn[None, :] * stride_on,
        out,
        mask=out_mask,
    )
    if i != j:
        # Mirror the tile across the diagonal.
        tl.store(
            out_ptr + batch * stride_ob + rn[:, None] * stride_om + rm[None, :] * stride_on,
            tl.trans(out),
            mask=(rn[:, None] < M) & (rm[None, :] < M),
        )


def _sym_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """out = beta*C + alpha*(A @ B), output assumed symmetric (M == N)."""
    assert A.ndim == 3 and B.ndim == 3
    bsz, M, K = A.shape
    assert B.shape[1] == K and B.shape[2] == M, "symmetric GEMM requires square output"
    out = torch.empty((bsz, M, M), device=A.device, dtype=A.dtype)

    if C is None:
        # Dummy pointer; HAS_C=False means it is never dereferenced.
        c, sc = A, (0, 0, 0)
    else:
        assert C.shape == out.shape
        c, sc = C, C.stride()

    def grid(meta):
        tiles = triton.cdiv(M, meta["BLOCK"])
        return (tiles * (tiles + 1) // 2, bsz)

    _sym_bmm_kernel[grid](
        A,
        B,
        c,
        out,
        M,
        K,
        *A.stride(),
        *B.stride(),
        *sc,
        *out.stride(),
        alpha,
        beta,
        HAS_C=C is not None,
    )
    return out


def make_triton_sym_backend() -> SimpleNamespace:
    """Backend namespace plugging into GramNewtonSchulz._kernel_backend."""
    return SimpleNamespace(
        sym_mm=lambda A, B: _sym_bmm(A, B, None, 1.0, 0.0),
        sym_baddbmm=lambda A, B, C, alpha=1.0, beta=1.0: _sym_bmm(A, B, C, alpha, beta),
        mm=lambda A, B: A @ B,
        mm_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
    )
