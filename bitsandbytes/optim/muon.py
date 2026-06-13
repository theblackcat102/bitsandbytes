# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Muon optimizer with optional 8-bit or 4-bit quantized momentum buffer.

Design notes (Phase 1 — pure-Python, no new C++ kernels):
  - Persistent state is a single momentum buffer quantized either to
    uint8 (blockwise dynamic 8-bit, blocksize=256) or to packed NF4/FP4
    (4-bit, blocksize=64), matching the Optimizer1State layout used by SGD8bit.
  - Per step, per parameter (or parameter batch):
      1. Dequantize m → fp32 transient.
      2. Momentum update: m = β·m + g; Nesterov input u = g + β·m (or u = m).
      3. Requantize m back into state1/absmax1.
      4. Orthogonalize u via Newton-Schulz in bf16.
      5. Apply weight decay + param update.
  - An `orthogonalize_fn` hook accepts any callable X → X. If the
    `gram_newton_schulz` package is importable, GramNewtonSchulz (torch
    backend; CuTeDSL kernels only on sm90/sm100 with quack installed) is used
    by default; otherwise a pure-torch standard NS fallback.
  - Same-shape parameters are processed in chunks of `ns_chunk_size` through a
    pre-allocated bf16 buffer to bound the transient memory of the batched NS.
  - Parameters with ndim < 2 should be placed in a separate AdamW/SGD group.

Memory footprint (approximate, per trainable element):
  - 32-bit:  4 bytes  (fp32 momentum)
  -  8-bit:  1 byte   (uint8 blockwise dynamic, blocksize=256)
  -  4-bit:  0.5 byte (NF4/FP4 packed, blocksize=64) + absmax overhead ≈ 0.52 bytes total
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import math
from typing import Optional

import torch
from torch import Tensor

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import MockArgs, Optimizer8bit

try:
    from bitsandbytes.backends.triton.kernels_muon import (
        muon_momentum_4bit_fused,
        muon_momentum_8bit_fused,
        muon_momentum_nvfp4_fused,
    )
except ImportError:  # triton not available
    muon_momentum_8bit_fused = None
    muon_momentum_4bit_fused = None
    muon_momentum_nvfp4_fused = None

# ---------------------------------------------------------------------------
# Polar Express coefficients (https://arxiv.org/pdf/2505.16932)
# ---------------------------------------------------------------------------
_UNMODIFIED_POLAR_EXPRESS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
_SAFETY = 1.05
POLAR_EXPRESS_COEFFICIENTS: list[tuple[float, float, float]] = [
    (a / _SAFETY, b / _SAFETY**3, c / _SAFETY**5) for (a, b, c) in _UNMODIFIED_POLAR_EXPRESS
]


# ---------------------------------------------------------------------------
# Pure-PyTorch Newton-Schulz (standard, operates on the full matrix)
# ---------------------------------------------------------------------------
def _standard_newton_schulz(
    X: Tensor,
    coefficients: list[tuple[float, float, float]] = POLAR_EXPRESS_COEFFICIENTS,
    eps: float = 1e-7,
) -> Tensor:
    """
    Standard Newton-Schulz orthogonalization (5 iterations by default).

    Operates in bf16 after Frobenius normalization, matching the gram-repo design.
    Works on a single 2-D matrix or a batch with leading batch dims.

    Args:
        X: (..., m, n) tensor
        coefficients: list of (a, b, c) triples; one per iteration
        eps: small value for safe Frobenius normalization

    Returns:
        Orthogonalized tensor of the same shape as X
    """
    orig_dtype = X.dtype
    orig_shape = X.shape

    # baddbmm requires 3-D batched tensors
    if X.ndim == 2:
        X = X.unsqueeze(0)
    elif X.ndim > 3:
        X = X.reshape(-1, *X.shape[-2:])

    X = X.to(torch.bfloat16)

    # Frobenius normalize
    norm = X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    X = X / norm

    # Determine if we need to transpose for the "tall" case (m > n)
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = X.mT
        transposed = True

    X = X.contiguous()
    for a, b, c in coefficients:
        A = X @ X.mT
        # B = b*A + c*(A @ A);  X = a*X + B @ X   — two fused kernels keep the
        # live temporaries to one Gram-sized and one X-sized tensor.
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a)

    if transposed:
        X = X.mT

    return X.to(orig_dtype).reshape(orig_shape)


# ---------------------------------------------------------------------------
# NVFP4 Python eager helpers (CPU / Triton-unavailable fallback)
# ---------------------------------------------------------------------------
# NVFP4 magnitude values normalised to [-1, 1] (indexed by mag code 0-7):
_NVFP4_MAG_VALUES: list[float] = [
    0.0,
    0.5 / 6,
    1.0 / 6,
    1.5 / 6,
    2.0 / 6,
    3.0 / 6,
    4.0 / 6,
    1.0,
]
# Quantise thresholds: midpoints between adjacent magnitudes (in [0, 1]):
_NVFP4_THRESHOLDS: list[float] = [
    0.25 / 6,  # 0 ↔ 1
    0.75 / 6,  # 1 ↔ 2
    1.25 / 6,  # 2 ↔ 3
    1.75 / 6,  # 3 ↔ 4
    2.5 / 6,  # 4 ↔ 5
    3.5 / 6,  # 5 ↔ 6
    5.0 / 6,  # 6 ↔ 7
]


def _nvfp4_dequantize_eager(
    packed: Tensor,  # uint8, flat, (ceil(n/2),)
    absmax: Tensor,  # float32, (ceil(n/blocksize),)
    n: int,
    blocksize: int,
    device,
) -> Tensor:
    """Pure-Python NVFP4 dequantize (CPU / fallback).  Returns float32 (n,)."""
    code = torch.tensor(_NVFP4_MAG_VALUES, dtype=torch.float32, device=device)
    n_paired = (n + 1) // 2
    pairs_per_block = blocksize // 2

    code_hi = (packed >> 4).long()  # (n_paired,) — first element of each pair
    code_lo = (packed & 0xF).long()  # (n_paired,) — second element

    sign_hi = torch.where(code_hi >= 8, -1.0, 1.0)
    sign_lo = torch.where(code_lo >= 8, -1.0, 1.0)
    val_hi = sign_hi * code[code_hi & 0x7]  # (n_paired,)
    val_lo = sign_lo * code[code_lo & 0x7]  # (n_paired,)

    pair_idx = torch.arange(n_paired, device=device)
    blk_idx = (pair_idx // pairs_per_block).clamp(max=absmax.numel() - 1)
    s = absmax[blk_idx]  # (n_paired,)

    out = torch.empty(n, dtype=torch.float32, device=device)
    out[0::2] = val_hi * s
    n_lo = n_paired if n % 2 == 0 else n_paired - 1
    if n_lo > 0:
        out[1::2] = (val_lo * s)[:n_lo]
    return out


def _nvfp4_quantize_eager(
    m: Tensor,  # float32, flat (n,)
    packed: Tensor,  # uint8, flat (ceil(n/2),) — modified in place
    absmax: Tensor,  # float32 (ceil(n/blocksize),) — modified in place
    blocksize: int,
) -> None:
    """Pure-Python NVFP4 quantise (CPU / fallback).  Updates packed and absmax."""
    n = m.numel()
    n_paired = (n + 1) // 2
    device = m.device
    pairs_per_block = blocksize // 2

    # Pad to even length for pair-processing
    m_pad = m if n % 2 == 0 else torch.cat([m, m.new_zeros(1)])

    # Per-block absmax
    n_blocks = (n + blocksize - 1) // blocksize
    m_blk = m.reshape(-1, blocksize)[:n_blocks]  # last block may be partial
    absmax_new = m_blk.abs().amax(dim=1)  # (n_blocks,)
    absmax.copy_(absmax_new)

    # Per-pair scale
    pair_idx = torch.arange(n_paired, device=device)
    blk_idx = (pair_idx // pairs_per_block).clamp(max=n_blocks - 1)
    s = absmax_new[blk_idx].clamp(min=1e-12)  # (n_paired,)

    # Normalise and quantise (sign-magnitude)
    m_hi = m_pad[0::2] / s  # (n_paired,) — even elements ∈ [-1, 1]
    m_lo = m_pad[1::2] / s  # (n_paired,)

    def _quant_mag(x: Tensor) -> Tensor:
        """Map abs(x) ∈ [0,1] to magnitude code 0-7."""
        x_abs = x.abs().clamp(0.0, 1.0)
        code = torch.zeros_like(x_abs, dtype=torch.long)
        for thresh in _NVFP4_THRESHOLDS:
            code += (x_abs >= thresh).long()
        return code

    sign_hi = (m_hi < 0).long() * 8
    sign_lo = (m_lo < 0).long() * 8
    codes_hi = (sign_hi | _quant_mag(m_hi)).to(torch.uint8)  # (n_paired,)
    codes_lo = (sign_lo | _quant_mag(m_lo)).to(torch.uint8)  # (n_paired,)

    # Pack: high nibble = first element (even index), low nibble = second (odd)
    packed.copy_(((codes_hi << 4) | (codes_lo & 0xF)).to(torch.uint8))


# ---------------------------------------------------------------------------
# NVFP4 x sm100: Newton-Schulz using Blackwell FP4 tensor cores for X @ X.T
# ---------------------------------------------------------------------------


def _to_nvfp4_with_scales(X: Tensor, block_size: int = 16) -> tuple[Tensor, Tensor]:
    """Pack (M, N) bf16/fp32 tensor into NVFP4 with 1xblock_size FP8 block scales.

    Returns:
        X_fp4:  (M, N//2)      dtype=torch.float4_e2m1fn_x2
        scales: (M, N//block_size)  dtype=torch.float8_e4m3fn

    Packing follows the same convention as torch._bfloat16_to_float4_e2m1fn_x2:
      packed_byte[i] = (code[2*i+1] << 4) | (code[2*i] & 0xF)
      i.e. the odd-index element goes to the high nibble.

    Requires N to be a multiple of block_size (= 16 for sm100 BlockWise1x16).
    """
    X = X.float()
    M, N = X.shape
    assert N % block_size == 0, f"_to_nvfp4_with_scales: N={N} must be divisible by block_size={block_size}"
    N_blocks = N // block_size

    # Per-block absmax → FP8 scales
    X_blk = X.reshape(M, N_blocks, block_size)  # (M, Nb, B)
    absmax = X_blk.abs().amax(dim=-1).clamp(min=1e-12)  # (M, Nb)
    FP4_MAX, FP8_MAX = 6.0, 448.0
    scale_fp8 = (absmax / FP4_MAX).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)  # (M, Nb)

    # Normalize to [-FP4_MAX, FP4_MAX] then quantize
    X_norm = (X_blk / absmax.unsqueeze(-1) * FP4_MAX).clamp(-FP4_MAX, FP4_MAX)  # (M, Nb, B)
    X_flat = X_norm.reshape(M, N)  # (M, N)

    x_abs = X_flat.abs()
    sign_bit = (X_flat < 0).to(torch.int32) * 8  # (M, N)
    mag = (x_abs >= 5.0).long() * 7
    for the, lv in [(3.5, 6), (2.5, 5), (1.75, 4), (1.25, 3), (0.75, 2), (0.25, 1)]:
        mag = torch.where((x_abs >= the) & (mag == 0), torch.full_like(mag, lv), mag)
    # Compact equivalent:
    mag = (
        ((x_abs >= 5.0).long()) * 7
        + ((x_abs >= 3.5) & (x_abs < 5.0)).long() * 6
        + ((x_abs >= 2.5) & (x_abs < 3.5)).long() * 5
        + ((x_abs >= 1.75) & (x_abs < 2.5)).long() * 4
        + ((x_abs >= 1.25) & (x_abs < 1.75)).long() * 3
        + ((x_abs >= 0.75) & (x_abs < 1.25)).long() * 2
        + ((x_abs >= 0.25) & (x_abs < 0.75)).long() * 1
    )
    codes = (sign_bit | mag).to(torch.uint8)  # (M, N), values 0-15

    # Pack pairs (odd → high nibble, even → low nibble — matches pack_uint4)
    packed = (codes[:, 1::2].to(torch.int32) << 4 | codes[:, ::2].to(torch.int32) & 0xF).to(torch.uint8)  # (M, N//2)
    X_fp4 = packed.view(torch.float4_e2m1fn_x2)  # (M, N//2)
    return X_fp4, scale_fp8


def _make_sm100_nvfp4_ns_fn(
    coefficients: list[tuple[float, float, float]] = POLAR_EXPRESS_COEFFICIENTS,
    eps: float = 1e-7,
) -> Callable[[Tensor], Tensor]:
    """Return a Newton-Schulz function that uses Blackwell sm100 FP4 tensor
    cores for the X @ X.T GEMM (the most expensive step).

    Only the Gram matrix computation uses FP4 (via torch._scaled_grouped_mm_v2
    with float4_e2m1fn_x2 inputs and float8_e4m3fn block scales, BlockWise1x16,
    SWIZZLE_32_4_4).  All other operations (A@A, polynomial, X update) stay in
    BF16 for numerical stability.  Iteration count and coefficients are the
    same as _standard_newton_schulz.

    Falls back to _standard_newton_schulz if:
    - torch._scaled_grouped_mm_v2 is not available, or
    - N (or M after potential transpose) is not a multiple of 16, or
    - not running on CUDA.
    """
    try:
        from torch.nn.functional import ScalingType, SwizzleType

        _sgmm = torch._scaled_grouped_mm_v2
    except (ImportError, AttributeError):
        return _default_orthogonalize_fn

    def _fp4_gram(X_single: Tensor) -> Tensor:
        """Compute X @ X.T for one (m, n) matrix using sm100 FP4 GEMM → bf16."""
        _m, n = X_single.shape
        if n % 16 != 0:
            # Dimensions not suitable for BlockWise1x16; use bf16 fallback.
            return X_single @ X_single.mT
        X_fp4, scale_X = _to_nvfp4_with_scales(X_single.contiguous(), block_size=16)
        # mat2 = X.T packed as mat2: transpose the packed tensor (each byte
        # still encodes the correct pair of elements for the contracted K dim).
        X_T_fp4 = X_fp4.t().contiguous()  # (n//2, m)
        scale_X_T = scale_X.t().contiguous()  # (n//16, m)
        return _sgmm(
            X_fp4,
            X_T_fp4,
            [scale_X],
            [ScalingType.BlockWise1x16],
            [SwizzleType.SWIZZLE_32_4_4],
            [scale_X_T],
            [ScalingType.BlockWise1x16],
            [SwizzleType.SWIZZLE_32_4_4],
            None,
            torch.bfloat16,
        )

    def _nvfp4_ns(X: Tensor) -> Tensor:
        if not X.is_cuda:
            return _standard_newton_schulz(X, coefficients, eps)

        orig_dtype = X.dtype
        orig_shape = X.shape

        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.reshape(-1, *X.shape[-2:])

        X = X.to(torch.bfloat16)
        norm = X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
        X = X / norm

        transposed = False
        if X.shape[-2] > X.shape[-1]:
            X = X.mT.contiguous()
            transposed = True

        B, _m, _n = X.shape

        for a, b, c in coefficients:
            # FP4 GEMM for X @ X.T (one call per batch element)
            A = torch.stack([_fp4_gram(X[bi]) for bi in range(B)], dim=0)  # (B, m, m)
            # BF16 polynomial: B = b*A + c*(A@A)
            Bpoly = torch.baddbmm(A, A, A, beta=b, alpha=c)
            # BF16 update: X = a*X + Bpoly@X
            X = torch.baddbmm(X, Bpoly, X, beta=a)

        if transposed:
            X = X.mT.contiguous()

        return X.to(orig_dtype).reshape(orig_shape)

    return _nvfp4_ns


def _default_orthogonalize_fn(
    X: Tensor,
) -> Tensor:
    """Default orthogonalize function using pure-PyTorch standard NS."""
    return _standard_newton_schulz(X)


_DEFAULT_ORTHOGONALIZE_FN: Optional[Callable[[Tensor], Tensor]] = None


def newton_schulz(X: Tensor) -> Tensor:
    """Backward-compatible public Newton-Schulz helper.

    The optimizer feeds Newton-Schulz with bf16 momentum directions and uses
    GramNewtonSchulz when available; this helper mirrors that default path for
    tests and external callers that compare against Muon step-by-step.
    """
    return _get_default_orthogonalize_fn()(X.to(torch.bfloat16))


def _make_default_orthogonalize_fn() -> Callable[[Tensor], Tensor]:
    """
    Prefer GramNewtonSchulz (Dao-AILab) when the package is importable.

    The Gram iteration runs on the small nxn Gram matrix instead of the full
    mxn matrix (~2x fewer FLOPs and far smaller transients for rectangular
    weights) and works on any GPU through its torch backend. The CuTeDSL
    symmetric-GEMM kernels are only enabled on sm90/sm100 (H100/B200) when
    quack is installed; consumer/workstation Blackwell (sm120) is not a
    supported kernel target.

    compile_kwargs=None: reduce-overhead mode uses CUDA graphs whose memory
    pools inflate reserved memory across the varying chunk shapes seen here.
    """
    try:
        from gram_newton_schulz import GramNewtonSchulz
    except ImportError:
        return _default_orthogonalize_fn

    use_kernels = False
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) in ((9, 0), (10, 0), (10, 3)):
            try:
                import quack  # noqa: F401

                use_kernels = True
            except ImportError:
                use_kernels = False

    gns = GramNewtonSchulz(ns_use_kernels=use_kernels, compile_kwargs=None)

    if not use_kernels:
        # No CuTeDSL kernels on this GPU: attach our Triton symmetric-GEMM
        # backend (lower-triangle tiles only + mirrored store, fused
        # beta*C + alpha*A@B epilogue). Non-symmetric products stay on cuBLAS.
        try:
            from bitsandbytes.backends.triton.kernels_sym_gemm import make_triton_sym_backend

            gns._kernel_backend = make_triton_sym_backend()
        except ImportError:
            pass

    return gns.__call__


def _get_default_orthogonalize_fn() -> Callable[[Tensor], Tensor]:
    global _DEFAULT_ORTHOGONALIZE_FN
    if _DEFAULT_ORTHOGONALIZE_FN is None:
        _DEFAULT_ORTHOGONALIZE_FN = _make_default_orthogonalize_fn()
    return _DEFAULT_ORTHOGONALIZE_FN


# ---------------------------------------------------------------------------
# LR adjustment helpers (ported from gram-newton-schulz)
# ---------------------------------------------------------------------------
def _adjust_lr_rms_norm(lr: float, shape: tuple[int, ...]) -> float:
    fan_out, fan_in = shape[-2], shape[-1]
    return lr * 0.2 * math.sqrt(max(fan_out, fan_in))


def _adjust_lr_spectral_norm(lr: float, shape: tuple[int, ...]) -> float:
    fan_out, fan_in = shape[-2], shape[-1]
    return lr * math.sqrt(fan_out / fan_in)


_ADJUST_LR_MAP = {
    "rms_norm": _adjust_lr_rms_norm,
    "spectral_norm": _adjust_lr_spectral_norm,
    None: lambda lr, shape: lr,
}


def _resolve_adjust_lr(adjust_lr) -> Callable[[float, tuple[int, ...]], float]:
    if adjust_lr is None:
        return lambda lr, shape: lr
    if isinstance(adjust_lr, str):
        if adjust_lr not in _ADJUST_LR_MAP:
            raise ValueError(
                f"Invalid adjust_lr: {adjust_lr!r}. Must be 'rms_norm', 'spectral_norm', None, or a callable."
            )
        return _ADJUST_LR_MAP[adjust_lr]
    if callable(adjust_lr):
        return adjust_lr
    raise TypeError(f"adjust_lr must be str, None, or callable; got {type(adjust_lr)}")


# ---------------------------------------------------------------------------
# Parameter batching (group same-shape params for batched NS)
# ---------------------------------------------------------------------------
def _create_param_batches(params: list[Tensor]) -> list[list[Tensor]]:
    """Group params by (shape, dtype, device) for a single batched NS call."""
    groups: dict = defaultdict(list)
    for p in params:
        groups[(p.shape, p.dtype, p.device)].append(p)
    batches = []
    for group in groups.values():
        group.sort(key=lambda p: p.data_ptr())
        batches.append(group)
    return batches


def _split_tensor_for_orthogonalization(x: Tensor):
    """Return 2-D matrices and enough metadata to rebuild x."""
    if x.ndim == 2:
        return [x], ("matrix", x.shape)
    if x.ndim == 3:
        return [x[i] for i in range(x.shape[0])], ("batch3d", x.shape)
    return [x.reshape(x.shape[0], -1)], ("flatten", x.shape)


def _reconstruct_tensor_from_matrices(spec, matrices: list[Tensor]) -> Tensor:
    kind, shape = spec
    if kind == "matrix":
        return matrices[0]
    if kind == "batch3d":
        return torch.stack(matrices, dim=0)
    if kind == "flatten":
        return matrices[0].reshape(shape)
    raise RuntimeError(f"Unknown Muon reconstruction spec: {kind}")


def _validate_param_split_fn(param_split_fn: Callable, x: Tensor, splits) -> list[Tensor]:
    fn_name = getattr(param_split_fn, "__name__", repr(param_split_fn))
    if not isinstance(splits, (list, tuple)) or len(splits) == 0:
        raise ValueError(f"param_split_fn ({fn_name}) must return a non-empty list/tuple of tensors")

    split_tensors = list(splits)
    for split in split_tensors:
        if not isinstance(split, torch.Tensor):
            raise TypeError(f"param_split_fn ({fn_name}) returned a non-tensor value: {type(split)}")
        if split.ndim != x.ndim:
            raise ValueError(f"param_split_fn ({fn_name}) must preserve ndim. Input: {x.ndim}D, output: {split.ndim}D")
        if split.ndim < 2:
            raise ValueError(f"param_split_fn ({fn_name}) returned a tensor with fewer than 2 dimensions")
        if x.ndim == 3 and split.shape[0] != x.shape[0]:
            raise ValueError(
                f"param_split_fn ({fn_name}) for 3D tensors must preserve the first dimension. "
                f"Input shape: {tuple(x.shape)}, output shape: {tuple(split.shape)}"
            )
    return split_tensors


def _prepare_orthogonalization_inputs(
    ns_inputs: list[Tensor],
    param_split_fn: Optional[Callable],
):
    matrices_by_shape: dict = defaultdict(list)
    per_param_specs = []

    for x in ns_inputs:
        if param_split_fn is None:
            split_tensors = [x]
        else:
            split_tensors = _validate_param_split_fn(param_split_fn, x, param_split_fn(x))

        param_specs = []
        for split in split_tensors:
            matrices, spec = _split_tensor_for_orthogonalization(split)
            refs = []
            for matrix in matrices:
                shape = tuple(matrix.shape)
                refs.append((shape, len(matrices_by_shape[shape])))
                matrices_by_shape[shape].append(matrix)
            param_specs.append((spec, refs))
        per_param_specs.append(param_specs)

    return matrices_by_shape, per_param_specs


def _reconstruct_orthogonalized_updates(
    orthogonalized_by_shape: dict,
    per_param_specs,
    param_recombine_fn: Optional[Callable],
) -> list[Tensor]:
    updates = []
    for param_specs in per_param_specs:
        split_updates = []
        for spec, refs in param_specs:
            matrices = [orthogonalized_by_shape[shape][idx] for shape, idx in refs]
            split_updates.append(_reconstruct_tensor_from_matrices(spec, matrices))

        if param_recombine_fn is None:
            updates.append(split_updates[0])
        else:
            updates.append(param_recombine_fn(split_updates))
    return updates


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------
class MuonBase(Optimizer8bit):
    """
    Muon optimizer with optional 8-bit quantized momentum buffer.

    Muon (Momentum + Orthogonalization via Newton-Schulz) applies:
      1. EMA momentum update.
      2. Newton-Schulz orthogonalization of the Nesterov / momentum direction.
      3. RMS-norm LR adjustment (optional).
      4. Decoupled weight decay + parameter update.

    Only parameters with ndim >= 2 are supported. Place 1-D params (biases,
    LayerNorm weights) in a separate AdamW or SGD group.

    Arguments:
        params: Parameter groups.
        lr: Learning rate.
        momentum: EMA decay factor β (default 0.95).
        weight_decay: Decoupled weight decay coefficient (default 0.1).
        nesterov: Use Nesterov-style momentum input to NS (default True).
        adjust_lr: LR adjustment strategy applied after orthogonalization.
            - "rms_norm" (default): scale by 0.2 * sqrt(max(fan_out, fan_in))
            - "spectral_norm": scale by sqrt(fan_out / fan_in)
            - None: no adjustment
            - Callable: custom function (lr, shape) -> float
        orthogonalize_fn: Callable X -> X for orthogonalization. Defaults to
            GramNewtonSchulz when the gram_newton_schulz package is available
            (torch backend on any GPU; CuTeDSL kernels on sm90/sm100 with
            quack), else a pure-PyTorch 5-step standard Newton-Schulz with
            Polar Express coefficients.
        ns_chunk_size: Maximum number of same-shape matrices orthogonalized
            per batched NS call. Bounds the step's transient memory; the NS
            working set is roughly 3-4 chunk-sized bf16 buffers.
        optim_bits: 32 for fp32, 8 for 8-bit dynamic quantisation, 4 for 4-bit
            NF4/FP4 quantisation (~0.5 bytes/param for momentum).
        min_8bit_size: Minimum element count to use quantised storage; smaller
            parameters fall back to fp32 regardless of optim_bits.
        quant_type: 4-bit quantisation scheme when optim_bits=4.
            "nf4" (default) uses NormalFloat4 (better for momentum distributions);
            "fp4" uses floating-point 4-bit encoding.
            Ignored when optim_bits != 4.
        is_paged: Use paged (CPU-offload) memory for optimizer state.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        optim_bits: int = 32,
        min_8bit_size: int = 4096,
        quant_type: str = "nf4",
        is_paged: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ns_chunk_size < 1:
            raise ValueError(f"Invalid ns_chunk_size: {ns_chunk_size}")
        if optim_bits not in (4, 8, 32):
            raise ValueError(f"optim_bits must be 4, 8, or 32; got {optim_bits}")
        if quant_type not in ("nf4", "fp4", "nvfp4"):
            raise ValueError(f"quant_type must be 'nf4', 'fp4', or 'nvfp4'; got {quant_type!r}")

        # Pack momentum into betas[0] so GlobalOptimManager override_config works.
        defaults = dict(
            lr=lr,
            betas=(momentum, 0.0),
            eps=1e-8,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults, optim_bits=optim_bits, is_paged=is_paged)

        # Args object expected by get_config() / Optimizer8bit
        args: dict = {
            "optim_bits": optim_bits,
            "min_8bit_size": min_8bit_size,
            "max_unorm": 0.0,
            "skip_zeros": False,
        }
        self.args = MockArgs(args)
        self.optimizer_name = "muon"
        self._quant_type = quant_type  # used only when optim_bits == 4

        self._orthogonalize_fn = orthogonalize_fn if orthogonalize_fn is not None else _get_default_orthogonalize_fn()
        self.ns_chunk_size = ns_chunk_size
        # Fused Triton paths (dequant + momentum + requant + NS-input write in
        # one pass). Auto-enabled when triton imports successfully.
        self._use_fused_8bit = muon_momentum_8bit_fused is not None
        # 4-bit fused kernels: NF4 and NVFP4 have dedicated Triton kernels;
        # bitsandbytes FP4 falls back to the eager path.
        self._use_fused_4bit = muon_momentum_4bit_fused is not None and quant_type == "nf4"
        self._use_fused_nvfp4 = muon_momentum_nvfp4_fused is not None and quant_type == "nvfp4"

        if optim_bits == 8:
            self.fill_qmap()

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        "MuonBase only supports parameters with 2 or more dimensions. "
                        "Place 1-D parameters (biases, norms) in a separate AdamW group."
                    )

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)
        optim_bits = config["optim_bits"]
        n = p.numel()

        state = self.state[p]
        state["step"] = 0

        # Fall back to fp32 for small parameters regardless of optim_bits.
        if n < config["min_8bit_size"] or optim_bits == 32:
            state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
            return

        if optim_bits == 8:
            if "dynamic" not in self.name2qmap:
                self.fill_qmap()
            self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)

            state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
            state["qmap1"] = self.name2qmap["dynamic"]

            blocksize = 256
            blocks = (n + blocksize - 1) // blocksize
            state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)

        elif optim_bits == 4:
            blocksize = 64
            if self._quant_type == "nvfp4":
                # NVFP4 bypasses the C++ kernel entirely; allocate directly.
                # Shape (ceil(n/2), 1) mirrors what quantize_4bit returns so
                # the reshape(-1) in _update_batch always produces contiguous bytes.
                n_paired = (n + 1) // 2
                n_blocks = (n + blocksize - 1) // blocksize
                state["state1"] = torch.zeros(n_paired, 1, dtype=torch.uint8, device=p.device)
                state["absmax1"] = torch.zeros(n_blocks, dtype=torch.float32, device=p.device)
            else:
                # NF4 / FP4: call quantize_4bit once on zeros to get exactly the
                # right packed-shape (ceil(n/2), 1) without guessing it.
                _m_zero = torch.zeros(n, dtype=torch.float32, device=p.device)
                with torch.no_grad():
                    packed_init, quant_state_init = F.quantize_4bit(
                        _m_zero, blocksize=blocksize, quant_type=self._quant_type
                    )
                state["state1"] = packed_init  # (ceil(n/2), 1), uint8
                state["absmax1"] = quant_state_init.absmax
                del _m_zero, packed_init, quant_state_init
            state["quant_type1"] = self._quant_type
            state["blocksize1"] = blocksize

        else:
            raise NotImplementedError(f"Unsupported optim_bits: {optim_bits}")

    # ------------------------------------------------------------------
    # Step override: batch params of same shape, run NS once per batch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            # Collect params with gradients; initialise state lazily
            active = []
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError(
                        "MuonBase only supports parameters with 2 or more dimensions. "
                        "Place 1-D parameters (biases, norms) in a separate AdamW group."
                    )
                p.data = p.data.contiguous()
                p.grad = p.grad.contiguous()
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.prefetch_state(p)
                active.append((pindex, p))

            if not active:
                continue

            # Resolve LR adjustment function once per group
            adjust_lr_fn = _resolve_adjust_lr(group["adjust_lr"])

            # Batch same-shape params together for a single NS call
            param_list = [p for _, p in active]
            for batch in _create_param_batches(param_list):
                self._update_batch(group, gindex, batch, adjust_lr_fn)

        return loss

    # ------------------------------------------------------------------
    # Batch update: momentum → quantize → NS → param update
    # ------------------------------------------------------------------
    def _update_batch(
        self,
        group: dict,
        gindex: int,
        params: list[Tensor],
        adjust_lr_fn: Callable,
    ):
        """Apply one Muon step to a batch of same-shape parameters.

        Same-shape params are processed in chunks of `ns_chunk_size`. Momentum
        is prepared per parameter, then Newton-Schulz inputs are split into 2-D
        matrices, grouped by matrix shape, orthogonalized, LR-scaled by each
        matrix shape, and reconstructed to the original parameter shape.
        """
        beta = group["betas"][0]  # momentum
        nesterov = group["nesterov"]
        lr = group["lr"]
        wd = group["weight_decay"]
        param_split_fn = group.get("param_split_fn", None)
        param_recombine_fn = group.get("param_recombine_fn", None)
        if (param_split_fn is None) != (param_recombine_fn is None):
            raise ValueError("param_split_fn and param_recombine_fn must both be provided or both be None")

        shape = params[0].shape  # all same shape in this batch

        for start in range(0, len(params), self.ns_chunk_size):
            chunk = params[start : start + self.ns_chunk_size]
            ns_inputs: list[Tensor] = []

            for p in chunk:
                state = self.state[p]
                state["step"] += 1
                grad = p.grad
                u_buf = torch.empty_like(p, dtype=torch.bfloat16)

                can_fuse_nvfp4 = (
                    "quant_type1" in state
                    and state["quant_type1"] == "nvfp4"
                    and self._use_fused_nvfp4
                    and grad.is_cuda
                    and grad.is_contiguous()
                    and state["state1"].is_contiguous()
                    and state["absmax1"].is_contiguous()
                    and u_buf.is_contiguous()
                )
                if can_fuse_nvfp4:
                    # Fused Triton NVFP4 path: single pass over data. state1 is
                    # contiguous, so reshape(-1) is a writable flat view.
                    muon_momentum_nvfp4_fused(
                        grad,
                        state["state1"].reshape(-1),
                        state["absmax1"],
                        u_buf,
                        beta,
                        nesterov,
                        blocksize=state["blocksize1"],
                    )
                    ns_inputs.append(u_buf)
                    continue

                can_fuse_4bit = (
                    "quant_type1" in state
                    and state["quant_type1"] == "nf4"
                    and self._use_fused_4bit
                    and grad.is_cuda
                    and grad.is_contiguous()
                    and state["state1"].is_contiguous()
                    and state["absmax1"].is_contiguous()
                    and u_buf.is_contiguous()
                )
                if can_fuse_4bit:
                    # Fused Triton NF4 path: single pass over data. state1 is
                    # contiguous, so reshape(-1) is a writable flat view.
                    muon_momentum_4bit_fused(
                        grad,
                        state["state1"].reshape(-1),
                        state["absmax1"],
                        u_buf,
                        beta,
                        nesterov,
                        blocksize=state["blocksize1"],
                    )
                    ns_inputs.append(u_buf)
                    continue

                can_fuse_8bit = (
                    state["state1"].dtype == torch.uint8
                    and "qmap1" in state
                    and self._use_fused_8bit
                    and grad.is_cuda
                    and grad.is_contiguous()
                    and state["state1"].is_contiguous()
                    and state["absmax1"].is_contiguous()
                    and state["qmap1"].is_contiguous()
                    and u_buf.is_contiguous()
                )
                if can_fuse_8bit:
                    # Fused Triton 8-bit path: one pass does dequant + momentum
                    # + requant (in place) and writes the NS input to u_buf.
                    muon_momentum_8bit_fused(
                        grad,
                        state["state1"],
                        state["absmax1"],
                        state["qmap1"],
                        u_buf,
                        beta,
                        nesterov,
                    )
                    ns_inputs.append(u_buf)
                    continue

                if state["state1"].dtype == torch.float32:
                    m = state["state1"]
                    # m = beta*m + g (mixed-dtype add casts grad in-kernel)
                    m.mul_(beta).add_(grad)
                    u = (grad + beta * m) if nesterov else m
                elif "quant_type1" in state:
                    if state["quant_type1"] == "nvfp4":
                        # NVFP4 eager path: dequantize -> update -> requantize.
                        m = _nvfp4_dequantize_eager(
                            state["state1"].reshape(-1),
                            state["absmax1"],
                            p.numel(),
                            state["blocksize1"],
                            p.device,
                        ).view(shape)
                        m.mul_(beta).add_(grad)
                        u = (grad + beta * m) if nesterov else m
                        _nvfp4_quantize_eager(
                            m.to(torch.float32).reshape(-1),
                            state["state1"].reshape(-1),
                            state["absmax1"],
                            state["blocksize1"],
                        )
                    else:
                        # 4-bit eager path: dequantize -> update -> requantize.
                        quant_state = F.QuantState(
                            absmax=state["absmax1"],
                            shape=p.shape,
                            dtype=torch.float32,
                            blocksize=state["blocksize1"],
                            quant_type=state["quant_type1"],
                        )
                        m = F.dequantize_4bit(state["state1"], quant_state=quant_state)
                        m = m.view(shape)
                        m.mul_(beta).add_(grad)
                        u = (grad + beta * m) if nesterov else m
                        # Requantize; out= and absmax= write into pre-allocated buffers.
                        F.quantize_4bit(
                            m.to(torch.float32).reshape(-1),
                            out=state["state1"],
                            absmax=state["absmax1"],
                            blocksize=state["blocksize1"],
                            quant_type=state["quant_type1"],
                        )
                else:
                    # 8-bit eager path: dequantize -> update -> requantize.
                    # dequantize_blockwise returns fp32 when given raw absmax.
                    m = F.dequantize_blockwise(
                        state["state1"],
                        absmax=state["absmax1"],
                        code=state["qmap1"],
                        blocksize=256,
                    )
                    m.mul_(beta).add_(grad)
                    u = (grad + beta * m) if nesterov else m
                    # Requantize momentum; absmax=/out= write state in place.
                    F.quantize_blockwise(
                        m,
                        code=state["qmap1"],
                        absmax=state["absmax1"],
                        out=state["state1"],
                        blocksize=256,
                    )

                u_buf.copy_(u)
                ns_inputs.append(u_buf)
                del u, m

            if param_split_fn is None and len(shape) == 2:
                # Preserve the historical regular-matrix path exactly: tests
                # compare the CPU 32-bit trajectory with very tight tolerances.
                stacked = torch.stack(ns_inputs, dim=0)
                orthogonalized = self._orthogonalize_fn(stacked)
                adjusted_lr = adjust_lr_fn(lr, shape)
                for p, update in zip(chunk, orthogonalized.unbind(0)):
                    p.data.mul_(1.0 - lr * wd).add_(update, alpha=-adjusted_lr)
                del stacked, orthogonalized
                continue

            matrices_by_shape, per_param_specs = _prepare_orthogonalization_inputs(ns_inputs, param_split_fn)
            orthogonalized_by_shape = {}
            for matrix_shape, matrices in matrices_by_shape.items():
                batched_input = torch.stack(matrices, dim=0)
                orthogonalized = self._orthogonalize_fn(batched_input)
                orthogonalized_by_shape[matrix_shape] = orthogonalized.float().mul(adjust_lr_fn(lr, matrix_shape))

            updates = _reconstruct_orthogonalized_updates(
                orthogonalized_by_shape,
                per_param_specs,
                param_recombine_fn,
            )

            # --- Decoupled weight decay + parameter update ---
            for p, update in zip(chunk, updates):
                if update.shape != p.shape:
                    raise RuntimeError(
                        f"Muon orthogonalized update shape {tuple(update.shape)} does not match "
                        f"parameter shape {tuple(p.shape)}"
                    )
                p.data.mul_(1.0 - lr * wd).add_(update.to(dtype=p.dtype), alpha=-1.0)

            del ns_inputs, matrices_by_shape, orthogonalized_by_shape, updates

    # Satisfy abstract interface (not used since we override step())
    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        pass  # Logic lives in _update_batch; this method is never called.


# ---------------------------------------------------------------------------
# Public API: Muon, Muon8bit, Muon32bit
# ---------------------------------------------------------------------------
class Muon(MuonBase):
    """
    Muon optimizer (32-bit momentum buffer by default).

    For quantized variants see :class:`Muon8bit` and :class:`Muon4bit`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        optim_bits: int = 32,
        min_8bit_size: int = 4096,
        quant_type: str = "nf4",
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=optim_bits,
            min_8bit_size=min_8bit_size,
            quant_type=quant_type,
            is_paged=is_paged,
        )


class Muon8bit(MuonBase):
    """
    Muon optimizer with 8-bit quantized momentum buffer.

    Persistent state is ~1 byte/param (uint8 + absmax overhead), giving
    roughly half the footprint of AdamW8bit for the same model.

    See :class:`MuonBase` for full documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=8,
            min_8bit_size=min_8bit_size,
            is_paged=is_paged,
        )


class Muon32bit(MuonBase):
    """
    Muon optimizer with 32-bit (fp32) momentum buffer.

    See :class:`MuonBase` for full documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=32,
            min_8bit_size=min_8bit_size,
            is_paged=is_paged,
        )


class Muon4bit(MuonBase):
    """
    Muon optimizer with 4-bit quantized momentum buffer.

    Persistent state is ~0.5 bytes/param (two NF4/FP4 codes packed per byte,
    blocksize=64) plus a small fp32 absmax vector (~1/64 bytes/param overhead),
    giving a total of ~0.52 bytes/param — roughly half the footprint of
    Muon8bit and ~8x less than AdamW.

    NF4 (NormalFloat4) is recommended for momentum buffers because the
    quantisation levels are optimal for normally-distributed data.  FP4 is
    provided as an alternative for data with heavy tails.

    When Triton is available and quant_type="nf4", the momentum update
    (dequantise + EMA + Nesterov + requantise + write NS input) is fused
    into a single kernel pass.  FP4 uses the eager path (quantize_4bit /
    dequantize_4bit from bitsandbytes.functional).

    See :class:`MuonBase` for full documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        quant_type: str = "nf4",
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=4,
            min_8bit_size=min_8bit_size,
            quant_type=quant_type,
            is_paged=is_paged,
        )


class Muon4bitNVFP4(MuonBase):
    """
    Muon optimizer with NVIDIA FP4 (e2m1) quantized momentum buffer.

    Momentum storage: ~0.52 bytes/param (packed e2m1 uint8, blocksize=64
    + fp32 absmax overhead) — same footprint as Muon4bit(quant_type='nf4').

    On Blackwell (sm100, B200/B300) the Newton-Schulz X @ X.T Gram GEMM is
    routed through ``torch._scaled_grouped_mm_v2`` using float4_e2m1fn_x2
    inputs with float8_e4m3fn block scales (BlockWise1x16, SWIZZLE_32_4_4),
    activating native FP4 tensor core instructions (tcgen05.mma).  All other
    NS ops (A@A, B@X update, polynomial) stay in BF16.  On non-sm100 hardware
    the standard BF16 NS path is used automatically.

    The elementwise momentum update (dequant → EMA → requant) always runs as
    a fused Triton kernel using software NVFP4 encoding — there are no scalar
    FP4 ALU instructions on sm100; only tensor cores use FP4.

    See :class:`MuonBase` for full parameter documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        # Activate the FP4 tensor-core NS path only on data-center Blackwell
        # (sm100, B200/B300). It relies on BlockWise1x16 NVFP4 scaled GEMM
        # (torch._scaled_grouped_mm_v2) which is not available on consumer/
        # workstation Blackwell (sm120) or earlier; those use the standard
        # bf16 NS path. Capabilities mirror _make_default_orthogonalize_fn.
        if orthogonalize_fn is None:
            try:
                cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
                if cap in ((10, 0), (10, 3)):
                    orthogonalize_fn = _make_sm100_nvfp4_ns_fn()
            except Exception:
                pass
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=4,
            min_8bit_size=min_8bit_size,
            quant_type="nvfp4",
            is_paged=is_paged,
        )
