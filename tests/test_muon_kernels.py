"""Equivalence tests for the fused Muon Triton kernels.

The kernels (kernels_muon.py) fuse dequant -> momentum/Nesterov -> requant ->
NS-input write into one pass. Each kernel is compared against an independent
*pure-torch* reference of the same quant scheme, so these tests do not depend
on the compiled bitsandbytes C++ library (only CUDA + Triton).

The NVFP4 reference reuses the eager helpers shipped in muon.py
(_nvfp4_quantize_eager / _nvfp4_dequantize_eager).
"""

import math

import pytest
import torch

import bitsandbytes.functional as F
from bitsandbytes.optim.muon import _nvfp4_dequantize_eager, _nvfp4_quantize_eager

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("Muon Triton kernels require a CUDA device", allow_module_level=True)

from bitsandbytes.backends.triton.kernels_muon import (  # noqa: E402
    muon_momentum_4bit_fused,
    muon_momentum_8bit_fused,
    muon_momentum_nvfp4_fused,
)

if muon_momentum_8bit_fused is None:
    pytest.skip("Muon Triton kernels unavailable", allow_module_level=True)

DEVICE = "cuda"

# NF4 quantisation levels (must match _dequantize_nf4 in kernels_muon.py).
_NF4_VALUES = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------------
# Pure-torch references
# ---------------------------------------------------------------------------
def _ref_quant_blockwise_dynamic(m, code, blocksize=256):
    """Reference 8-bit blockwise dynamic quantize. n must be a multiple of blocksize."""
    n = m.numel()
    mb = m.reshape(-1, blocksize)
    absmax = mb.abs().amax(dim=1)
    denom = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    norm = (mb / denom[:, None]).clamp(-1.0, 1.0)
    codes = torch.argmin((norm[..., None] - code).abs(), dim=-1).to(torch.uint8)
    return codes.reshape(n), absmax


def _ref_dequant_blockwise_dynamic(state, absmax, code, blocksize=256):
    vals = code[state.long()].reshape(-1, blocksize)
    return (vals * absmax[:, None]).reshape(-1)


def _ref_quant_nf4(m, values, blocksize=64):
    """Reference NF4 quantize -> (packed uint8, absmax). n multiple of blocksize, even."""
    n = m.numel()
    mb = m.reshape(-1, blocksize)
    absmax = mb.abs().amax(dim=1)
    denom = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    norm = (mb / denom[:, None]).clamp(-1.0, 1.0).reshape(n)
    codes = torch.argmin((norm[:, None] - values).abs(), dim=-1).to(torch.uint8)
    # pack: high nibble = first of pair (even idx), low nibble = second
    packed = (codes[0::2] << 4) | (codes[1::2] & 0xF)
    return packed.to(torch.uint8), absmax


def _ref_dequant_nf4(packed, absmax, values, n, blocksize=64):
    hi = (packed >> 4).long()
    lo = (packed & 0xF).long()
    vals = torch.empty(n, dtype=torch.float32, device=packed.device)
    vals[0::2] = values[hi]
    vals[1::2] = values[lo]
    vals = vals.reshape(-1, blocksize)
    return (vals * absmax[:, None]).reshape(-1)


def _momentum_ref(m, g, beta, nesterov):
    m_new = beta * m + g
    u = (g + beta * m_new) if nesterov else m_new
    return m_new, u


def _assert_mostly_close(a, b, atol, rtol, max_error_count):
    """Allow a few one-level disagreements at exact quantisation midpoints."""
    idx = torch.isclose(a, b, atol=atol, rtol=rtol)
    n_bad = (~idx).sum().item()
    assert n_bad <= max_error_count, f"{n_bad} elements exceed tolerance (allowed {max_error_count})"


# Sizes are multiples of blocksize but NOT of the per-program tile, exercising
# the masking path (8-bit tile = 256*8 = 2048; 4-bit tile = 64*4 = 256).
_SIZES_8BIT = [256 * 3, 256 * 8, 256 * 9]
_SIZES_4BIT = [64 * 3, 64 * 4, 64 * 7]


# ---------------------------------------------------------------------------
# 8-bit fused kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", _SIZES_8BIT, ids=lambda n: f"n{n}")
@pytest.mark.parametrize("nesterov", [True, False], ids=lambda v: f"nesterov{int(v)}")
def test_muon_8bit_fused_matches_reference(n, nesterov):
    torch.manual_seed(0)
    code = F.create_dynamic_map(signed=True).to(DEVICE, dtype=torch.float32)
    m0 = torch.randn(n, device=DEVICE)
    g = torch.randn(n, device=DEVICE) * 0.1

    state, absmax = _ref_quant_blockwise_dynamic(m0, code)
    state_k, absmax_k = state.clone(), absmax.clone()

    # Reference: dequant from the same state -> momentum -> requant
    m_deq = _ref_dequant_blockwise_dynamic(state, absmax, code)
    m_new, u_ref = _momentum_ref(m_deq, g, beta=0.9, nesterov=nesterov)
    state_ref, absmax_ref = _ref_quant_blockwise_dynamic(m_new, code)

    u_out = torch.empty(n, device=DEVICE, dtype=torch.bfloat16)
    muon_momentum_8bit_fused(g, state_k, absmax_k, code, u_out, 0.9, nesterov)

    torch.testing.assert_close(u_out.float(), u_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(absmax_k, absmax_ref, atol=1e-5, rtol=1e-4)
    # Dequantized requantized momentum should match (codes may differ by 1 at
    # exact midpoints; compare the reconstructed values).
    m_k = _ref_dequant_blockwise_dynamic(state_k, absmax_k, code)
    m_r = _ref_dequant_blockwise_dynamic(state_ref, absmax_ref, code)
    _assert_mostly_close(m_k, m_r, atol=1e-2, rtol=1e-2, max_error_count=max(1, n // 200))


# ---------------------------------------------------------------------------
# 4-bit NF4 fused kernel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", _SIZES_4BIT, ids=lambda n: f"n{n}")
@pytest.mark.parametrize("nesterov", [True, False], ids=lambda v: f"nesterov{int(v)}")
def test_muon_4bit_nf4_fused_matches_reference(n, nesterov):
    torch.manual_seed(0)
    values = _NF4_VALUES.to(DEVICE)
    m0 = torch.randn(n, device=DEVICE)
    g = torch.randn(n, device=DEVICE) * 0.1

    packed, absmax = _ref_quant_nf4(m0, values)
    packed_k, absmax_k = packed.clone(), absmax.clone()

    m_deq = _ref_dequant_nf4(packed, absmax, values, n)
    m_new, u_ref = _momentum_ref(m_deq, g, beta=0.9, nesterov=nesterov)
    packed_ref, absmax_ref = _ref_quant_nf4(m_new, values)

    u_out = torch.empty(n, device=DEVICE, dtype=torch.bfloat16)
    muon_momentum_4bit_fused(g, packed_k, absmax_k, u_out, 0.9, nesterov, blocksize=64)

    torch.testing.assert_close(u_out.float(), u_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(absmax_k, absmax_ref, atol=1e-5, rtol=1e-4)
    m_k = _ref_dequant_nf4(packed_k, absmax_k, values, n)
    m_r = _ref_dequant_nf4(packed_ref, absmax_ref, values, n)
    _assert_mostly_close(m_k, m_r, atol=2e-2, rtol=2e-2, max_error_count=max(1, n // 200))


# ---------------------------------------------------------------------------
# NVFP4 fused kernel (reference = the eager helpers in muon.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", _SIZES_4BIT, ids=lambda n: f"n{n}")
@pytest.mark.parametrize("nesterov", [True, False], ids=lambda v: f"nesterov{int(v)}")
def test_muon_nvfp4_fused_matches_eager(n, nesterov):
    torch.manual_seed(0)
    blocksize = 64
    n_paired = (n + 1) // 2
    n_blocks = (n + blocksize - 1) // blocksize
    m0 = torch.randn(n, device=DEVICE)
    g = torch.randn(n, device=DEVICE) * 0.1

    packed = torch.zeros(n_paired, dtype=torch.uint8, device=DEVICE)
    absmax = torch.zeros(n_blocks, dtype=torch.float32, device=DEVICE)
    _nvfp4_quantize_eager(m0, packed, absmax, blocksize)

    packed_k, absmax_k = packed.clone(), absmax.clone()

    # eager reference
    m_deq = _nvfp4_dequantize_eager(packed, absmax, n, blocksize, DEVICE)
    m_new, u_ref = _momentum_ref(m_deq, g, beta=0.9, nesterov=nesterov)
    packed_ref, absmax_ref = packed.clone(), absmax.clone()
    _nvfp4_quantize_eager(m_new, packed_ref, absmax_ref, blocksize)

    u_out = torch.empty(n, device=DEVICE, dtype=torch.bfloat16)
    muon_momentum_nvfp4_fused(g, packed_k, absmax_k, u_out, 0.9, nesterov, blocksize=blocksize)

    torch.testing.assert_close(u_out.float(), u_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(absmax_k, absmax_ref, atol=1e-5, rtol=1e-4)
    m_k = _nvfp4_dequantize_eager(packed_k, absmax_k, n, blocksize, DEVICE)
    m_r = _nvfp4_dequantize_eager(packed_ref, absmax_ref, n, blocksize, DEVICE)
    _assert_mostly_close(m_k, m_r, atol=2e-2, rtol=2e-2, max_error_count=max(1, n // 200))


# ---------------------------------------------------------------------------
# NVFP4 eager codec roundtrip (pure torch, CPU-friendly)
# ---------------------------------------------------------------------------
# Includes sizes that are NOT a multiple of blocksize (partial final block) to
# exercise the eager helper's padding path.
@pytest.mark.parametrize("n", [64, 65, 128, 192, 200, 10000], ids=lambda n: f"n{n}")
def test_nvfp4_eager_roundtrip(n):
    torch.manual_seed(0)
    blocksize = 64
    n_paired = (n + 1) // 2
    n_blocks = (n + blocksize - 1) // blocksize
    m = torch.randn(n)

    packed = torch.zeros(n_paired, dtype=torch.uint8)
    absmax = torch.zeros(n_blocks, dtype=torch.float32)
    _nvfp4_quantize_eager(m, packed, absmax, blocksize)
    recon = _nvfp4_dequantize_eager(packed, absmax, n, blocksize, torch.device("cpu"))

    # per-block absmax matches exactly (last block may be partial -> pad).
    pad = n_blocks * blocksize - n
    m_padded = m if pad == 0 else torch.cat([m, torch.zeros(pad)])
    block_absmax = m_padded.reshape(n_blocks, blocksize).abs().amax(dim=1)
    torch.testing.assert_close(absmax, block_absmax, atol=1e-6, rtol=1e-5)
    # NVFP4 has 7 magnitude levels; per-element error is bounded by ~half the
    # largest grid gap (1/6 of absmax).
    assert math.isfinite(recon.float().mean().item())
    assert (recon - m).abs().max() <= (absmax.max().item() * (1.0 / 6.0) + 1e-4)
