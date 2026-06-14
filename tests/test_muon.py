"""Tests for the Muon optimizer (bitsandbytes.optim.muon).

Muon has no PyTorch builtin reference (like AdEMAMix), so the quantized
variants are validated against the pure-fp32 ``Muon32bit`` path, and the
fp32 math is validated against a small in-test reference step.

Newton-Schulz selection is machine dependent (gram_newton_schulz +
CuTeDSL/Triton on capable GPUs, pure-torch fallback otherwise). To keep
trajectory comparisons deterministic and backend independent, every test
that compares updates injects ``orthogonalize_fn=_standard_newton_schulz``
into *both* sides.
"""

import functools
import io

import pytest
import torch

from bitsandbytes.optim.muon import (
    Muon,
    Muon4bit,
    Muon4bitNVFP4,
    Muon8bit,
    Muon32bit,
    _adjust_lr_rms_norm,
    _adjust_lr_spectral_norm,
    _create_param_batches,
    _default_orthogonalize_fn,
    _make_default_orthogonalize_fn,
    _make_sm100_nvfp4_ns_fn,
    _prepare_orthogonalization_inputs,
    _reconstruct_orthogonalized_updates,
    _reconstruct_tensor_from_matrices,
    _resolve_adjust_lr,
    _split_tensor_for_orthogonalization,
    _standard_newton_schulz,
    _to_nvfp4_with_scales,
    _validate_param_split_fn,
    newton_schulz,
)
from tests.helpers import get_available_devices, id_formatter


def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if error_count > max_error_count:
        print(f"Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Capability detection (per device/variant). The CUDA C++ kernels may be
# unbuilt or a device may be missing; detect at runtime and skip cleanly.
# ---------------------------------------------------------------------------

# Substrings (matched case-insensitively) that indicate a genuine hardware or
# software capability gap rather than a code bug.  Only RuntimeErrors whose
# message contains one of these patterns are treated as "not supported here".
_SKIP_PATTERNS: tuple[str, ...] = (
    "no cuda gpus are available",
    "cuda is not available",
    "not compiled with cuda",
    "no kernel image is available",  # SM-architecture mismatch
    "device capability",
)


def _is_capability_error(exc: Exception) -> bool:
    """Return True if *exc* signals a hardware/software gap, not a code bug.

    Recognised as capability gaps:
    - ``ImportError`` / ``AttributeError``: optional package or torch op absent
      (Triton, gram-newton-schulz, quack, torch._scaled_grouped_mm_v2, …).
    - ``RuntimeError`` whose message matches a known CUDA availability or
      SM-capability pattern.

    Everything else (``ValueError``, ``TypeError``, ``AssertionError``, …) is
    considered a code bug and must not be silenced.
    """
    if isinstance(exc, (ImportError, AttributeError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(pat in msg for pat in _SKIP_PATTERNS)
    return False


@functools.cache
def _variant_supported(device: str, key: str) -> bool:
    """Return True if a one-step Muon update of the given variant runs here.

    Capability gaps (no CUDA, missing optional package, SM mismatch) cause the
    function to return False so the caller can skip cleanly.  Any other
    exception is re-raised so that implementation bugs surface as test failures
    rather than silent skips.
    """
    try:
        p = torch.randn(128, 64, device=device)  # 8192 elems > min_8bit_size
        p.grad = torch.randn_like(p) * 0.01
        opt = _make_variant(key, [p])
        opt.step()
        return True
    except Exception as exc:
        if _is_capability_error(exc):
            return False
        raise


def _make_variant(key: str, params, **kw):
    kw.setdefault("orthogonalize_fn", _standard_newton_schulz)
    if key == "muon32bit":
        return Muon32bit(params, **kw)
    if key == "muon8bit":
        return Muon8bit(params, **kw)
    if key == "muon4bit_nf4":
        return Muon4bit(params, quant_type="nf4", **kw)
    if key == "muon4bit_fp4":
        return Muon4bit(params, quant_type="fp4", **kw)
    if key == "muon4bit_nvfp4":
        return Muon4bitNVFP4(params, **kw)
    raise KeyError(key)


QUANT_VARIANTS = ["muon8bit", "muon4bit_nf4", "muon4bit_fp4", "muon4bit_nvfp4"]


# ===========================================================================
# 1. Constructor validation (pure, no device needed)
# ===========================================================================
class TestConstructorValidation:
    def _p(self):
        return [torch.randn(8, 8)]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": -1.0},
            {"momentum": 1.0},
            {"momentum": -0.1},
            {"weight_decay": -0.5},
            {"ns_chunk_size": 0},
            {"optim_bits": 16},
            {"quant_type": "bogus"},
        ],
    )
    def test_invalid_kwargs_raise(self, kwargs):
        with pytest.raises(ValueError):
            Muon(self._p(), **kwargs)

    def test_one_dim_param_rejected(self):
        with pytest.raises(ValueError, match="2 or more dimensions"):
            Muon([torch.randn(16)])

    def test_valid_construction_sets_metadata(self):
        opt = Muon4bit(self._p(), quant_type="nf4")
        assert opt.optimizer_name == "muon"
        assert opt._quant_type == "nf4"


# ===========================================================================
# 2. Learning-rate adjustment helpers (pure)
# ===========================================================================
class TestLRAdjust:
    def test_rms_norm_formula(self):
        import math

        assert _adjust_lr_rms_norm(0.1, (256, 64)) == pytest.approx(0.1 * 0.2 * math.sqrt(256))

    def test_spectral_norm_formula(self):
        import math

        assert _adjust_lr_spectral_norm(0.1, (256, 64)) == pytest.approx(0.1 * math.sqrt(256 / 64))

    def test_resolve_none_is_identity(self):
        fn = _resolve_adjust_lr(None)
        assert fn(0.3, (10, 20)) == 0.3

    def test_resolve_known_strings(self):
        assert _resolve_adjust_lr("rms_norm")(0.1, (256, 64)) == _adjust_lr_rms_norm(0.1, (256, 64))
        assert _resolve_adjust_lr("spectral_norm")(0.1, (256, 64)) == _adjust_lr_spectral_norm(0.1, (256, 64))

    def test_resolve_callable_passthrough(self):
        fn = lambda lr, shape: lr * 2.0
        assert _resolve_adjust_lr(fn)(0.5, (4, 4)) == 1.0

    def test_resolve_bad_string_raises(self):
        with pytest.raises(ValueError):
            _resolve_adjust_lr("not_a_strategy")

    def test_resolve_bad_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_adjust_lr(123)


# ===========================================================================
# 3. Newton-Schulz orthogonalization (in-repo _standard_newton_schulz)
# ===========================================================================
class TestNewtonSchulz:
    @pytest.mark.parametrize("shape", [(64, 128), (128, 64), (3, 32, 48), (2, 4, 16, 24)])
    def test_shape_and_dtype_preserved(self, shape):
        x = torch.randn(*shape)
        out = _standard_newton_schulz(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_wide_matrix_is_semi_orthogonal(self):
        x = torch.randn(64, 128)
        o = _standard_newton_schulz(x).float()
        gram = o @ o.mT  # (64, 64) should approx I
        eye = torch.eye(64)
        assert (gram - eye).abs().max() < 0.15

    def test_tall_matrix_is_semi_orthogonal(self):
        x = torch.randn(128, 64)
        o = _standard_newton_schulz(x).float()
        gram = o.mT @ o  # (64, 64) should approx I
        eye = torch.eye(64)
        assert (gram - eye).abs().max() < 0.15

    def test_batched_orthogonality(self):
        x = torch.randn(4, 32, 64)
        o = _standard_newton_schulz(x).float()
        for i in range(4):
            gram = o[i] @ o[i].mT
            assert (gram - torch.eye(32)).abs().max() < 0.2

    def test_zero_input_no_nan(self):
        out = _standard_newton_schulz(torch.zeros(16, 32))
        assert torch.isfinite(out).all()

    def test_public_helper_runs(self):
        out = newton_schulz(torch.randn(32, 32))
        assert out.shape == (32, 32)
        assert torch.isfinite(out.float()).all()


# ===========================================================================
# 4. Parameter batching + split/reconstruct machinery (pure)
# ===========================================================================
class TestBatchingAndSplit:
    def test_create_param_batches_groups_by_shape(self):
        a, b, c = torch.randn(8, 8), torch.randn(8, 8), torch.randn(4, 4)
        batches = _create_param_batches([a, b, c])
        sizes = sorted(len(g) for g in batches)
        assert sizes == [1, 2]

    def test_create_param_batches_sorted_by_ptr(self):
        ps = [torch.randn(8, 8) for _ in range(3)]
        batch = _create_param_batches(ps)[0]
        ptrs = [p.data_ptr() for p in batch]
        assert ptrs == sorted(ptrs)

    def test_split_reconstruct_matrix(self):
        x = torch.randn(16, 32)
        mats, spec = _split_tensor_for_orthogonalization(x)
        assert spec[0] == "matrix" and len(mats) == 1
        torch.testing.assert_close(_reconstruct_tensor_from_matrices(spec, mats), x)

    def test_split_reconstruct_batch3d(self):
        x = torch.randn(4, 16, 32)
        mats, spec = _split_tensor_for_orthogonalization(x)
        assert spec[0] == "batch3d" and len(mats) == 4
        torch.testing.assert_close(_reconstruct_tensor_from_matrices(spec, mats), x)

    def test_split_reconstruct_flatten(self):
        x = torch.randn(4, 3, 8, 8)
        mats, spec = _split_tensor_for_orthogonalization(x)
        assert spec[0] == "flatten" and mats[0].ndim == 2
        torch.testing.assert_close(_reconstruct_tensor_from_matrices(spec, mats), x)

    def test_validate_split_fn_errors(self):
        x = torch.randn(8, 8)
        with pytest.raises(ValueError):
            _validate_param_split_fn(lambda t: [], x, [])
        with pytest.raises(TypeError):
            _validate_param_split_fn(lambda t: ["nope"], x, ["nope"])
        with pytest.raises(ValueError):  # ndim changed
            _validate_param_split_fn(lambda t: [t.reshape(-1)], x, [x.reshape(-1)])

    def test_prepare_reconstruct_roundtrip_with_custom_split(self):
        x = torch.randn(8, 16)
        split_fn = lambda t: [t[:, :8], t[:, 8:]]
        recombine_fn = lambda parts: torch.cat(parts, dim=1)
        by_shape, specs = _prepare_orthogonalization_inputs([x], split_fn)
        # identity "orthogonalization"
        ortho = {shape: torch.stack(mats) for shape, mats in by_shape.items()}
        updates = _reconstruct_orthogonalized_updates(ortho, specs, recombine_fn)
        torch.testing.assert_close(updates[0], x)


# ===========================================================================
# 5. State initialization (device-parametrized)
# ===========================================================================
@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
class TestStateInit:
    def test_fp32_state(self, device):
        p = torch.randn(128, 64, device=device)
        p.grad = torch.randn_like(p) * 0.01
        opt = Muon32bit([p], orthogonalize_fn=_standard_newton_schulz)
        opt.step()
        assert opt.state[p]["state1"].dtype == torch.float32
        assert opt.state[p]["state1"].shape == p.shape

    def test_small_param_falls_back_to_fp32(self, device):
        # numel = 1024 < min_8bit_size (4096) -> fp32 even for Muon8bit
        p = torch.randn(32, 32, device=device)
        p.grad = torch.randn_like(p) * 0.01
        opt = Muon8bit([p], orthogonalize_fn=_standard_newton_schulz)
        opt.step()
        assert opt.state[p]["state1"].dtype == torch.float32

    def test_8bit_state_layout(self, device):
        if not _variant_supported(device, "muon8bit"):
            pytest.skip("8-bit Muon not runnable in this environment")
        p = torch.randn(128, 64, device=device)
        p.grad = torch.randn_like(p) * 0.01
        opt = Muon8bit([p], orthogonalize_fn=_standard_newton_schulz)
        opt.step()
        st = opt.state[p]
        assert st["state1"].dtype == torch.uint8
        assert "qmap1" in st
        assert st["absmax1"].numel() == (p.numel() + 255) // 256

    def test_nvfp4_state_layout(self, device):
        if not _variant_supported(device, "muon4bit_nvfp4"):
            pytest.skip("NVFP4 Muon not runnable in this environment")
        p = torch.randn(128, 64, device=device)
        p.grad = torch.randn_like(p) * 0.01
        opt = Muon4bitNVFP4([p], orthogonalize_fn=_standard_newton_schulz)
        opt.step()
        st = opt.state[p]
        assert st["state1"].dtype == torch.uint8
        assert st["state1"].numel() == (p.numel() + 1) // 2
        assert st["absmax1"].numel() == (p.numel() + 63) // 64


# ===========================================================================
# 6. Muon32bit numeric trajectory vs an explicit reference step
# ===========================================================================
class _RefMuon:
    """Minimal pure-fp32 Muon reference (single group, 2-D params)."""

    def __init__(self, params, lr, momentum, weight_decay, nesterov, ortho, adjust):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.nesterov, self.ortho, self.adjust = nesterov, ortho, adjust
        self.m = {}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            m = self.m.setdefault(p, torch.zeros_like(p))
            m.mul_(self.momentum).add_(g)
            u = (g + self.momentum * m) if self.nesterov else m
            o = self.ortho(u.unsqueeze(0)).squeeze(0)
            alr = self.adjust(self.lr, p.shape)
            p.mul_(1.0 - self.lr * self.wd).add_(o, alpha=-alr)


@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
@pytest.mark.parametrize("nesterov", [True, False], ids=id_formatter("nesterov"))
@pytest.mark.parametrize("adjust_lr", ["rms_norm", None], ids=id_formatter("adjust"))
def test_muon32bit_matches_reference(device, nesterov, adjust_lr):
    torch.manual_seed(0)
    p_ref = torch.randn(64, 48, device=device)
    p_bnb = p_ref.clone()

    opt = Muon32bit(
        [p_bnb],
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.1,
        nesterov=nesterov,
        adjust_lr=adjust_lr,
        orthogonalize_fn=_standard_newton_schulz,
    )
    ref = _RefMuon(
        [p_ref],
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.1,
        nesterov=nesterov,
        ortho=_standard_newton_schulz,
        adjust=_resolve_adjust_lr(adjust_lr),
    )

    for _ in range(8):
        g = torch.randn(64, 48, device=device) * 0.05
        p_ref.grad = g.clone()
        p_bnb.grad = g.clone()
        ref.step()
        opt.step()
        torch.testing.assert_close(opt.state[p_bnb]["state1"], ref.m[p_ref], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(p_bnb, p_ref, atol=1e-4, rtol=1e-3)


# ===========================================================================
# 7. Quantized variants stay close to fp32 Muon (params)
# ===========================================================================
@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
@pytest.mark.parametrize("variant", QUANT_VARIANTS, ids=id_formatter("variant"))
def test_quantized_close_to_fp32(device, variant):
    if not _variant_supported(device, variant):
        pytest.skip(f"{variant} not runnable in this environment")

    torch.manual_seed(0)
    p32 = torch.randn(128, 96, device=device)
    pq = p32.clone()

    o32 = Muon32bit([p32], lr=5e-3, momentum=0.9, weight_decay=0.0, orthogonalize_fn=_standard_newton_schulz)
    oq = _make_variant(variant, [pq], lr=5e-3, momentum=0.9, weight_decay=0.0)

    for _ in range(10):
        g = torch.randn(128, 96, device=device) * 0.05
        p32.grad = g.clone()
        pq.grad = g.clone()
        o32.step()
        oq.step()
        # quantized momentum introduces small per-step deviation; bound it and
        # reset params so the comparison measures one-step divergence.
        assert_most_approx_close(pq.float(), p32.float(), atol=2e-2, rtol=2e-2, max_error_count=p32.numel() // 20)
        pq.data.copy_(p32.data)


# ===========================================================================
# 8. End-to-end training reduces loss without NaN
# ===========================================================================
@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
def test_training_reduces_loss(device):
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(64, 64, bias=False),
        nn.Linear(64, 32, bias=False),
    ).to(device)
    opt = Muon32bit(model.parameters(), lr=2e-2, orthogonalize_fn=_standard_newton_schulz)

    x = torch.randn(32, 64, device=device)
    target = torch.randn(32, 32, device=device)

    losses = []
    for _ in range(30):
        out = model(x)
        loss = ((out - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
        losses.append(loss.item())

    assert losses[-1] < losses[0]


# ===========================================================================
# 9. state_dict save / load roundtrip
# ===========================================================================
@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
@pytest.mark.parametrize("variant", ["muon32bit", "muon8bit", "muon4bit_nvfp4"], ids=id_formatter("variant"))
def test_state_dict_roundtrip(device, variant):
    if not _variant_supported(device, variant):
        pytest.skip(f"{variant} not runnable in this environment")

    torch.manual_seed(0)
    p = torch.randn(128, 64, device=device)
    opt = _make_variant(variant, [p], lr=1e-2)

    for _ in range(5):
        p.grad = torch.randn_like(p) * 0.05
        opt.step()

    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)

    p2 = p.clone()
    opt2 = _make_variant(variant, [p2], lr=1e-2)
    buf.seek(0)
    opt2.load_state_dict(torch.load(buf))

    for k, v in opt.state[p].items():
        if isinstance(v, torch.Tensor):
            v2 = opt2.state[p2][k]
            assert v.shape == v2.shape and v.dtype == v2.dtype
            torch.testing.assert_close(v, v2)

    # resume training, no NaN
    for _ in range(3):
        p2.grad = torch.randn_like(p2) * 0.05
        opt2.step()
        assert torch.isfinite(p2).all()


# ===========================================================================
# 10. ns_chunk_size does not change the result
# ===========================================================================
@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
def test_ns_chunk_size_invariance(device):
    torch.manual_seed(0)
    shape = (48, 48)
    ps1 = [torch.randn(*shape, device=device) for _ in range(4)]
    ps2 = [p.clone() for p in ps1]
    grads = [torch.randn(*shape, device=device) * 0.05 for _ in range(4)]

    o1 = Muon32bit(ps1, lr=1e-2, ns_chunk_size=1, orthogonalize_fn=_standard_newton_schulz)
    o2 = Muon32bit(ps2, lr=1e-2, ns_chunk_size=8, orthogonalize_fn=_standard_newton_schulz)

    for p, g in zip(ps1, grads):
        p.grad = g.clone()
    for p, g in zip(ps2, grads):
        p.grad = g.clone()
    o1.step()
    o2.step()

    for a, b in zip(ps1, ps2):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


# ===========================================================================
# 11. Default orthogonalize_fn uses the sped-up gram_newton_schulz when present
# ===========================================================================
def test_default_uses_gram_when_available():
    pytest.importorskip("gram_newton_schulz")
    from gram_newton_schulz import GramNewtonSchulz

    fn = _make_default_orthogonalize_fn()
    # GramNewtonSchulz returns its bound __call__; the pure-torch fallback is a
    # module-level function named _default_orthogonalize_fn.
    assert getattr(fn, "__self__", None).__class__ is GramNewtonSchulz

    opt = Muon([torch.randn(8, 8)])  # no explicit orthogonalize_fn
    assert getattr(opt._orthogonalize_fn, "__self__", None).__class__ is GramNewtonSchulz


# ===========================================================================
# 12. SM100 FP4 tensor-core Gram Newton-Schulz path
# ===========================================================================
# These tests verify that the _make_sm100_nvfp4_ns_fn() path — which uses the
# private torch._scaled_grouped_mm_v2 API with float4_e2m1fn_x2 / float8_e4m3fn
# tensors for the X @ X.T Gram GEMM — numerically matches _standard_newton_schulz
# and produces near-orthogonal output.
#
# All tests are automatically skipped when:
#   • CUDA is unavailable, or
#   • the GPU is not data-center Blackwell (sm100: B200=(10,0) / B300=(10,3)), or
#   • torch._scaled_grouped_mm_v2 / float4_e2m1fn_x2 / ScalingType are absent.
#
# The gating in Muon4bitNVFP4.__init__ (muon.py:1718) enables
# _make_sm100_nvfp4_ns_fn only on those two capabilities; the tests here
# exercise that exact code path.
# ---------------------------------------------------------------------------


def _get_sm100_nvfp4_ns_fn():
    """Return the FP4 NS closure, or call pytest.skip.

    Performs every prerequisite check in the same order as
    _make_sm100_nvfp4_ns_fn itself so that the test accurately reflects what
    the production code will do on a real SM100 node.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cap = torch.cuda.get_device_capability()
    if cap not in ((10, 0), (10, 3)):
        pytest.skip(
            f"SM100 FP4 tensor-core path requires device capability (10,0) or "
            f"(10,3) (B200/B300); current GPU reports {cap}"
        )

    # Check for the private torch FP4 GEMM API
    try:
        from torch.nn.functional import ScalingType, SwizzleType  # noqa: F401

        _ = torch._scaled_grouped_mm_v2
        _ = torch.float4_e2m1fn_x2
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"torch FP4 GEMM API unavailable in this PyTorch build: {exc}")

    fn = _make_sm100_nvfp4_ns_fn()
    if fn is _default_orthogonalize_fn:
        # _make_sm100_nvfp4_ns_fn fell back to BF16 despite the checks above
        pytest.skip(
            "_make_sm100_nvfp4_ns_fn returned the BF16 fallback unexpectedly; "
            "FP4 GEMM import must have failed inside the factory"
        )
    return fn


class TestSM100NVFP4GramNewtonSchulz:
    """Numerical correctness tests for the SM100 FP4 tensor-core NS path.

    Structure
    ---------
    1. test_fp4_gram_close_to_bf16_gram
       Directly exercises _to_nvfp4_with_scales + torch._scaled_grouped_mm_v2
       and verifies the resulting Gram matrix is within FP4 quantization noise
       of the bf16 reference X @ X.T.

    2. test_nvfp4_ns_matches_standard_ns  (parametrised over shapes)
       Compares the full _nvfp4_ns output against _standard_newton_schulz for
       shapes where N % 16 == 0 (FP4 Gram path is active).  Expected tolerance
       reflects five iterations of accumulated FP4 quantization noise.

    3. test_nvfp4_ns_n_not_multiple_of_16_falls_back
       For N % 16 != 0 the _fp4_gram inner function falls back to a pure-BF16
       X @ X.T.  The full NS outputs must then be numerically identical to
       _standard_newton_schulz (within BF16 rounding).

    4. test_nvfp4_ns_output_near_orthogonal  (parametrised)
       Orthogonality sanity check: (X_out @ X_out.T) / scale ≈ I.

    5. test_muon4bitnvfp4_uses_fp4_ns_on_sm100
       Optimizer-level smoke test: Muon4bitNVFP4 with orthogonalize_fn=None
       (auto-select) completes a step without error and produces finite params.

    6. test_muon4bitnvfp4_fp4_ns_trajectory_close_to_bf16
       10-step trajectory: Muon4bitNVFP4 (FP4 NS, FP4 momentum) stays within
       atol=0.05 of Muon32bit (_standard_newton_schulz) per step.
    """

    DEVICE = "cuda"

    @pytest.fixture(autouse=True)
    def _setup(self):
        # Skip the whole class if prerequisites are not met; store the FP4 NS
        # function so individual tests can call it via self._nvfp4_ns.
        self._nvfp4_ns = _get_sm100_nvfp4_ns_fn()

    # ------------------------------------------------------------------
    # 1. FP4 Gram ≈ BF16 Gram
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "shape",
        [(64, 64), (32, 128), (48, 96), (64, 128)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_fp4_gram_close_to_bf16_gram(self, shape):
        """X @ X.T via FP4 GEMM should be within FP4 quantization noise of bf16.

        FP4 has 4-bit precision with 1x16 block-wise FP8 scales.  Per-element
        quantization error is bounded by (absmax / 7) / 2 per block.  After
        accumulating over N columns the Gram-matrix element error grows as
        ~sqrt(N) x (per-element error), which for the shapes tested here stays
        well below atol=0.05.

        This test exercises _to_nvfp4_with_scales and torch._scaled_grouped_mm_v2
        directly, exactly as _fp4_gram does inside _make_sm100_nvfp4_ns_fn.
        """
        from torch.nn.functional import ScalingType, SwizzleType

        m, n = shape
        assert n % 16 == 0, "test requires N divisible by 16 to exercise the FP4 path"

        torch.manual_seed(7)
        # Frobenius-normalize as _nvfp4_ns would before computing the Gram
        X = torch.randn(m, n, dtype=torch.bfloat16, device=self.DEVICE)
        X = (X / X.norm()).contiguous()

        # BF16 reference
        G_bf16 = (X @ X.mT).float()

        # FP4 path
        X_fp4, scale_X = _to_nvfp4_with_scales(X, block_size=16)
        X_T_fp4 = X_fp4.t().contiguous()  # (n//2, m)
        scale_X_T = scale_X.t().contiguous()  # (n//16, m)
        G_fp4 = torch._scaled_grouped_mm_v2(
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
        ).float()

        torch.testing.assert_close(
            G_fp4,
            G_bf16,
            atol=0.05,
            rtol=0.1,
            msg=(
                f"FP4 Gram matrix deviates from BF16 reference beyond expected "
                f"quantization noise for shape {shape}. "
                "This likely indicates a bug in _to_nvfp4_with_scales packing or "
                "an incorrect scale/swizzle configuration in the _sgmm call."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Full NS: FP4 path ≈ BF16 reference (N % 16 == 0)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "shape",
        [
            (64, 64),  # square, FP4 Gram active
            (48, 96),  # wide (m < n), FP4 Gram active
            (96, 48),  # tall (m > n), transposed internally → FP4 Gram on (48, 96)
            (32, 128),  # wide, larger N
        ],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_nvfp4_ns_matches_standard_ns(self, shape):
        """_nvfp4_ns should match _standard_newton_schulz within FP4 noise budget.

        Both paths Frobenius-normalize X, run 5 NS iterations with the same
        POLAR_EXPRESS_COEFFICIENTS, and return the near-orthogonal result.  The
        FP4 path replaces only the X @ X.T Gram GEMM; all other arithmetic
        (polynomial evaluation, X update, transpose bookkeeping) stays in BF16.

        Five iterations of accumulated FP4 quantization noise → atol=0.1 is a
        generous but finite bound.  A value much larger than this would indicate
        the FP4 Gram is a poor approximation that destabilises the NS recurrence.
        """
        torch.manual_seed(13)
        X = torch.randn(*shape, dtype=torch.bfloat16, device=self.DEVICE)

        out_fp4 = self._nvfp4_ns(X.clone()).float()
        out_bf16 = _standard_newton_schulz(X.clone()).float()

        torch.testing.assert_close(
            out_fp4,
            out_bf16,
            atol=0.1,
            rtol=0.2,
            msg=(
                f"SM100 FP4 NS output deviates from BF16 NS by more than the "
                f"expected FP4 quantization noise budget for shape {shape}. "
                "Check _fp4_gram packing or the NS iteration loop."
            ),
        )

    # ------------------------------------------------------------------
    # 3. N not divisible by 16 → BF16 fallback inside _fp4_gram
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "shape",
        [(64, 48), (32, 60)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_nvfp4_ns_n_not_multiple_of_16_falls_back(self, shape):
        """When N % 16 != 0, _fp4_gram uses bf16 X @ X.T and outputs must match.

        The _fp4_gram function guards with ``if n % 16 != 0: return X @ X.mT``.
        In that branch the full _nvfp4_ns is arithmetically identical to
        _standard_newton_schulz (same BF16 ops, same coefficients), so the
        outputs should agree up to floating-point rounding (atol=1e-4).

        This test confirms the fallback guard is reached and does not silently
        use FP4 on unsupported dimensions, which would produce wrong results.
        """
        _, n = shape
        assert n % 16 != 0, "test is only meaningful when N is not divisible by 16"

        torch.manual_seed(99)
        X = torch.randn(*shape, dtype=torch.bfloat16, device=self.DEVICE)

        out_fp4 = self._nvfp4_ns(X.clone()).float()
        out_bf16 = _standard_newton_schulz(X.clone()).float()

        torch.testing.assert_close(
            out_fp4,
            out_bf16,
            atol=1e-4,
            rtol=1e-3,
            msg=(
                f"BF16 fallback path in _nvfp4_ns should be numerically identical "
                f"to _standard_newton_schulz for shape {shape} (N % 16 != 0). "
                "The fallback guard may be broken or the wrong code path was taken."
            ),
        )

    # ------------------------------------------------------------------
    # 4. Output is near-orthogonal
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "shape",
        [(64, 64), (48, 96), (96, 48)],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_nvfp4_ns_output_near_orthogonal(self, shape):
        """FP4 NS output X satisfies X @ X.T ≈ (trace/k) * I (near-orthogonal).

        Newton-Schulz orthogonalizes the input; the output should satisfy
        X X^T ≈ scaled identity for m ≤ n, or X^T X ≈ scaled identity for m > n.
        We measure the normalized off-diagonal Frobenius norm of the Gram matrix
        and require it to be below 0.1 (10 % of the diagonal magnitude).

        A failure here means FP4 quantization noise is large enough to prevent
        the NS recurrence from converging to a near-orthogonal matrix, which
        would make the optimizer update step numerically meaningless.
        """
        torch.manual_seed(17)
        X = torch.randn(*shape, dtype=torch.bfloat16, device=self.DEVICE)
        out = self._nvfp4_ns(X).float()

        m, n = shape
        if m <= n:
            gram = out @ out.T  # (m, m)
            k = m
        else:
            gram = out.T @ out  # (n, n)
            k = n

        trace_per_dim = gram.trace() / k
        eye_approx = torch.eye(k, device=self.DEVICE, dtype=torch.float32) * trace_per_dim
        off_diag_norm = (gram - eye_approx).norm()
        normalized_error = off_diag_norm / (k * trace_per_dim.abs().clamp(min=1e-6))

        assert normalized_error < 0.1, (
            f"FP4 NS output not near-orthogonal for shape {shape}: "
            f"normalized off-diagonal Frobenius norm = {normalized_error:.4f} > 0.1. "
            "FP4 quantization noise may be too large for the NS recurrence to converge."
        )

    # ------------------------------------------------------------------
    # 5. Batched (B, m, n) input handled correctly
    # ------------------------------------------------------------------
    def test_nvfp4_ns_batched_input(self):
        """_nvfp4_ns handles (B, m, n) input and matches _standard_newton_schulz."""
        torch.manual_seed(31)
        X = torch.randn(3, 48, 64, dtype=torch.bfloat16, device=self.DEVICE)

        out_fp4 = self._nvfp4_ns(X.clone()).float()
        out_bf16 = _standard_newton_schulz(X.clone()).float()

        torch.testing.assert_close(out_fp4, out_bf16, atol=0.1, rtol=0.2)

    # ------------------------------------------------------------------
    # 6. Optimizer smoke test: Muon4bitNVFP4 runs without error on SM100
    # ------------------------------------------------------------------
    def test_muon4bitnvfp4_uses_fp4_ns_on_sm100(self):
        """Muon4bitNVFP4 (orthogonalize_fn=None) should use the FP4 NS path on SM100.

        When orthogonalize_fn is not supplied, Muon4bitNVFP4.__init__ detects
        SM100 capability and calls _make_sm100_nvfp4_ns_fn() (muon.py:1718-1719).
        We verify the step completes without error and produces finite parameters.

        A failure here (non-finite output, RuntimeError, etc.) indicates the
        FP4 GEMM API is broken for the shapes that arise in a real optimizer step.
        """
        torch.manual_seed(42)
        p = torch.randn(96, 64, device=self.DEVICE)
        p.grad = torch.randn_like(p) * 0.01

        opt = Muon4bitNVFP4([p])  # orthogonalize_fn=None → auto-selects FP4 NS
        opt.step()

        assert torch.isfinite(p).all(), (
            "Muon4bitNVFP4 step produced non-finite parameter values on SM100. "
            "The FP4 tensor-core NS path or the FP4 momentum update kernel has "
            "a numerical stability problem."
        )

    # ------------------------------------------------------------------
    # 7. 10-step trajectory: FP4 NS stays close to BF16 Muon32bit
    # ------------------------------------------------------------------
    def test_muon4bitnvfp4_fp4_ns_trajectory_close_to_bf16(self):
        """On SM100, Muon4bitNVFP4 (FP4 NS + FP4 momentum) stays near Muon32bit.

        Both optimisers share the same hyperparameters.  Muon32bit uses
        _standard_newton_schulz; Muon4bitNVFP4 auto-selects _nvfp4_ns on SM100.
        We compare parameter values after each of 10 steps, resetting the
        quantised params to the fp32 reference after each step so we measure
        per-step divergence rather than accumulated drift.

        atol=0.05 is intentionally generous: it catches catastrophic failures
        (wrong sign, NaN, large truncation error in the FP4 path) while
        permitting the expected ~1-2 % per-step quantisation error from both the
        FP4 NS and the FP4 momentum buffer.
        """
        torch.manual_seed(0)
        p32 = torch.randn(96, 64, device=self.DEVICE)
        pfp4 = p32.clone()

        o32 = Muon32bit(
            [p32],
            lr=5e-3,
            momentum=0.9,
            weight_decay=0.0,
            orthogonalize_fn=_standard_newton_schulz,
        )
        # orthogonalize_fn=None → SM100 auto-selects _nvfp4_ns
        ofp4 = Muon4bitNVFP4([pfp4], lr=5e-3, momentum=0.9, weight_decay=0.0)

        for step in range(10):
            g = torch.randn(96, 64, device=self.DEVICE) * 0.05
            p32.grad = g.clone()
            pfp4.grad = g.clone()
            o32.step()
            ofp4.step()

            assert_most_approx_close(
                pfp4.float(),
                p32.float(),
                atol=0.05,
                rtol=0.05,
                max_error_count=p32.numel() // 10,
            )
            # Reset quantised params so divergence is measured per-step
            pfp4.data.copy_(p32.data)
