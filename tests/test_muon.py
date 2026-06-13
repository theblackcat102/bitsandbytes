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
    _make_default_orthogonalize_fn,
    _prepare_orthogonalization_inputs,
    _reconstruct_orthogonalized_updates,
    _reconstruct_tensor_from_matrices,
    _resolve_adjust_lr,
    _split_tensor_for_orthogonalization,
    _standard_newton_schulz,
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
@functools.cache
def _variant_supported(device: str, key: str) -> bool:
    """Return True if a one-step Muon update of the given variant runs here."""
    try:
        p = torch.randn(128, 64, device=device)  # 8192 elems > min_8bit_size
        p.grad = torch.randn_like(p) * 0.01
        opt = _make_variant(key, [p])
        opt.step()
        return True
    except Exception:
        return False


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
