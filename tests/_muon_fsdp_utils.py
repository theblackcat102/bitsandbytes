"""Shared test scaffolding for the Muon FSDP2 equivalence / checkpoint tests.

This is the **Layer C** seam frozen in ``docs/muon_distributed_design.md`` §11.C.
The implementer provides nothing here; the test author owns it, but the
signatures (``make_tiny_model`` / ``make_batch`` / ``run_reference`` /
``run_fsdp2`` / ``MUON_FSDP_TOL`` / ``MUON_FSDP_CHECKPOINT_TOL``) are fixed so
the equivalence test (``test_muon_fsdp.py``) and the multi-rank worker
(``fsdp_muon_worker.py``) build on the same primitives.

The module is import-safe without ``torch.distributed`` initialised: only
``run_fsdp2`` requires a live process group (it is called from the torchrun
worker), everything else runs in a plain single process.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bitsandbytes.optim.muon import (
    Muon4bit,
    Muon8bit,
    Muon32bit,
    _standard_newton_schulz,
)

# Backend-independent NS so the single-rank oracle and the FSDP run agree
# regardless of which Newton-Schulz kernel each environment selects (see the
# module docstring of tests/test_muon.py).
_FORCE_NS = {"orthogonalize_fn": _standard_newton_schulz}

# Dimensions kept small but >= min_8bit_size (4096) so the 8/4-bit variants use
# their quantised storage path rather than silently falling back to fp32.
_DIM = 256
_STEPS_DEFAULT = 5

# Five-step FSDP2 equivalence tolerances keyed by optim class, as (rtol, atol).
#
# NOTE: Muon32bit is looser than the (1e-4, 1e-5) originally pencilled into
# the design note because those CPU tests never cross the DTensor boundary.
# Newton-Schulz runs in bf16 (muon.py:119), so the tiny trajectory difference
# between the DTensor path and the plain replicated path grows to about 4e-3
# after 5 steps. Keep the 32-bit equivalence budget tight enough to catch
# genuine NS-on-shard / reshard bugs while leaving the quantized variants at
# their quantization-dominated tolerances.
MUON_FSDP_TOL: dict[type, tuple[float, float]] = {
    Muon32bit: (5e-3, 5e-3),
    Muon8bit: (2e-2, 2e-2),
    Muon4bit: (5e-2, 5e-2),
}

# Checkpoint resume runs 8 total steps across a save/load/reshard boundary; the
# 32-bit path empirically reaches about 1.4e-2 there, so keep the wider tolerance
# scoped to that scenario.
MUON_FSDP_CHECKPOINT_TOL: dict[type, tuple[float, float]] = {
    **MUON_FSDP_TOL,
    Muon32bit: (2e-2, 2e-2),
}


class _TinyMLP(nn.Module):
    """Deterministic 2-layer MLP: two square 2-D Muon params + 1-D biases.

    The weights (``net.0.weight``, ``net.2.weight``) are square so Newton-Schulz
    has a well-conditioned target; the biases (``net.0.bias``, ``net.2.bias``)
    form the 1-D group that must go to a separate elementwise optimiser under
    FSDP2 (§7.6).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_DIM, _DIM, bias=True)
        self.fc2 = nn.Linear(_DIM, _DIM, bias=True)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def make_tiny_model(seed: int = 0) -> nn.Module:
    """Deterministic 2-layer MLP with all-2-D Muon params + a 1-D bias group.

    Constructed identically on every rank for the given seed.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _TinyMLP()


def make_batch(seed: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic ``(input, target)`` batch on ``device``.

    Generated on CPU (so the RNG fork is device-independent and identical on
    every rank) then moved to ``device``.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1000 + seed)
        x = torch.randn(16, _DIM)
        target = torch.randn(16, _DIM)
    return x.to(device), target.to(device)


def split_muon_params(model: nn.Module):
    """Partition parameters into (>=2-D Muon group, 1-D elementwise group)."""
    muon_params, other_params = [], []
    for p in model.parameters():
        (muon_params if p.dim() >= 2 else other_params).append(p)
    return muon_params, other_params


def _build_optimizers(model: nn.Module, optim_cls, optim_kwargs: dict):
    """Muon for the 2-D weights, a DTensor-safe AdamW for the 1-D biases.

    Plain ``torch.optim.AdamW`` is elementwise, so it is correct on both a full
    tensor (reference) and a sharded DTensor (FSDP2) without modification.
    """
    muon_params, other_params = split_muon_params(model)
    kw = {**_FORCE_NS, **optim_kwargs}
    muon_opt = optim_cls(muon_params, **kw)
    bias_opt = torch.optim.AdamW(other_params, lr=kw.get("lr", 1e-3)) if other_params else None
    return muon_opt, bias_opt


def _train_loop(model, muon_opt, bias_opt, steps, device):
    for s in range(steps):
        x, target = make_batch(s, device)
        out = model(x)
        loss = ((out - target) ** 2).mean()
        muon_opt.zero_grad()
        if bias_opt is not None:
            bias_opt.zero_grad()
        loss.backward()
        muon_opt.step()
        if bias_opt is not None:
            bias_opt.step()


def _gather_named_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return ``{name: full_param}`` on CPU, all-gathering DTensors if needed."""
    out = {}
    for name, p in model.named_parameters():
        t = p.detach()
        # full_tensor() exists on DTensor (FSDP2); plain tensors lack it.
        full = t.full_tensor() if hasattr(t, "full_tensor") else t
        out[name] = full.float().cpu()
    return out


def run_reference(
    optim_cls, optim_kwargs: dict, steps: int = _STEPS_DEFAULT, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Single process, replicated (no FSDP). The ORACLE trajectory.

    Returns ``{param_name: full_param}`` after ``steps`` Muon steps.
    """
    device = torch.device("cpu")
    model = make_tiny_model(seed).to(device)
    muon_opt, bias_opt = _build_optimizers(model, optim_cls, optim_kwargs)
    _train_loop(model, muon_opt, bias_opt, steps, device)
    return _gather_named_params(model)


def run_fsdp2(optim_cls, optim_kwargs: dict, steps: int = _STEPS_DEFAULT, seed: int = 0) -> dict[str, torch.Tensor]:
    """Under torchrun. Wraps ``make_tiny_model`` with ``fully_shard`` and runs
    ``steps`` steps, then gathers every param via ``full_tensor()``.

    Requires an initialised process group. Returns an identical dict on every
    rank.
    """
    import torch.distributed as dist

    device = _worker_device()
    model = shard_model(make_tiny_model(seed).to(device), device)

    muon_opt, bias_opt = _build_optimizers(model, optim_cls, optim_kwargs)
    _train_loop(model, muon_opt, bias_opt, steps, device)
    dist.barrier()
    return _gather_named_params(model)


def shard_model(model: nn.Module, device) -> nn.Module:
    """Apply ``fully_shard`` on an explicit mesh matching ``device``.

    ``fully_shard`` defaults to the accelerator mesh whenever CUDA is available,
    which mismatches a CPU-forced run. Pin the mesh to ``device.type`` so each
    parameter becomes a DTensor with Shard(0) placement and its global shape
    preserved (§5).
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    fully_shard(model, mesh=mesh)
    return model


def _worker_device():
    """Resolve this rank's compute device for the torchrun worker.

    Honors ``BNB_MUON_FSDP_DEVICE=cpu`` so the harness can force a portable
    CPU+gloo run (avoids GPU oversubscription when nproc > device_count).
    """
    import os

    import torch.distributed as dist

    if os.environ.get("BNB_MUON_FSDP_DEVICE") == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
    return torch.device("cpu")
