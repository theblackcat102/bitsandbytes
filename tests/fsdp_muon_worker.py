"""Multi-rank worker for the Muon FSDP2 tests (Layer B of §11).

Launched via torchrun by ``tests/test_muon_fsdp.py``; never run directly. It
proves the invariant from ``docs/muon_distributed_design.md`` §11.D: the
distributed (FSDP2 / DTensor) Muon step is tolerance-equivalent to the
single-rank replicated step.

Usage (driven by the pytest harness)::

    torchrun --nproc_per_node=2 tests/fsdp_muon_worker.py equivalence Muon32bit
    torchrun --nproc_per_node=2 tests/fsdp_muon_worker.py ckpt-save  Muon32bit /tmp/sd.pt
    torchrun --nproc_per_node=4 tests/fsdp_muon_worker.py ckpt-load  Muon32bit /tmp/sd.pt

Exit code 0 == all assertions held; non-zero == failure (stdout/stderr are
surfaced by the calling pytest test).
"""

import sys

import torch
import torch.distributed as dist

# Allow `torchrun tests/fsdp_muon_worker.py` from the repo root.
sys.path.insert(0, ".")

from bitsandbytes.optim.muon import Muon4bit, Muon8bit, Muon32bit
from tests._muon_fsdp_utils import (
    MUON_FSDP_CHECKPOINT_TOL,
    MUON_FSDP_TOL,
    _build_optimizers,
    _gather_named_params,
    _train_loop,
    _worker_device,
    make_tiny_model,
    run_reference,
    shard_model,
)

_CLASSES = {"Muon32bit": Muon32bit, "Muon8bit": Muon8bit, "Muon4bit": Muon4bit}
_SAVE_STEPS = 5
_RESUME_STEPS = 3


def _init():
    import os

    force_cpu = os.environ.get("BNB_MUON_FSDP_DEVICE") == "cpu"
    use_cuda = torch.cuda.is_available() and not force_cpu
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def _assert_matches(got, ref, optim_cls, label, *, only_muon=False, tolerances=MUON_FSDP_TOL):
    rtol, atol = tolerances[optim_cls]
    for name in ref:
        # Muon manages the 2-D weights; 1-D biases use a separate AdamW that is
        # outside the Muon checkpoint contract (§7.6), so the checkpoint test
        # compares only the Muon-managed params.
        if only_muon and not name.endswith(".weight"):
            continue
        a, b = got[name], ref[name]
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            diff = (a - b).abs().max().item()
            raise AssertionError(f"[{label}] param {name!r} diverged: max|Δ|={diff:.3e} (rtol={rtol}, atol={atol})")


def _scenario_equivalence(optim_cls):
    """§11.B.1: FSDP2 trajectory == single-rank oracle within tolerance."""
    from tests._muon_fsdp_utils import run_fsdp2

    kw = dict(lr=1e-2, momentum=0.9, weight_decay=0.1)
    got = run_fsdp2(optim_cls, kw, steps=_SAVE_STEPS, seed=0)
    ref = run_reference(optim_cls, kw, steps=_SAVE_STEPS, seed=0)
    _assert_matches(got, ref, optim_cls, f"equivalence ws={dist.get_world_size()}")


def _fsdp_model_and_opt(optim_cls, kw):
    device = _worker_device()
    model = shard_model(make_tiny_model(0).to(device), device)
    muon_opt, bias_opt = _build_optimizers(model, optim_cls, kw)
    return model, muon_opt, bias_opt, device


def _scenario_ckpt_save(optim_cls, path):
    """Train ``_SAVE_STEPS`` steps; DCP-save through the standard FSDP2 API."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    kw = dict(lr=1e-2, momentum=0.9, weight_decay=0.1)
    model, muon_opt, bias_opt, device = _fsdp_model_and_opt(optim_cls, kw)
    _train_loop(model, muon_opt, bias_opt, _SAVE_STEPS, device)

    sd = get_optimizer_state_dict(model, muon_opt)
    dcp.save({"opt": sd}, checkpoint_id=path)
    dist.barrier()


def _scenario_ckpt_load(optim_cls, path):
    """§11.B.4: load the ws=W1 checkpoint at this (different) world size, reshard
    via load_state_dict, resume, and match the single-rank oracle of N+K steps."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

    kw = dict(lr=1e-2, momentum=0.9, weight_decay=0.1)
    model, muon_opt, bias_opt, device = _fsdp_model_and_opt(optim_cls, kw)

    # Build a template state_dict on the *current* mesh (DTensors with this
    # world size's placements), then let DCP fill it from the saved shards —
    # this is the reshard from W1 -> W2.
    _train_loop(model, muon_opt, bias_opt, _SAVE_STEPS, device)
    template = get_optimizer_state_dict(model, muon_opt)
    dcp.load({"opt": template}, checkpoint_id=path)
    set_optimizer_state_dict(model, muon_opt, template)

    # Resume and compare the full trajectory to the oracle.
    _train_loop(model, muon_opt, bias_opt, _RESUME_STEPS, device)
    got = _gather_named_params(model)
    ref = run_reference(optim_cls, kw, steps=_SAVE_STEPS + _RESUME_STEPS, seed=0)
    _assert_matches(
        got,
        ref,
        optim_cls,
        f"ckpt-load ws={dist.get_world_size()}",
        only_muon=True,
        tolerances=MUON_FSDP_CHECKPOINT_TOL,
    )


def main():
    scenario = sys.argv[1]
    optim_cls = _CLASSES[sys.argv[2]]
    _init()
    try:
        if scenario == "equivalence":
            _scenario_equivalence(optim_cls)
        elif scenario == "ckpt-save":
            _scenario_ckpt_save(optim_cls, sys.argv[3])
        elif scenario == "ckpt-load":
            _scenario_ckpt_load(optim_cls, sys.argv[3])
        else:
            raise SystemExit(f"unknown scenario {scenario!r}")
        if dist.get_rank() == 0:
            print(f"{scenario}/{sys.argv[2]}: SUCCESS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
