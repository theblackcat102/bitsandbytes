"""FSDP2 / DTensor support for the Muon optimizer.

Tests are written against the **frozen interface contract** in
``docs/muon_distributed_design.md`` §11 (the test author writes against the
contract without seeing the implementation). They are organised by the three
layers of that contract:

* **Layer A** — pure, collective-free helpers (``_is_sharded``,
  ``_assert_supported_layout``, ``_global_matrix_shape``, ``_assign_owners``).
  Unit-testable today; no torchrun, no CUDA.
* **Layer B** — public optimizer behaviour under ``fully_shard``: single-rank vs
  multi-rank equivalence and cross-world-size checkpoint reshard. Launched via
  ``torchrun`` (``tests/fsdp_muon_worker.py``).
* **Guards** — unsupported layouts (FSDP1 FlatParameter / ZeRO-3 / 1-D global
  shape, ``is_paged`` + sharded) raise ``NotImplementedError`` with the
  contractually-required bracketed token.

Until the Pattern-B implementation lands, the Layer-A helpers do not exist yet;
those tests skip cleanly so the suite stays green during parallel dev.
"""

from __future__ import annotations

import os
import pathlib
import platform
import socket
import subprocess

import pytest
import torch

from tests.helpers import get_available_devices, id_formatter

# --- Optional import of the frozen Layer-A helpers (§11.A) -----------------
# These do not exist until the FSDP path is implemented; gate on availability
# so the file is collectable during parallel test/impl development.
try:
    from bitsandbytes.optim.muon import (
        _assert_supported_layout,
        _assign_owners,
        _global_matrix_shape,
        _is_sharded,
    )

    _HELPERS = True
except ImportError:
    _HELPERS = False

from bitsandbytes.optim.muon import Muon8bit, Muon32bit

requires_helpers = pytest.mark.skipif(not _HELPERS, reason="Muon FSDP helpers not implemented yet (§11.A)")


# ===========================================================================
# Layer A — _assign_owners (pure, no torch needed beyond lists)
# ===========================================================================
@requires_helpers
class TestAssignOwners:
    def test_length_and_range(self):
        costs = [10, 20, 30, 5, 1]
        owners = _assign_owners(costs, world_size=3)
        assert len(owners) == len(costs)
        assert all(0 <= o < 3 for o in owners)

    def test_deterministic(self):
        costs = [7, 7, 3, 9, 1, 4]
        assert _assign_owners(costs, 4) == _assign_owners(costs, 4)

    def test_balances_load(self):
        # Largest-processing-time greedy should keep per-rank load near-balanced.
        costs = [8, 7, 6, 5, 4, 3, 2, 1]
        ws = 4
        owners = _assign_owners(costs, ws)
        loads = [0] * ws
        for c, o in zip(costs, owners):
            loads[o] += c
        # Optimal makespan for this set over 4 ranks is 9; greedy LPT achieves it.
        assert max(loads) - min(loads) <= max(costs)

    def test_single_rank_all_zero(self):
        assert _assign_owners([5, 1, 9], world_size=1) == [0, 0, 0]

    def test_tie_break_prefers_lower_rank(self):
        # All equal costs: first assignment goes to rank 0, then 1, ...
        owners = _assign_owners([1, 1, 1, 1], world_size=2)
        assert owners[0] == 0 and owners[1] == 1


# ===========================================================================
# Layer A — _global_matrix_shape (pure)
# ===========================================================================
@requires_helpers
class TestGlobalMatrixShape:
    def test_2d_passthrough(self):
        p = torch.empty(256, 64)
        assert tuple(_global_matrix_shape(p)) == (256, 64)

    def test_3d_collapses_trailing(self):
        p = torch.empty(8, 16, 4)
        assert tuple(_global_matrix_shape(p)) == (8, 64)

    def test_4d_collapses_trailing(self):
        p = torch.empty(4, 3, 8, 8)
        assert tuple(_global_matrix_shape(p)) == (4, 192)


# ===========================================================================
# Layer A — _is_sharded / _assert_supported_layout on plain tensors (no dist)
# ===========================================================================
@requires_helpers
class TestPlainTensorLayout:
    def test_plain_tensor_not_sharded(self):
        assert _is_sharded(torch.empty(8, 8)) is False

    def test_plain_tensor_layout_ok(self):
        # No-op (does not raise) for ordinary tensors.
        _assert_supported_layout(torch.empty(16, 16))


# ===========================================================================
# Layer A / Guards — real DTensors via a single-rank process group
# ===========================================================================
@pytest.fixture(scope="module")
def single_rank_pg():
    """Init a 1-rank gloo group so we can build real DTensors without torchrun."""
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        with socket.socket() as s:
            s.bind(("", 0))
            os.environ.setdefault("MASTER_PORT", str(s.getsockname()[1]))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    elif not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    yield
    # Leave the group up for the module; torch tears it down at exit.


def _mesh():
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


@requires_helpers
class TestDTensorLayout:
    def test_replicate_dtensor_not_sharded(self, single_rank_pg):
        from torch.distributed.tensor import Replicate, distribute_tensor

        dt = distribute_tensor(torch.empty(16, 16), _mesh(), [Replicate()])
        assert _is_sharded(dt) is False
        _assert_supported_layout(dt)  # replicate is supported, no raise

    def test_2d_shard_dtensor_is_sharded_and_supported(self, single_rank_pg):
        from torch.distributed.tensor import Shard, distribute_tensor

        dt = distribute_tensor(torch.empty(16, 16), _mesh(), [Shard(0)])
        assert _is_sharded(dt) is True
        _assert_supported_layout(dt)  # 2-D global shape -> supported

    def test_1d_shard_global_shape_raises_ndim(self, single_rank_pg):
        from torch.distributed.tensor import Shard, distribute_tensor

        dt = distribute_tensor(torch.empty(64), _mesh(), [Shard(0)])
        with pytest.raises(NotImplementedError, match=r"\[ndim\]"):
            _assert_supported_layout(dt)

    def test_min_8bit_size_uses_global_numel(self, single_rank_pg):
        from torch.distributed.tensor import DTensor, Shard

        local = torch.empty(64, 64)
        dt = DTensor.from_local(
            local,
            _mesh(),
            [Shard(0)],
            run_check=False,
            shape=torch.Size((128, 64)),
            stride=(64, 1),
        )
        opt = Muon8bit([dt], min_8bit_size=6000)
        opt.init_state(opt.param_groups[0], dt, 0, 0)

        assert local.numel() < 6000 < dt.numel()
        assert opt.state[dt]["state1"].dtype == torch.uint8
        assert opt.state[dt]["state1"].shape == local.shape

    def test_quantized_checkpoint_rejects_cross_world_reshard(self, single_rank_pg):
        from torch.distributed.tensor import Shard, distribute_tensor

        dt = distribute_tensor(torch.empty(128, 64), _mesh(), [Shard(0)])
        opt = Muon8bit([dt])
        opt.init_state(opt.param_groups[0], dt, 0, 0)
        state_dict = opt.state_dict()
        param_state = next(iter(state_dict["state"].values()))

        key = opt._FSDP2_QUANTIZED_WORLD_SIZE_KEY
        assert param_state[key] == 1
        param_state[key] = 2
        with pytest.raises(NotImplementedError, match="cross-world-size"):
            opt.load_state_dict(state_dict)


# ===========================================================================
# Guards — framework-specific unsupported layouts (§11.A tokens)
# ===========================================================================
@requires_helpers
class TestUnsupportedFrameworkGuards:
    def test_fsdp1_flatparameter_raises(self):
        # Real FSDP1 FlatParameters only exist after FSDP1-wrapping (needs a live
        # process group); constructing one standalone collapses to a plain
        # Parameter. The impl detects FSDP1 by the class name, so a fake subclass
        # named FlatParameter (a flat 1-D blob) exercises the same code path.
        class FlatParameter(torch.Tensor):
            pass

        flat = FlatParameter._make_subclass(FlatParameter, torch.empty(1024))
        with pytest.raises(NotImplementedError, match=r"\[FSDP1\]"):
            _assert_supported_layout(flat)

    def test_zero3_partitioned_param_raises(self):
        # DeepSpeed ZeRO-3 tags params with ds_* attributes on a 1-D shard.
        class _ZeRO3Param(torch.Tensor):
            pass

        p = _ZeRO3Param._make_subclass(_ZeRO3Param, torch.empty(512))
        p.ds_id = 0
        p.ds_shape = (256, 256)
        p.ds_tensor = torch.empty(512)
        with pytest.raises(NotImplementedError, match=r"\[ZeRO-3\]"):
            _assert_supported_layout(p)


# ===========================================================================
# Guards — is_paged + sharded params unsupported for the first cut (§11.B.6)
# ===========================================================================
@requires_helpers
def test_paged_plus_sharded_raises(single_rank_pg):
    from torch.distributed.tensor import Shard, distribute_tensor

    dt = distribute_tensor(torch.empty(128, 64), _mesh(), [Shard(0)])
    dt.grad = torch.zeros_like(dt)
    with pytest.raises(NotImplementedError, match=r"\[paged\]"):
        opt = Muon8bit([dt], lr=1e-3, is_paged=True)
        opt.step()


# ===========================================================================
# Layer B — multi-rank equivalence + checkpoint reshard (torchrun)
# ===========================================================================
_WORKER = pathlib.Path(__file__).with_name("fsdp_muon_worker.py")
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _torchrun(nproc: int, *args: str, timeout: int = 300):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        str(_WORKER),
        *args,
    ]
    # Force CPU+gloo so the tests are hardware-independent: nproc=4 on a 2-GPU
    # box would oversubscribe NCCL (1 process/GPU). Drop this env var to run on
    # accelerators when ranks <= device_count.
    env = {**os.environ, "BNB_MUON_FSDP_DEVICE": "cpu"}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=_REPO_ROOT, env=env)


def _require_torchrun():
    if platform.system() == "Windows":
        pytest.skip("FSDP2 / torchrun not supported on Windows")
    import shutil

    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")


@pytest.mark.parametrize("nproc", [2, 4], ids=id_formatter("nproc"))
@pytest.mark.parametrize("optim_name", ["Muon32bit", "Muon8bit"], ids=id_formatter("optim"))
def test_fsdp2_equivalence(nproc, optim_name):
    """§8 / §11.B.1: N FSDP2 steps == single-rank reference within tolerance."""
    _require_torchrun()
    # 8-bit needs quantised storage to be meaningful; CPU gloo handles fp32 fine.
    res = _torchrun(nproc, "equivalence", optim_name)
    if res.returncode != 0:
        pytest.fail(f"equivalence ({optim_name}, ws={nproc}) failed:\n{res.stdout}\n{res.stderr}")
    assert "SUCCESS" in res.stdout


def test_fsdp2_checkpoint_reshard(tmp_path):
    """§11.B.4: save full optim state at ws=2, load at ws=4, resume, match oracle."""
    _require_torchrun()
    sd = tmp_path / "muon_dcp_ckpt"  # DCP writes a directory, not a single file

    save = _torchrun(2, "ckpt-save", "Muon32bit", str(sd))
    if save.returncode != 0:
        pytest.fail(f"ckpt-save failed:\n{save.stdout}\n{save.stderr}")
    assert sd.is_dir(), "DCP checkpoint directory was not written"

    load = _torchrun(4, "ckpt-load", "Muon32bit", str(sd))
    if load.returncode != 0:
        pytest.fail(f"ckpt-load (reshard ws2->ws4) failed:\n{load.stdout}\n{load.stderr}")
    assert "SUCCESS" in load.stdout


# ===========================================================================
# Smoke — the Layer-C oracle harness runs without distributed
# ===========================================================================
@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
def test_reference_harness_runs(device):
    """The single-rank oracle (run_reference) is the comparison baseline for all
    Layer-B tests; verify it trains the tiny model without NaNs."""
    if device != "cpu":
        pytest.skip("reference oracle is defined on CPU (§11.C)")
    from tests._muon_fsdp_utils import run_reference

    out = run_reference(Muon32bit, dict(lr=1e-2, momentum=0.9, weight_decay=0.1), steps=3)
    assert out, "no params returned"
    for name, t in out.items():
        assert torch.isfinite(t).all(), f"{name} has non-finite values"
