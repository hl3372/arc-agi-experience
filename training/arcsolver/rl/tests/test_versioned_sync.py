"""Multi-node FP8 sync: every node's writer rank exports the same gathered tensors into a fresh v<step> dir; the copies'
full manifests agree (they are compared before any engine reloads), a non-writer writes nothing, the frozen tensors are
written once and linked into each version, and a checkpoint's model_fp8 export from a version dir equals the export
from the single sync dir. Plus the node-local checkpoint completeness check (fsdp_layout.dcp_missing)."""
import os
import socket
import tempfile

import pytest

from arcsolver.rl import fsdp_layout as L
from arcsolver.rl import identity as RI

from .helpers import fp8_checkpoint, raw_bytes, safetensors_dir, tiny_qwen35


def test_dcp_missing_and_fs_type(tmp_path):
    ck = tmp_path / "step_000010"
    (ck / "dcp").mkdir(parents=True)
    for r in (0, 1):                                   # node 0's ranks only, no .metadata yet
        (ck / "dcp" / f"__{r}_0.distcp").write_bytes(b"x")
    miss = L.dcp_missing(str(ck), 4)
    assert "dcp/.metadata" in miss and "dcp/__2_*.distcp" in miss and "dcp/__3_*.distcp" in miss
    (ck / "dcp" / ".metadata").write_bytes(b"m")
    for r in (2, 3):
        (ck / "dcp" / f"__{r}_0.distcp").write_bytes(b"x")
    assert L.dcp_missing(str(ck), 4) == []
    sp = tmp_path / "split"
    (sp / "dcp_model").mkdir(parents=True)
    (sp / "dcp_optim").mkdir()
    for sub in ("dcp_model", "dcp_optim"):
        (sp / sub / ".metadata").write_bytes(b"m")
        (sp / sub / "__0_0.distcp").write_bytes(b"x")
    assert L.dcp_missing(str(sp), 1) == [] and L.dcp_missing(str(sp), 2) == ["dcp_model/__1_*.distcp",
                                                                             "dcp_optim/__1_*.distcp"]
    assert L.dcp_missing(str(tmp_path / "none"), 1)
    assert L.fs_type("/") != "" and L.fs_type(str(tmp_path)) != ""


def test_dcp_missing_follows_metadata_across_world_sizes(tmp_path):
    """A checkpoint saved by fewer ranks (8 on one node) restored by more (16 on two): the complete set is the files its
    .metadata references, not one file per restoring rank."""
    torch = pytest.importorskip("torch")
    import torch.distributed.checkpoint as dcp
    ck = tmp_path / "step_000001"
    for sub in ("dcp_model", "dcp_optim"):
        dcp.save({"w": torch.arange(64.0), "b": torch.ones(8)}, storage_writer=dcp.FileSystemWriter(str(ck / sub)),
                 no_dist=True)
    files = sorted(f for f in os.listdir(ck / "dcp_model") if f.endswith(".distcp"))
    assert files and all(f.startswith("__0_") for f in files)          # written by one rank
    assert L.dcp_missing(str(ck), 16) == []                             # complete for any restoring world size
    os.remove(ck / "dcp_optim" / files[0])
    assert L.dcp_missing(str(ck), 16) == [f"dcp_optim/{files[0]}"]      # a referenced file missing is still caught
    (ck / "dcp_model" / ".metadata").write_bytes(b"not a metadata file")  # unreadable: the per-rank fallback
    assert "dcp_model/__1_*.distcp" in L.dcp_missing(str(ck), 2)


def test_versioned_sync_copies_agree():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    import torch.distributed as dist
    torch.backends.mkldnn.enabled = False
    d = tempfile.mkdtemp(prefix="vsync_")
    tiny_qwen35(d)
    d8 = d + "_fp8"
    fp8_checkpoint(d, d8)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", WORLD_SIZE="1")
    dist.init_process_group("gloo")
    try:
        from arcsolver.rl import fsdp as F
        model = F.load(d, torch.device("cpu"), chunk=8, keep_gathered=False, freeze_embed=True)
        node0 = F.SyncWriterFP8(d8, os.path.join(d, "n0", "sync"))      # two nodes' sync dirs (same path per node)
        node1 = F.SyncWriterFP8(d8, os.path.join(d, "n1", "sync"))
        v0 = os.path.join(node0.dir, "v000001")
        v1 = os.path.join(node1.dir, "v000001")
        st0 = node0.write_stream(model, 0, writer=True, dest=v0)
        st1 = node1.write_stream(model, 3, writer=True, dest=v1)        # a non-zero rank that is its node's writer
        assert st0["shards"] >= 1 and st1["bytes"] == st0["bytes"]
        m0, m1 = RI.dir_manifest(v0), RI.dir_manifest(v1)
        assert m0["sha256"] == m1["sha256"] and "model-frozen.safetensors" in m0["files"]
        # the frozen file lives once under <sync>/frozen and is linked (same inode) into the version
        fz = os.path.join(node0.dir, "frozen", "model-frozen.safetensors")
        assert os.path.exists(fz) and os.stat(fz).st_ino == os.stat(os.path.join(v0, "model-frozen.safetensors")).st_ino
        # a non-writer gathers (collective) but writes nothing
        quiet = os.path.join(d, "n2", "sync")
        assert F.SyncWriterFP8(d8, quiet).write_stream(model, 1, writer=False, dest=os.path.join(quiet, "v000001")) is None
        assert not os.path.exists(quiet)
        # the version holds exactly the base checkpoint's tensors (first sync = no change), like the single-dir layout
        a, b = safetensors_dir(d8), safetensors_dir(v0)
        assert set(a) == set(b) and all(torch.equal(raw_bytes(a[k]), raw_bytes(b[k])) for k in a)
        single = F.SyncWriterFP8(d8, os.path.join(d, "single"))
        single.write_stream(model, 0)
        e1, e2 = os.path.join(d, "exp_v"), os.path.join(d, "exp_single")
        F.export_fp8_dir(v0, d8, e1, {"step": 1})
        F.export_fp8_dir(single.dir, d8, e2, {"step": 1})
        x, y = safetensors_dir(e1), safetensors_dir(e2)
        assert set(x) == set(y) and all(torch.equal(raw_bytes(x[k]), raw_bytes(y[k])) for k in x)
        # a second version after an update differs, and the first stays intact until the caller removes it
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "layers.0.mlp" in n:
                    p.add_(0.5)
        v0b = os.path.join(node0.dir, "v000002")
        node0.write_stream(model, 0, writer=True, dest=v0b)
        assert RI.dir_manifest(v0b)["sha256"] != m0["sha256"] and RI.dir_manifest(v0)["sha256"] == m0["sha256"]
    finally:
        dist.destroy_process_group()
