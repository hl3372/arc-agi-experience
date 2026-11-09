"""Checkpoint DCP layout detection without importing torch (train.py's resume scan runs before the trainer loads)."""
import os


def dcp_layout(ckpt_dir: str):
    """'joint' (dcp/), 'split' (dcp_model/ + dcp_optim/), 'weights' (dcp_model/ only) or None."""
    if os.path.isdir(os.path.join(ckpt_dir, "dcp")):
        return "joint"
    if os.path.isdir(os.path.join(ckpt_dir, "dcp_model")):
        return "split" if os.path.isdir(os.path.join(ckpt_dir, "dcp_optim")) else "weights"
    return None


def _metadata_files(dcp_dir: str):
    """The shard files a DCP checkpoint's .metadata references (what a load reads, whatever the loading world size), or
    None when the metadata cannot be read (then the caller falls back to one file per restoring rank)."""
    try:
        from torch.distributed.checkpoint import FileSystemReader
        md = FileSystemReader(dcp_dir).read_metadata()
        files = {getattr(v, "relative_path", None) for v in (md.storage_data or {}).values()}
        files.discard(None)
        return files or None
    except Exception:  # noqa: BLE001
        return None


def dcp_missing(ckpt_dir: str, world: int) -> list:
    """What a restore by `world` ranks still lacks in this node's copy of ckpt_dir (node-local multi-node checkpoints:
    DCP dedups replicated entries onto single ranks, so every node needs every rank's shard files and the coordinator's
    .metadata, not only its own ranks' shards). [] = complete.
    The files are the ones the checkpoint's .metadata references (a checkpoint saved by 8 ranks and restored by 16 has
    8 shard files per directory); one file per restoring rank only when the metadata is unreadable."""
    layout = dcp_layout(ckpt_dir)
    if layout is None:
        return [f"{ckpt_dir}: no DCP state"]
    subs = {"joint": ["dcp"], "split": ["dcp_model", "dcp_optim"], "weights": ["dcp_model"]}[layout]
    miss = []
    for sub in subs:
        d = os.path.join(ckpt_dir, sub)
        names = set(os.listdir(d))
        if ".metadata" not in names:
            miss.append(f"{sub}/.metadata")
        need = _metadata_files(d) if ".metadata" in names else None
        if need is not None:
            miss += [f"{sub}/{f}" for f in sorted(need) if f not in names]
            continue
        for r in range(world):
            if not any(n.startswith(f"__{r}_") and n.endswith(".distcp") for n in names):
                miss.append(f"{sub}/__{r}_*.distcp")
    return miss


def fs_type(path: str) -> str:
    """The filesystem type of the mount holding path (/proc/mounts, longest matching mount point); '' if unknown."""
    path = os.path.realpath(path)
    best, kind = "", ""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, kind = mnt, parts[2]
    except OSError:
        return ""
    return kind
