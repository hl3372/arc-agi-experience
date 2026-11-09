"""Pool and policy identities for train.py.

Pool identity: the ordered (source, task_id, task-content sha256, rendered-prompt sha256, copies) list of the prompt
pool. Saved immutably as OUT/pools/pool_<sha16>.json (never overwritten) and referenced from every checkpoint; a resume
checks it before restoring the prompt order's index state, so an equal-size pool edit can no longer re-map saved
indices onto different tasks.

Policy identity: what a sampler serves = the complete weight directory it loaded, as a manifest of every file's sha256
and size (not a few sampled tensors, not the first/last MiB of a shard) plus the effective serving settings
(quantization / KV dtype / MTP / sampling config / logit-bias map). Its digest keys the rollout and eval stores and is
what every engine acknowledges after a reload.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# ---- pool --------------------------------------------------------------------------------------------------------------

def pool_manifest(pool, counts, load) -> dict:
    """pool: PromptItems (task_id, source, path, student_ids) in pool order; counts: copies per epoch; load(path) -> task."""
    items = []
    for it, c in zip(pool, counts):
        content = canonical_sha256(load(it.path))
        prompt = hashlib.sha256(json.dumps(list(it.student_ids or [])).encode()).hexdigest() if getattr(
            it, "student_ids", None) is not None else ""
        items.append([it.source, it.task_id, content, prompt, int(c)])
    return {"version": 1, "items": items, "sha256": canonical_sha256(items)}


def write_pool_manifest(out: str, manifest: dict) -> str:
    """OUT/pools/pool_<sha16>.json, written once (an existing file must hold the same manifest). Returns the relative path."""
    rel = os.path.join("pools", f"pool_{manifest['sha256'][:16]}.json")
    path = os.path.join(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("sha256") != manifest["sha256"] or canonical_sha256(old.get("items")) != manifest["sha256"]:
            raise RuntimeError(f"{path} exists with different content")
        return rel
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return rel


def pool_diff(saved_items: list, items: list, limit: int = 5) -> list:
    """Readable differences between two pool manifests' item lists (position, before, after)."""
    out = []
    if len(saved_items) != len(items):
        out.append(f"size {len(saved_items)} -> {len(items)}")
    for i, (a, b) in enumerate(zip(saved_items, items)):
        if list(a) != list(b):
            out.append(f"#{i}: {a[:2]} content {str(a[2])[:8]} -> {b[:2]} content {str(b[2])[:8]}")
            if len(out) >= limit:
                break
    return out


def check_pool_resume(out: str, state: dict, manifest: dict, allow_change: bool, legacy_ok: bool) -> list:
    """[] if the checkpoint's pool manifest equals this launch's (or a change is explicitly allowed), else reasons."""
    saved_sha = state.get("pool_manifest_sha256")
    if saved_sha is None:
        return [] if legacy_ok else ["the checkpoint has no pool manifest (saved before pool identities): "
                                     "--allow-objective-change legacy to resume it anyway"]
    if saved_sha == manifest["sha256"] or allow_change:
        return []
    reasons = [f"pool manifest {saved_sha[:16]} in the checkpoint, {manifest['sha256'][:16]} now"]
    rel = state.get("pool_manifest_file")
    if rel and os.path.exists(os.path.join(out, rel)):
        with open(os.path.join(out, rel)) as f:
            reasons += pool_diff(json.load(f)["items"], manifest["items"])
    return reasons


# ---- policy ------------------------------------------------------------------------------------------------------------

def _sha_file(path: str, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_manifest(d: str, threads: int = 8) -> dict:
    """{relative file: [size, sha256]} of every regular file under d (sorted), hashed in parallel; plus its digest."""
    files = []
    for root, _dirs, names in os.walk(d):
        for n in names:
            p = os.path.join(root, n)
            if os.path.isfile(p) and not n.endswith((".part", ".tmp")):
                files.append(os.path.relpath(p, d))
    files.sort()
    with cf.ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        shas = list(ex.map(lambda r: _sha_file(os.path.join(d, r)), files))
    m = {r: [os.path.getsize(os.path.join(d, r)), s] for r, s in zip(files, shas)}
    return {"files": m, "sha256": canonical_sha256(m)}


def serving_settings(args, extra=None) -> dict:
    """The engine settings that change what a policy samples (beyond its weights)."""
    keys = ("sampler_fp8", "kv_cache_dtype", "mtp_tokens", "temperature", "sampler_max_model_len",
            "engine_attention_backend", "arm", "think_prefix", "chat_template", "reasoning_effort")
    s = {k: getattr(args, k, None) for k in keys}
    s.update(extra or {})
    return s


def policy_id(weights_manifest_sha: str, settings: dict) -> str:
    return canonical_sha256({"weights": weights_manifest_sha, "settings": settings})


def logit_bias_sha(lb) -> str:
    """The full bias map's hash ('' when there is none): a boolean marker cannot tell two different maps apart."""
    return canonical_sha256(sorted((str(k), float(v)) for k, v in (lb or {}).items())) if lb else ""
