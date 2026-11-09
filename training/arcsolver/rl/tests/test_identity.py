"""Pool and policy identities (identity.py) and the bank-holdout content check: a same-size pool change, edited task
content and repetition counts all change the pool identity; weights are hashed in full; the logit bias by its map."""
import json
import os
from types import SimpleNamespace as NS

from arcsolver.rl import core as oc
from arcsolver.rl import identity as RI


def task(k):
    return {"train": [{"input": [[k]], "output": [[k + 1]]}], "test": [{"input": [[k, k]], "output": [[k]]}]}


def items(names, tasks):
    return [NS(source="bank", task_id=n, path=n, student_ids=[len(n), tasks[n]["train"][0]["input"][0][0]]) for n in names]


def test_equal_size_pool_change_is_detected(tmp_path):
    tasks = {n: task(i) for i, n in enumerate("ABCD")}
    load = tasks.__getitem__
    m0 = RI.pool_manifest(items("ABC", tasks), [1, 1, 1], load)
    m1 = RI.pool_manifest(items("ACD", tasks), [1, 1, 1], load)        # same size, same counts
    assert m0["sha256"] != m1["sha256"]
    rel = RI.write_pool_manifest(str(tmp_path), m0)
    state = {"pool_manifest_sha256": m0["sha256"], "pool_manifest_file": rel}
    assert RI.check_pool_resume(str(tmp_path), state, m0, False, False) == []
    reasons = RI.check_pool_resume(str(tmp_path), state, m1, False, False)
    assert reasons and any("#1" in r for r in reasons)
    assert RI.check_pool_resume(str(tmp_path), state, m1, True, False) == []
    assert RI.check_pool_resume(str(tmp_path), {}, m0, False, False)            # legacy checkpoint refused ...
    assert RI.check_pool_resume(str(tmp_path), {}, m0, False, True) == []       # ... unless allowed
    # same ids, edited content
    tasks2 = dict(tasks, B=task(9))
    m2 = RI.pool_manifest(items("ABC", tasks2), [1, 1, 1], tasks2.__getitem__)
    assert m2["sha256"] != m0["sha256"]
    # repetition counts are part of the identity
    assert RI.pool_manifest(items("ABC", tasks), [1, 2, 1], load)["sha256"] != m0["sha256"]


def test_pool_manifest_file_is_immutable(tmp_path):
    tasks = {n: task(i) for i, n in enumerate("AB")}
    m = RI.pool_manifest(items("AB", tasks), [1, 1], tasks.__getitem__)
    rel = RI.write_pool_manifest(str(tmp_path), m)
    assert RI.write_pool_manifest(str(tmp_path), m) == rel                       # same content: fine
    path = os.path.join(str(tmp_path), rel)
    with open(path) as f:
        d = json.load(f)
    d["items"][0][1] = "Z"
    with open(path, "w") as f:
        json.dump(d, f)
    try:
        RI.write_pool_manifest(str(tmp_path), m)
        raise AssertionError("tampered manifest accepted")
    except RuntimeError:
        pass


def test_dir_manifest_sees_a_middle_byte(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    blob = bytearray(os.urandom(3 << 20))
    (d / "model-trained-00000.safetensors").write_bytes(bytes(blob))
    (d / "config.json").write_text("{}")
    a = RI.dir_manifest(str(d))
    blob[(3 << 20) // 2] ^= 0xFF                                              # a mid-shard change
    (d / "model-trained-00000.safetensors").write_bytes(bytes(blob))
    b = RI.dir_manifest(str(d))
    assert a["sha256"] != b["sha256"] and set(a["files"]) == {"config.json", "model-trained-00000.safetensors"}
    (d / "x.part").write_bytes(b"partial")                                   # in-flight files are not part of it
    assert RI.dir_manifest(str(d))["sha256"] == b["sha256"]


def test_logit_bias_and_policy_ids():
    assert RI.logit_bias_sha(None) == "" and RI.logit_bias_sha({}) == ""
    assert RI.logit_bias_sha({"5": -1.0}) != RI.logit_bias_sha({"5": -10.0})   # a boolean marker could not tell
    assert RI.logit_bias_sha({"5": -1.0, "7": 2.0}) == RI.logit_bias_sha({"7": 2.0, "5": -1})
    s = {"kv_cache_dtype": "fp8", "mtp_tokens": 3}
    assert RI.policy_id("a", s) != RI.policy_id("b", s) != RI.policy_id("a", dict(s, mtp_tokens=0))


def test_renamed_holdout_task_is_excluded(tmp_path):
    bank, arc, ev = tmp_path / "bank", tmp_path / "arc", tmp_path / "eval"
    for d in (bank, arc, ev):
        d.mkdir()
    (bank / "held.json").write_text(json.dumps(task(1)))
    (bank / "renamed.json").write_text(json.dumps(task(1)))                  # the same content under another id
    (bank / "other.json").write_text(json.dumps(task(2)))
    (ev / "e1.json").write_text(json.dumps(task(7)))
    load = lambda p: json.load(open(p))  # noqa: E731
    got, report = oc.build_pool(str(bank), str(arc), str(ev), set(), holdout_ids={"held"}, load=load)
    ids = {it.task_id for it in got}
    assert ids == {"other"} and report["dropped_holdout_content"] == [("renamed", "held")]
