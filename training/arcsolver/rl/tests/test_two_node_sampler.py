"""Multi-node sampler layer on CPU: node_service.py serving fake engines (fake_vllm_worker.py) over authenticated TCP,
workers.NodeServiceClient / RemoteEngineClient on the coordinator side and train.MultiEngine on top, wired as the
two-node RL run wires them, minus vLLM and GPUs.

Every service and engine runs as a real process on localhost; each test starts its own services under a unique run name
and its fixture tears them down (gracefully, checking that nothing is left, then by force). Every wait is bounded.
"""
import glob
import importlib.util
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import pytest

from arcsolver.rl import identity as RI
from arcsolver.rl import train as D
from arcsolver.rl import workers as W

if sys.platform != "linux" or shutil.which("pgrep") is None:
    pytest.skip("needs Linux (/proc, PR_SET_PDEATHSIG) and pgrep", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # the directory holding the arcsolver package
FAKE = os.path.join(HERE, "fake_vllm_worker.py")
SERVICE = [sys.executable, "-m", "arcsolver.rl.node_service"]
TOKEN = f"t2n{os.getpid()}x{uuid.uuid4().hex[:4]}"          # every run name of this module starts with it
_spec = importlib.util.spec_from_file_location("fake_vllm_worker", FAKE)
FW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FW)

PARAMS = {"n": 3, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "logprobs": True}


# ---- helpers -----------------------------------------------------------------------------------------------------------

def wait_until(cond, timeout, interval=0.05):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(interval)
    return bool(cond())


def pgrep(pattern):
    """pids whose command line matches `pattern` (always bracketed by the callers, so it never matches pgrep itself)."""
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def engine_pids(run):
    return pgrep(f"[f]ake_vllm_worker\\.py.*/rl_{run}_n")


def service_pids(run):
    return pgrep(f"[a]rcsolver\\.rl\\.node_service.*--run-name {run} ")


def sockets(run):
    return glob.glob(f"/tmp/rl_{run}_n*_sampler*.sock")


def pid_alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def in_thread(fn, *a, **kw):
    """Run fn in a daemon thread; returns a join(timeout) -> (finished, result, exception) function. A hang shows up
    as finished=False instead of hanging the test (teardown then kills the processes, which unblocks the thread)."""
    box = {}

    def run():
        try:
            box["result"] = fn(*a, **kw)
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()

    def join(timeout):
        t.join(timeout)
        return (not t.is_alive()), box.get("result"), box.get("error")
    return join


def bounded(fn, *a, timeout=30.0, **kw):
    """fn(*a, **kw), failing the test after `timeout` s instead of hanging it."""
    finished, result, err = in_thread(fn, *a, **kw)(timeout)
    if not finished:
        pytest.fail(f"{getattr(fn, '__qualname__', fn)} did not return within {timeout:.0f}s (hung)")
    if err is not None:
        raise err
    return result


class Bounded:
    """Proxy whose method calls go through bounded() (attributes pass through)."""

    def __init__(self, obj, timeout=30.0):
        self._obj, self._timeout = obj, timeout

    def __getattr__(self, name):
        v = getattr(self._obj, name)
        if not callable(v):
            return v
        return lambda *a, **kw: bounded(v, *a, timeout=self._timeout, **kw)


def policies(engines):
    return [Bounded(e).call("info")["policy"] for e in engines]


def request(n_prompts=10, seed0=1000, cap=48, vary=True):
    prompts = [[7, 11 + i, 13 * i % 97, 5] for i in range(n_prompts)]
    return dict(prompts=prompts, params=dict(PARAMS),
                max_tokens=[cap - (i % 4) * 6 for i in range(n_prompts)],
                seeds=[seed0 + 17 * i for i in range(n_prompts)],
                n_per_prompt=[(i % 4) + 1 for i in range(n_prompts)] if vary else None, tag="t")


def expected(req, policy):
    counts = req["n_per_prompt"] or [req["params"]["n"]] * len(req["prompts"])
    return [[FW.sample(p, s + j, policy, mt, req["params"]) for j in range(n)]
            for p, s, mt, n in zip(req["prompts"], req["seeds"], req["max_tokens"], counts)]


def strip(outputs):
    return [[{k: v for k, v in rec.items() if k != "fake_engine"} for rec in group] for group in outputs]


def check_records(outputs, policy):
    for group in outputs:
        for rec in group:
            assert rec["token_ids"] and all(isinstance(t, int) for t in rec["token_ids"])
            assert len(rec["logprobs"]) == len(rec["token_ids"])
            assert all(math.isfinite(x) and x <= 0 for x in rec["logprobs"])
            assert rec["finish"] in ("stop", "length") and isinstance(rec["text"], str)
            assert "stop_reason" in rec and rec["policy"] == policy


class Service:
    """One node service process (fake engines), as a node's local-rank-0 trainer starts it."""

    def __init__(self, tmp, run, node_rank, engines, key, policy="P0", env=None):
        self.run, self.node_rank, self.n, self.key = run, node_rank, engines, key
        self.port = free_port()
        self.log_dir = os.path.join(tmp, f"logs_{run}_n{node_rank}")
        self.ctl_dir = os.path.join(tmp, f"ctl_{run}")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ctl_dir, exist_ok=True)
        cli = os.path.join(tmp, f"cli_{run}_n{node_rank}.json")
        with open(cli, "w") as f:
            json.dump(["--role", "sampler", "--model", "/nonexistent/fake_model", "--gpu-memory-utilization", "0.5",
                       "--max-model-len", "4096", "--policy-id", policy], f)
        self.cmd = SERVICE + ["--node-rank", str(node_rank), "--bind", "127.0.0.1",
                    "--port", str(self.port), "--engines", str(engines), "--run-name", run, "--log-dir", self.log_dir,
                    "--vllm-python", sys.executable, "--cli-json", cli, "--worker", FAKE, "--stagger-s", "0"]
        # FAKE_PDEATHSIG=0: the fake engines do not die with their parent by themselves, so these tests see whether
        # the service itself takes them down
        pypath = os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH", "")) if p)
        self.env = {**os.environ, "RL_SERVICE_KEY": key.hex(), "FAKE_CTL_DIR": self.ctl_dir, "FAKE_PDEATHSIG": "0",
                    "PYTHONPATH": pypath, **(env or {})}
        self.out = os.path.join(self.log_dir, "service.out")
        self.proc, self.pid = None, None

    @property
    def addr(self):
        return ("127.0.0.1", self.port)

    def start(self):
        with open(self.out, "ab") as f:
            self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=f, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        self.pid = self.proc.pid

    def log(self):
        try:
            with open(self.out) as f:
                return f.read()
        except OSError:
            return ""

    def alive(self):
        if self.proc is not None:
            return self.proc.poll() is None
        return pid_alive(self.pid)

    def wait_serving(self, timeout=30.0):
        assert wait_until(lambda: "serving" in self.log() or not self.alive(), timeout), "service did not start"
        assert "serving" in self.log(), f"service failed:\n{self.log()[-3000:]}"

    def control(self, timeout=10.0):
        return W.NodeServiceClient(self.node_rank, self.addr, self.key, timeout=timeout)

    def engine_clients(self, ctl):
        return [W.RemoteEngineClient(f"n{self.node_rank}e{g}", ctl, g, timeout=10.0) for g in range(self.n)]

    def engine_name(self, g):
        return f"rl_{self.run}_n{self.node_rank}_sampler{g}"

    def inject(self, g, **faults):
        with open(os.path.join(self.ctl_dir, f"{self.engine_name(g)}.json"), "w") as f:
            json.dump(faults, f)

    def clear(self, g):
        try:
            os.unlink(os.path.join(self.ctl_dir, f"{self.engine_name(g)}.json"))
        except FileNotFoundError:
            pass

    def force_cleanup(self):
        if self.pid and self.alive():
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


def kill_run(run):
    for pid in engine_pids(run) + service_pids(run):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    wait_until(lambda: not engine_pids(run) and not service_pids(run), 10)
    for s in sockets(run):
        try:
            os.unlink(s)
        except FileNotFoundError:
            pass


class Cluster:
    def __init__(self, run, key, svcs, ctls, engines):
        self.run, self.key, self.svcs, self.ctls, self.engines = run, key, svcs, ctls, engines
        self.graceful = True                     # teardown: shut down through the coordinator and check what is left

    def close(self):
        leftovers = None
        try:
            if self.graceful:
                for e in self.engines:           # what train.main's finally does: close() every engine
                    e.close()
                for s in self.svcs:
                    if s.proc is not None:
                        s.proc.wait(timeout=30)
                wait_until(lambda: not engine_pids(self.run) and not sockets(self.run), 10)
                leftovers = (engine_pids(self.run), service_pids(self.run), sockets(self.run))
        finally:
            for s in self.svcs:
                s.force_cleanup()
            kill_run(self.run)
        if leftovers is not None:
            assert leftovers == ([], [], []), f"graceful shutdown left {leftovers}"


@pytest.fixture
def cluster(tmp_path):
    """cluster(name, layout=(2, 2)): one node service per layout entry (that many engines), all started in parallel,
    a control client per node and a RemoteEngineClient per engine (node-major order, as the coordinator builds them)."""
    made = []

    def make(name, layout=(2, 2), policy="P0", env=None):
        run = f"{TOKEN}_{name}"
        key = os.urandom(16)
        svcs = [Service(str(tmp_path), run, r, n, key, policy, env) for r, n in enumerate(layout)]
        c = Cluster(run, key, svcs, [], [])
        made.append(c)
        for s in svcs:
            s.start()
        for s in svcs:
            s.wait_serving()
        c.ctls = [s.control() for s in svcs]
        c.engines = [e for s, ctl in zip(svcs, c.ctls) for e in s.engine_clients(ctl)]
        return c

    yield make
    errors = []
    for c in made:
        try:
            c.close()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
    if errors:
        raise errors[0]


# ---- tests -------------------------------------------------------------------------------------------------------------

def test_two_node_streaming_is_complete_and_placement_free(cluster):
    two = cluster("pair")                       # 2 nodes x 2 engines
    one = cluster("single", layout=(3,))        # 1 node x 3 engines: a different placement of the same requests
    infos = [e.wait_ready() for e in two.engines]
    assert [i["policy"] for i in infos] == ["P0"] * 4
    assert len({i["pid"] for i in infos}) == 4
    assert policies(two.engines) == ["P0"] * 4
    assert [i["cuda_visible_devices"] for i in infos] == ["0", "1", "0", "1"]   # one local GPU per engine and node

    req = request(n_prompts=10)
    want = expected(req, "P0")
    seen, lock = [], threading.Lock()

    def on_sample(i, j, rec):
        with lock:
            seen.append((i, j))

    m2 = Bounded(D.MultiEngine(two.engines, stream_seqs=2, policy="P0"))
    r2 = m2.call("generate", on_sample=on_sample, **req)
    pairs = [(i, j) for i, n in enumerate(req["n_per_prompt"]) for j in range(n)]
    assert sorted(seen) == pairs and len(set(seen)) == len(seen)          # every (prompt, sample) exactly once
    assert [len(g) for g in r2["outputs"]] == req["n_per_prompt"]
    check_records(r2["outputs"], "P0")
    assert strip(r2["outputs"]) == want                                   # each sample = f(prompt, seed + j, policy)
    used = {rec["fake_engine"] for g in r2["outputs"] for rec in g}
    assert used == {s.engine_name(g) for s in two.svcs for g in range(s.n)}   # all 4 engines on both nodes sampled
    assert r2["gen_tokens"] == sum(len(rec["token_ids"]) for g in r2["outputs"] for rec in g)

    m1 = Bounded(D.MultiEngine(one.engines, stream_seqs=3, policy="P0"))
    r1 = m1.call("generate", **req)
    assert strip(r1["outputs"]) == strip(r2["outputs"])                    # same results on one node, other placement
    assert {rec["fake_engine"] for g in r1["outputs"] for rec in g} == {one.svcs[0].engine_name(g) for g in range(3)}

    # --train-force-inline: a capped sample's forced answer comes back with it, the same on either placement
    force = {"suffix_ids": [9, 9], "max_model_len": 4096, "budget": 16, "seed_salt": 13,
             "params": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "logprobs": True}}
    fr2 = m2.call("generate", **dict(req, force=force))
    fr1 = m1.call("generate", **dict(req, force=force))
    assert strip(fr1["outputs"]) == strip(fr2["outputs"])
    capped = [rec for g in fr2["outputs"] for rec in g if rec["finish"] == "length"]
    assert capped and all(rec["forced"]["token_ids"] for rec in capped)
    assert not any("forced" in rec for g in fr2["outputs"] for rec in g if rec["finish"] != "length")
    assert sum(e["forced"] for e in fr2["dispatch"]["engines"]) == len(capped)

    # the non-streaming dispatchers over the same two nodes give the same records too
    for kw in ({"dispatch_prompts": 0}, {"dispatch_prompts": 2}):
        r = Bounded(D.MultiEngine(two.engines, policy="P0", **kw)).call("generate", **req)
        assert strip(r["outputs"]) == want
    # score / sleep / wake / free_host_cache / digest reach the remote engines unchanged
    sc = Bounded(D.MultiEngine(two.engines, policy="P0")).call("score", seqs=[[1, 2, 3, 4]], starts=[2])
    assert sc["logprobs"] == [FW.score_logprobs([1, 2, 3, 4], 2, "P0")]
    m2.call("sleep", level=1)
    with pytest.raises(RuntimeError, match="sleep"):
        m2.call("generate", **request(n_prompts=2))
    m2.call("wake", tags=["weights"])
    m2.call("wake", tags=["kv_cache"])
    m2.call("free_host_cache")
    assert len({Bounded(e).call("digest")["digests"]["embed_tokens.weight"] for e in two.engines}) == 1
    assert strip(m2.call("generate", **req)["outputs"]) == want

    # concurrent requests on one remote engine client are serialized, never interleaved on its connection
    e = two.engines[3]
    seqs = [[1, 2, 3 + k, 4, 5 + k % 3] for k in range(24)]
    joins = [in_thread(e.call, "score", seqs=[q], starts=[1]) for q in seqs]
    results = [j(15) for j in joins]
    assert all(f and err is None for f, _, err in results), [err for _, _, err in results if err]
    assert [r["logprobs"][0] for _, r, _ in results] == [FW.score_logprobs(q, 1, "P0") for q in seqs]


def test_dynamic_rounds_abort_across_nodes_keeps_engines_in_sync(cluster):
    two = cluster("rounds", env={"FAKE_STEP_S": "0.005"})
    rounds = [request(n_prompts=8, seed0=5000 + 1000 * r, vary=False) for r in range(4)]
    rest = iter(rounds[1:])
    judged = []

    def on_round(r, outs):
        judged.append(r)
        return r >= 1                                   # stop after round 1: rounds 2-3 are cleared / aborted

    m = Bounded(D.MultiEngine(two.engines, stream_seqs=3, policy="P0"))
    res = m.call("generate_rounds", more=lambda: next(rest, None), on_round=on_round, heartbeat_s=0.05, **rounds[0])
    dyn = res["dispatch"]["dyn"]
    assert dyn["stopped"] and dyn["rounds_included"] == 2 and judged[:2] == [0, 1]
    assert strip(res["outputs"]) == expected(rounds[0], "P0") + expected(rounds[1], "P0")
    # after the abort every engine is idle and in step with the coordinator: plain commands and a new stream work
    assert policies(two.engines) == ["P0"] * 4
    req = request(n_prompts=6)
    assert strip(m.call("generate", **req)["outputs"]) == expected(req, "P0")


def test_reload_commits_only_when_every_engine_acknowledges(cluster, tmp_path):
    two = cluster("reload")
    m = Bounded(D.MultiEngine(two.engines, stream_seqs=2, policy="P0"))
    v1 = tmp_path / "v1"
    v1.mkdir()
    r = m.call("reload", weights_path=str(v1), policy="P1")
    assert r["acks"] == 4 and m.committed_policy == "P1"
    assert policies(two.engines) == ["P1"] * 4
    req = request(n_prompts=8)
    out = m.call("generate", **req)["outputs"]
    check_records(out, "P1")
    assert strip(out) == expected(req, "P1") and strip(out) != expected(req, "P0")

    # one engine (node 1, engine 1) acknowledges another policy: nothing is committed
    two.svcs[1].inject(1, ack_policy="STALE")
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        m.call("reload", weights_path=str(v1), policy="P2")
    assert m.committed_policy is None
    assert policies(two.engines) == ["P2", "P2", "P2", "STALE"]

    # a coordinator that holds P2 rejects the stale engine's records (whole-batch check)
    with pytest.raises(RuntimeError, match="mixed-policy"):
        Bounded(D.MultiEngine(two.engines, stream_seqs=2, policy="P2")).call("generate", **req)

    # an engine that fails its reload (error reply): raises, nothing committed, the policy it reports is None
    two.svcs[1].clear(1)
    two.svcs[0].inject(0, reload_fail=True)
    with pytest.raises(RuntimeError, match="injected reload failure"):
        m.call("reload", weights_path=str(v1), policy="P3")
    assert m.committed_policy is None and policies(two.engines[:1]) == [None]

    # recovery: a clean reload on every engine commits again
    two.svcs[0].clear(0)
    m.call("reload", weights_path=str(v1), policy="P4")
    assert m.committed_policy == "P4"
    assert strip(m.call("generate", **req)["outputs"]) == expected(req, "P4")

    # the per-sample check (--reward-stream's on_sample) stops the stream at the first stale record
    two.svcs[1].inject(0, ack_policy="STALE")
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        m.call("reload", weights_path=str(v1), policy="P5")
    with pytest.raises(RuntimeError, match="mixed-policy"):
        Bounded(D.MultiEngine(two.engines, stream_seqs=2, policy="P5")).call("generate", on_sample=lambda *a: None,
                                                                             **req)
    # that engine was left mid-stream: its client refuses further requests instead of reading the stream's leftovers
    with pytest.raises(RuntimeError, match="closed"):
        Bounded(two.engines[2]).call("info")
    assert policies([two.engines[i] for i in (0, 1, 3)]) == ["P5"] * 3


def test_verify_dir_same_bytes_same_digest(cluster, tmp_path):
    two = cluster("verify", layout=(1, 1))
    c0, c1 = Bounded(two.ctls[0]), Bounded(two.ctls[1])
    a = tmp_path / "export_node0"
    (a / "sub").mkdir(parents=True)
    (a / "model-00001-of-00002.safetensors").write_bytes(os.urandom(300_000))
    (a / "model-00002-of-00002.safetensors").write_bytes(os.urandom(70_000))
    (a / "config.json").write_text(json.dumps({"x": 1}))
    (a / "sub" / "extra.bin").write_bytes(b"\x00" * 1000)
    b = tmp_path / "export_node1"
    shutil.copytree(a, b)
    (b / "model-00003.safetensors.part").write_bytes(b"in flight")      # partial transfers are not part of a version
    want = RI.dir_manifest(str(a))["sha256"]
    s0 = c0.call("verify_dir", path=str(a))
    s1 = c1.call("verify_dir", path=str(b), expect=s0["sha256"])
    assert s0["sha256"] == s1["sha256"] == want and s1["ok"] and s0["files"] == s1["files"] == 4
    data = bytearray((b / "model-00001-of-00002.safetensors").read_bytes())
    data[123_456] ^= 1                                                    # one bit
    (b / "model-00001-of-00002.safetensors").write_bytes(bytes(data))
    s1b = c1.call("verify_dir", path=str(b), expect=want)
    assert s1b["sha256"] != want and not s1b["ok"]
    assert c0.call("verify_dir", path=str(a), expect=want)["ok"]
    with pytest.raises(RuntimeError, match="no directory"):              # a missing directory is never "verified"
        c1.call("verify_dir", path=str(tmp_path / "missing"))


def test_bad_clients_are_rejected_and_the_service_keeps_serving(cluster):
    from multiprocessing.connection import AuthenticationError, Client
    c = cluster("auth", layout=(2,))
    svc = c.svcs[0]
    with pytest.raises(AuthenticationError):
        bounded(Client, svc.addr, family="AF_INET", authkey=os.urandom(16))
    with pytest.raises((AuthenticationError, EOFError, OSError)):
        bounded(W.NodeServiceClient, 0, svc.addr, b"wrong-key", timeout=2.0)
    with socket.create_connection(svc.addr) as s:                         # not a multiprocessing client at all
        s.sendall(b"GET / HTTP/1.0\r\n\r\n")
    silent = socket.create_connection(svc.addr)                           # connects and never answers the challenge
    try:
        ctl2 = Bounded(bounded(svc.control, 10.0, timeout=10))           # fails if the silent client blocks accepting
        assert ctl2.call("ping")["alive"] == [True, True]
        assert policies(c.engines) == ["P0", "P0"]
        with pytest.raises(RuntimeError, match="already connected"):     # one coordinator connection per engine
            bounded(W.RemoteEngineClient, "dup", ctl2._obj, 0, timeout=5.0)
        with pytest.raises(RuntimeError, match="unknown"):
            bounded(W.RemoteEngineClient, "bad", ctl2._obj, 7, timeout=5.0)
        assert Bounded(c.ctls[0]).call("ping")["ok"]
        with pytest.raises(RuntimeError, match="unknown control command"):
            Bounded(c.ctls[0]).call("reboot")
    finally:
        silent.close()


def test_engine_crash_surfaces_as_an_error_not_a_hang(cluster, tmp_path):
    two = cluster("crash")
    two.svcs[1].inject(0, crash_after_samples=2)                          # node 1, engine 0 dies mid-stream
    m = Bounded(D.MultiEngine(two.engines, stream_seqs=2, policy="P0"), timeout=20)
    with pytest.raises(RuntimeError, match="n1e0"):
        m.call("generate", **request(n_prompts=12))
    assert Bounded(two.ctls[1]).call("ping")["alive"] == [False, True]
    assert policies(two.engines[:2] + two.engines[3:]) == ["P0"] * 3     # the other engines still serve
    with pytest.raises(RuntimeError):                                     # the dead engine fails fast again
        Bounded(two.engines[2], timeout=15).call("info")
    # an engine dying during a reload: the reload raises and nothing is committed
    two.svcs[0].inject(1, crash_on="reload")
    v = tmp_path / "v"
    v.mkdir()
    with pytest.raises(RuntimeError):
        m.call("reload", weights_path=str(v), policy="P1")
    assert m.committed_policy is None
    assert Bounded(two.ctls[0]).call("ping")["alive"] == [True, False]


@pytest.mark.parametrize("how", ["shutdown", "kill_now", "sigterm", "sigkill_pdeathsig"])
def test_service_stop_takes_its_engines_down(cluster, how):
    env = {"FAKE_PDEATHSIG": "1"} if how == "sigkill_pdeathsig" else None
    c = cluster(f"stop_{how}", layout=(2,), env=env)
    svc = c.svcs[0]
    pids = [e.wait_ready()["pid"] for e in c.engines]
    assert all(pid_alive(p) for p in pids) and len(sockets(c.run)) == 2
    if how == "shutdown":
        bounded(c.ctls[0].shutdown)
        code = 0
    elif how == "kill_now":                    # MultiEngine's fail-fast path: kill_now on a remote engine
        bounded(c.engines[1].kill_now)
        code = 1
    elif how == "sigterm":
        os.kill(svc.pid, signal.SIGTERM)
        code = 128 + signal.SIGTERM
    else:                                      # train.main's fatal path: killpg(service, SIGKILL); the engines'
        os.killpg(svc.pid, signal.SIGKILL)     # PR_SET_PDEATHSIG (vllm_worker sets it too) takes them down
        code = -signal.SIGKILL
    assert svc.proc.wait(timeout=20) == code
    assert wait_until(lambda: not any(pid_alive(p) for p in pids), 10), "engines outlived their service"
    assert engine_pids(c.run) == []
    if how != "sigkill_pdeathsig":             # a SIGKILLed engine cannot unlink its socket
        assert sockets(c.run) == []
    with pytest.raises(RuntimeError):          # the coordinator sees an error, not a hang
        Bounded(c.engines[0], timeout=15).call("info")
    c.graceful = False


INTERMEDIATE = r"""
import json, os, subprocess, sys, time
cmd, out, when = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
pattern = os.environ["T2N_ENGINE_PATTERN"]     # not in argv: this process's own command line must not match it
with open(out, "ab") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
seen, t = [], time.time()
while time.time() - t < 20:
    if when == "serving" and "serving" in open(out).read():
        break
    seen = subprocess.run(["pgrep", "-f", pattern], capture_output=True).stdout.split()
    if when == "starting" and len(seen) == 2:
        break
    time.sleep(0.05)
print(json.dumps({"service": p.pid, "engines": [int(x) for x in seen], "log": open(out).read()}), flush=True)
os._exit(0)                                    # the "trainer" dies; the service is now an orphan
"""


@pytest.mark.parametrize("when", ["serving", "starting"])
def test_service_dies_with_its_parent(tmp_path, when):
    run = f"{TOKEN}_orphan_{when}"
    env = {"FAKE_START_DELAY": "4"} if when == "starting" else None      # engines still loading when the parent dies
    svc = Service(str(tmp_path), run, 0, 2, os.urandom(16), env=env)
    try:
        p = subprocess.run([sys.executable, "-c", INTERMEDIATE, json.dumps(svc.cmd), svc.out, when],
                           env={**svc.env, "T2N_ENGINE_PATTERN": f"[f]ake_vllm_worker\\.py.*/rl_{run}_n"},
                           capture_output=True, text=True, timeout=40)
        seen = json.loads(p.stdout)
        svc.pid = seen["service"]
        if when == "starting":                 # the parent died while both engines were up but still loading
            assert len(seen["engines"]) == 2 and "serving" not in seen["log"], seen
        else:
            assert "serving" in seen["log"], seen
        t = time.time()
        gone = wait_until(lambda: not pid_alive(svc.pid) and not engine_pids(run) and not sockets(run), 20)
        assert gone, (f"orphaned service alive={pid_alive(svc.pid)} engines={engine_pids(run)} "
                      f"sockets={sockets(run)}\n{svc.log()[-2000:]}")
        assert time.time() - t < 10
        assert all(not pid_alive(e) for e in seen["engines"])
        assert "parent trainer is gone" in svc.log()
        if when == "starting":
            assert "serving" not in svc.log()  # taken down during startup, not after it
    finally:
        svc.force_cleanup()
        kill_run(run)


def test_zz_nothing_left_behind():
    assert pgrep(f"[f]ake_vllm_worker\\.py.*/rl_{TOKEN}_") == []
    assert pgrep(f"[a]rcsolver\\.rl\\.node_service.*--run-name {TOKEN}_") == []
    assert glob.glob(f"/tmp/rl_{TOKEN}_*") == []
