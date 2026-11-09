"""Client side of vllm_worker.py: spawn an engine process in the vLLM venv and call it.

Pure python (stdlib only), used from the training venv. The engine is started in its own session
(process group) so that close() can always clean up exactly the processes this trainer started.
The engine authenticates its socket with a per-engine random key passed in $RL_ENGINE_AUTHKEY.
"""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import tempfile
import time
from multiprocessing.connection import Client

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "vllm_worker.py")
# Optional engine start/stop lock (flock(2) on this file, shared by every process that uses the same path): an engine
# that profiles memory while another process allocates or frees GPU memory mis-sizes its KV cache or aborts, so every
# engine start (until ready) and stop (until its memory is freed) on a shared GPU can hold this lock. The trainer
# passes lock_path=None (engines on different GPUs need no lock); $ARC_ENGINE_LOCK overrides the default path.
ENGINE_LOCK = os.environ.get("ARC_ENGINE_LOCK", os.path.join(tempfile.gettempdir(), "arc_rl_engine_start.lock"))


class EngineLock:
    def __init__(self, path: str | None, settle: float = 10.0):
        self.path, self.settle, self.f = path, settle, None

    def acquire(self, who: str) -> float:
        if not self.path or self.f is not None:
            return 0.0
        self.f = open(self.path, "w")
        t = time.time()
        fcntl.flock(self.f, fcntl.LOCK_EX)
        waited = time.time() - t
        print(f"[engine-lock] {who}: acquired after {waited:.0f}s", flush=True)
        return waited

    def release(self, who: str, settle: bool = True) -> None:
        if self.f is None:
            return
        if settle and self.settle:
            time.sleep(self.settle)          # let allocations settle first
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        self.f = None
        print(f"[engine-lock] {who}: released", flush=True)


def engine_env(vllm_python: str) -> dict:
    """Environment for the vLLM venv: its bin first on PATH (vLLM's JIT needs its ninja), none of
    the training venv's python variables, and vLLM's default CUDA allocator (the trainer's
    expandable_segments setting is left to the trainer)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "RL_ENGINE_AUTHKEY", "PYTORCH_CUDA_ALLOC_CONF")}
    env["PATH"] = os.path.dirname(vllm_python) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = os.path.dirname(os.path.dirname(vllm_python))
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    # vllm_worker's "digest" command sends a function through collective_rpc; vLLM 0.30 only pickles
    # callables with this set. The IPC is local to the engine's own processes.
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    return env


class EngineClient:
    def __init__(self, name: str, vllm_python: str, cli_args: list, log_path: str, socket_path: str,
                 env_script: str | None = None, lock_path: str | None = ENGINE_LOCK, extra_env: dict | None = None,
                 worker: str | None = None):
        """env_script: an optional shell file sourced before exec'ing the engine (e.g. to point CUDA_HOME at a
        CUDA toolkit FlashInfer can JIT-compile with on a new GPU architecture). exec keeps the engine's pid =
        the session leader we may clean up.
        lock_path: the shared engine start/stop lock, held from spawn until wait_ready() returns and
        around close() until the engine's processes are gone (None = no lock)."""
        self.name, self.log_path, self.socket_path = name, log_path, socket_path
        self.lock = EngineLock(lock_path)
        self.authkey = os.urandom(16)
        env = engine_env(vllm_python)
        env.update(extra_env or {})                   # e.g. CUDA_VISIBLE_DEVICES for one engine per GPU
        env["RL_ENGINE_AUTHKEY"] = self.authkey.hex()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log = open(log_path, "ab")
        # -P: the worker's directory (this package) is not put on the engine's sys.path, so none of its module names
        # (config, data, core, ...) can shadow a package vLLM imports; the worker imports nothing from it
        cmd = ([vllm_python, worker, *map(str, cli_args)] if worker                    # worker: tests only
               else [vllm_python, "-P", WORKER, *map(str, cli_args)]) + ["--socket", socket_path]
        if env_script:
            if not os.path.exists(env_script):
                raise FileNotFoundError(f"vLLM env script {env_script} not found")
            cmd = ["bash", "-c", 'source "$0" && exec "$@"', env_script, *cmd]
        self.conn = None
        self.lock.acquire(f"start {name}")
        try:
            self.proc = subprocess.Popen(cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        except BaseException:
            self.lock.release(f"start {name} (failed)", settle=False)
            raise

    def log_tail(self, n: int = 40) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 20000))
                return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-n:])
        except OSError:
            return "(no log)"

    def _check_alive(self):
        if self.proc.poll() is not None:
            raise RuntimeError(f"{self.name} engine exited with code {self.proc.returncode}; "
                               f"log {self.log_path}:\n{self.log_tail()}")

    def wait_ready(self, timeout: float = 3600.0) -> dict:
        """Block until the engine listens (its LLM() init, memory profiling and KV allocation are
        done), then release the start lock. On failure the engine is torn down under the lock."""
        t0 = time.time()
        try:
            while not os.path.exists(self.socket_path):
                self._check_alive()
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"{self.name} engine not ready after {timeout:.0f}s; log {self.log_path}")
                time.sleep(2.0)
            self.conn = Client(self.socket_path, family="AF_UNIX", authkey=self.authkey)
            info = self.call("info")
        except BaseException:
            self._teardown()
            self.lock.release(f"start {self.name} (failed)")
            raise
        self.lock.release(f"start {self.name}")
        return info

    def call(self, cmd: str, **kw):
        if self.conn is None:
            raise RuntimeError(f"{self.name} engine not connected")
        self.conn.send({"cmd": cmd, **kw})
        while not self.conn.poll(5.0):
            self._check_alive()
        reply = self.conn.recv()
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"{self.name} engine failed on {cmd}:\n{reply['error']}")
        return reply

    def stream(self, cmd: str, on_message, **kw):
        """A command that talks back before its final reply (vllm_worker's "stream"): every intermediate message
        goes to on_message, whose non-None return value is sent to the engine; the {"type": "end"} reply ends it."""
        if self.conn is None:
            raise RuntimeError(f"{self.name} engine not connected")
        self.conn.send({"cmd": cmd, **kw})
        while True:
            while not self.conn.poll(5.0):
                self._check_alive()
            reply = self.conn.recv()
            if isinstance(reply, dict) and "error" in reply:
                raise RuntimeError(f"{self.name} engine failed on {cmd}:\n{reply['error']}")
            if reply.get("type") == "end":
                return reply
            answer = on_message(reply)
            if answer is not None:
                self.conn.send(answer)

    def kill_now(self) -> None:
        """Fail-fast cleanup (no shutdown handshake, no lock): SIGKILL the engine's whole process group."""
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _group_alive(self) -> bool:
        try:
            os.killpg(self.proc.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _teardown(self, timeout: float = 120.0) -> None:
        """Shut the engine down and wait until every process of its group (the engine core and GPU
        worker are descendants of our child, in its own session) has exited, i.e. its GPU memory
        is freed. Only our own child's process group is ever signalled."""
        try:
            if self.conn is not None and self.proc.poll() is None:
                self.conn.send({"cmd": "shutdown"})
                if self.conn.poll(30.0):
                    self.conn.recv()
        except (OSError, EOFError):
            pass
        self.conn = None
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        for sig, wait in ((None, 60.0), (signal.SIGTERM, 60.0), (signal.SIGKILL, 30.0)):
            if sig is not None and self._group_alive():
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    pass
            t = time.time()
            while self._group_alive() and time.time() - t < wait:
                self.proc.poll()                 # reap our child so it does not linger as a zombie
                time.sleep(1.0)
            if not self._group_alive():
                break
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def close(self, timeout: float = 120.0) -> None:
        self.lock.acquire(f"stop {self.name}")
        try:
            self._teardown(timeout)
        finally:
            self.lock.release(f"stop {self.name}")
            if not self._log.closed:
                self._log.close()


# ---- multi-node sampling: engines owned by a node service (node_service.py), reached over TCP ------------------------

def _keepalive(conn, idle: int = 30, interval: int = 10, count: int = 6) -> None:
    """A dead peer (a node gone) surfaces as a broken connection after ~idle + interval * count seconds."""
    import socket
    try:
        s = socket.socket(fileno=os.dup(conn.fileno()))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, v in (("TCP_KEEPIDLE", idle), ("TCP_KEEPINTVL", interval), ("TCP_KEEPCNT", count)):
            if hasattr(socket, opt):
                s.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), v)
        s.close()                                   # closes the dup only; the options stay on the shared socket
    except OSError:
        pass


def _connect(address, authkey: bytes, hello: dict, timeout: float):
    from multiprocessing.connection import Client
    t0 = time.time()
    while True:
        try:
            conn = Client(address, family="AF_INET", authkey=authkey)
            break
        except (ConnectionRefusedError, OSError):
            if time.time() - t0 > timeout:
                raise TimeoutError(f"node service {address} not reachable after {timeout:.0f}s")
            time.sleep(5.0)
    _keepalive(conn)
    conn.send(hello)
    return conn


class NodeServiceClient:
    """The coordinator's control channel to one node service (thread-safe)."""

    def __init__(self, node_rank: int, address, authkey: bytes, timeout: float = 3600.0):
        import threading
        self.node_rank, self.address, self.authkey = node_rank, tuple(address), authkey
        self.conn = _connect(self.address, authkey, {"hello": "control"}, timeout)
        self.lock = threading.Lock()
        self.closed = False

    def call(self, cmd: str, **kw):
        with self.lock:
            self.conn.send({"cmd": cmd, **kw})
            reply = self.conn.recv()
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"node {self.node_rank} service failed on {cmd}: {reply['error']}")
        return reply

    def infos(self, timeout: float = 3600.0) -> list:
        return self.call("info")["infos"]

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.call("shutdown")
        except (OSError, EOFError, RuntimeError):
            pass

    def kill(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            with self.lock:
                self.conn.send({"cmd": "kill"})
        except (OSError, EOFError):
            pass


class RemoteEngineClient:
    """EngineClient's interface for engine `index` of a node service: the same call / stream / close / kill_now, so
    MultiEngine and the streaming dispatch cannot tell a remote engine from a local one."""

    def __init__(self, name: str, service: NodeServiceClient, index: int, timeout: float = 3600.0):
        import threading
        self.name, self.service, self.index = name, service, index
        self.log_path = f"(node {service.node_rank}: engine {index}'s log on that node)"
        # one request at a time on the connection: when one engine's error ends a MultiEngine call,
        # the other engines' requests are still in flight, and the next call on the same client then read their replies
        # (or garbled frames read by two threads at once)
        self._lock = threading.Lock()
        self.conn = _connect(service.address, service.authkey, {"hello": "engine", "index": index}, timeout)
        first = self.conn.recv()
        if "error" in first:
            raise RuntimeError(f"{name}: {first['error']}")
        self._info = first.get("info") or {}

    def log_tail(self, n: int = 40) -> str:
        return self.log_path

    def _check_alive(self):
        if self.conn is None or self.conn.closed:
            raise RuntimeError(f"{self.name}: connection to node {self.service.node_rank} closed")

    def _abandon(self):
        """A command ended without its final reply (its engine is still in it): later replies on this connection
        would belong to it, so the connection is closed and every later call fails instead."""
        try:
            self.conn.close()
        except OSError:
            pass

    def wait_ready(self, timeout: float = 3600.0) -> dict:
        return self._info

    def _node_topk(self, kw):
        """Top-k arrays land on this engine's node: write them under <topk_dir>/n<node>, so a rollout's
        path names the node whose ranks can read it (train.topk_node / plan_minibatches(local_ws))."""
        if kw.get("topk_dir"):
            kw = dict(kw, topk_dir=os.path.join(kw["topk_dir"], f"n{self.service.node_rank}"))
        return kw

    def call(self, cmd: str, **kw):
        kw = self._node_topk(kw)
        with self._lock:
            self._check_alive()
            done = False
            try:
                self.conn.send({"cmd": cmd, **kw})
                while not self.conn.poll(5.0):
                    pass
                reply = self.conn.recv()
                done = True
            except (EOFError, OSError) as e:
                raise RuntimeError(f"{self.name} (node {self.service.node_rank}) lost during {cmd}: {e!r}") from e
            finally:
                if not done:
                    self._abandon()
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"{self.name} engine failed on {cmd}:\n{reply['error']}")
        return reply

    def stream(self, cmd: str, on_message, **kw):
        kw = self._node_topk(kw)
        with self._lock:
            self._check_alive()
            done = False
            try:
                self.conn.send({"cmd": cmd, **kw})
                while True:
                    while not self.conn.poll(5.0):
                        pass
                    reply = self.conn.recv()
                    if isinstance(reply, dict) and "error" in reply:
                        done = True                    # the engine ended the command itself
                        raise RuntimeError(f"{self.name} engine failed on {cmd}:\n{reply['error']}")
                    if reply.get("type") == "end":
                        done = True
                        return reply
                    answer = on_message(reply)         # may raise (e.g. a mixed-policy record): the engine is mid-stream
                    if answer is not None:
                        self.conn.send(answer)
            except (EOFError, OSError) as e:
                raise RuntimeError(f"{self.name} (node {self.service.node_rank}) lost during {cmd}: {e!r}") from e
            finally:
                if not done:
                    self._abandon()

    def kill_now(self) -> None:
        self.service.kill()

    def close(self, timeout: float = 120.0) -> None:
        try:
            if self.conn is not None and not self.conn.closed:
                self.conn.close()
        except OSError:
            pass
        self.service.shutdown()
