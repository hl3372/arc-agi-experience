"""Node-local sampler service: multi-node RL without remote CUDA ordinals or shared /dev/shm paths.

Each node's local-rank-0 trainer starts one of these (a CPU process, the trainer venv's python). It owns the node's
engines (workers.EngineClient, one per local GPU, CUDA_VISIBLE_DEVICES = the local index, AF_UNIX sockets) and serves
them to the step coordinator (global rank 0) over authenticated TCP (multiprocessing.connection, a per-run secret from
the environment variable RL_SERVICE_KEY, never written to disk or logs):

  - an "engine" connection (first message {"hello": "engine", "index": g}) is bridged to local engine g: every message
    the coordinator sends goes to the engine and every reply / intermediate stream message comes back, unchanged, so the
    coordinator's RemoteEngineClient behaves exactly like a local EngineClient (generate / stream / score / sleep /
    wake / reload / digest ...). One coordinator connection per engine at a time.
  - a "control" connection ({"hello": "control"}): info (every engine's ready info), verify_dir (identity.py
    dir_manifest of a local directory, compared with an expected digest), ping, shutdown (orderly engine close),
    kill (SIGKILL every engine's process group).

The service never outlives its trainer: it watches its parent pid and kills its engines when the parent is gone (or on
SIGTERM). It writes nothing outside its log directory.

    RL_SERVICE_KEY=<hex> python -m arcsolver.rl.node_service --node-rank N --bind ADDR --port P --engines 8 --run-name RUN
        --log-dir DIR --vllm-python PY --cli-json FILE [--env-script S] [--worker W] [--parent-pid PID]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import signal
import socket
import threading
import time
from multiprocessing.connection import Listener, answer_challenge, deliver_challenge

from arcsolver.rl import workers


def log(msg):
    print(f"[node-service {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def close_hard(conn):
    """Close a connection that another thread may be blocked reading: shutdown() first. A bare
    close() of the fd leaves the socket open while that read holds it (no FIN), so the peer would wait forever."""
    try:
        s = socket.socket(fileno=os.dup(conn.fileno()))
        try:
            s.shutdown(socket.SHUT_RDWR)
        finally:
            s.close()
    except (OSError, ValueError):
        pass
    try:
        conn.close()
    except OSError:
        pass


class Service:
    def __init__(self, a):
        self.a = a
        self.engines, self.infos = [], []
        self.busy = set()                      # engine indices with a coordinator connection
        self.bridges = {}                      # g -> (coordinator connection, its bridge's done event)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.spawn_lock = threading.RLock()    # kill_all vs start_engines (RLock: kill_all also runs in a signal handler)
        self.dying = False

    # ---- engines ------------------------------------------------------------------------------------------------
    def start_engines(self):
        with open(self.a.cli_json) as f:
            cli = json.load(f)
        for g in range(self.a.engines):
            with self.spawn_lock:              # no engine is started after kill_all
                if self.dying:
                    raise RuntimeError("the service is being killed")
                self.engines.append(workers.EngineClient(
                    f"n{self.a.node_rank}s{g}", self.a.vllm_python, cli,
                    os.path.join(self.a.log_dir, f"n{self.a.node_rank}_sampler{g}.log"),
                    f"/tmp/rl_{self.a.run_name}_n{self.a.node_rank}_sampler{g}.sock",
                    env_script=self.a.env_script or None, lock_path=None,
                    extra_env={"CUDA_VISIBLE_DEVICES": str(g)}, worker=self.a.worker or None))
            time.sleep(self.a.stagger_s)
        with cf.ThreadPoolExecutor(max_workers=max(1, len(self.engines))) as ex:
            self.infos = list(ex.map(lambda e: e.wait_ready(), self.engines))
        log(f"{len(self.engines)} engines ready")

    def kill_all(self):
        with self.spawn_lock:
            self.dying = True
            engines = list(self.engines)
        for e in engines:
            e.kill_now()
        self.remove_sockets(engines)

    def close_all(self):
        with cf.ThreadPoolExecutor(max_workers=max(1, len(self.engines))) as ex:
            list(ex.map(lambda e: e.close(), self.engines))
        self.remove_sockets(self.engines)

    @staticmethod
    def remove_sockets(engines):
        """The engines' AF_UNIX socket files (a killed or crashed engine cannot remove its own)."""
        for e in engines:
            try:
                os.unlink(e.socket_path)
            except OSError:
                pass

    # ---- connections --------------------------------------------------------------------------------------------
    def bridge(self, conn, g):
        """Coordinator <-> local engine g, message for message, until either side goes away."""
        e = self.engines[g]
        done = threading.Event()

        def down():                            # coordinator -> engine
            try:
                while not done.is_set():
                    msg = conn.recv()
                    e.conn.send(msg)
            except (EOFError, OSError, ConnectionResetError):
                pass
            done.set()

        with self.lock:
            self.bridges[g] = (conn, done)
        t = threading.Thread(target=down, daemon=True)
        t.start()
        try:                                   # engine -> coordinator
            while not done.is_set():
                if e.conn.poll(1.0):
                    conn.send(e.conn.recv())
                elif e.proc.poll() is not None:
                    log(f"engine {g} exited with code {e.proc.returncode}: closing its connection")
                    break
        except (EOFError, OSError, ConnectionResetError):
            pass
        finally:                               # whatever ended the bridge, the coordinator must see its connection end
            done.set()
            close_hard(conn)                   # wakes down()'s recv and sends the coordinator EOF
            with self.lock:
                self.busy.discard(g)
                self.bridges.pop(g, None)

    def detach_bridges(self, timeout=10.0):
        """End every coordinator <-> engine bridge before the engines are closed: an attached bridge reads the same
        engine connection as EngineClient.close() (it would take the shutdown reply and die on the connection
        close() had dropped, stalling the shutdown 30 s+ per engine)."""
        with self.lock:
            attached = list(self.bridges.values())
        for conn, done in attached:
            done.set()
            close_hard(conn)
        t = time.time()
        while time.time() - t < timeout:
            with self.lock:
                if not self.bridges:
                    return
            time.sleep(0.05)
        log(f"bridges {sorted(self.bridges)} did not end within {timeout:.0f}s")

    def control(self, conn):
        from arcsolver.rl import identity as RI
        try:
            while True:
                msg = conn.recv()
                cmd = msg.get("cmd")
                try:
                    if cmd == "info":
                        reply = {"node_rank": self.a.node_rank, "infos": self.infos}
                    elif cmd == "ping":
                        reply = {"ok": True, "alive": [e.proc.poll() is None for e in self.engines]}
                    elif cmd == "verify_dir":
                        t = time.time()
                        if not os.path.isdir(msg["path"]):     # an empty manifest must never pass as a verification
                            raise FileNotFoundError(f"no directory {msg['path']}")
                        m = RI.dir_manifest(msg["path"])
                        reply = {"sha256": m["sha256"], "files": len(m["files"]), "seconds": time.time() - t,
                                 "ok": msg.get("expect") in (None, m["sha256"])}
                    elif cmd == "shutdown":
                        conn.send({"ok": True})
                        self.stop.set()
                        return
                    elif cmd == "kill":
                        self.kill_all()
                        conn.send({"ok": True})
                        os._exit(1)
                    else:
                        reply = {"error": f"unknown control command {cmd!r}"}
                except Exception as ex:  # noqa: BLE001
                    reply = {"error": repr(ex)}
                conn.send(reply)
        except (EOFError, OSError, ConnectionResetError):
            pass

    def handle(self, conn, key):
        """Authenticate one connection and serve it, in its own thread (in the accept loop, a client that connected
        and never answered the challenge would block every later connection)."""
        try:
            deliver_challenge(conn, key)       # what Listener(authkey=key).accept() does, before anything is unpickled
            answer_challenge(conn, key)
            hello = conn.recv()
            kind = hello.get("hello")
        except Exception as ex:  # noqa: BLE001 - a failed handshake (bad key) must not stop the service
            log(f"rejected a connection: {ex!r}")
            close_hard(conn)
            return
        if kind == "control":
            self.control(conn)
        elif kind == "engine":
            g = int(hello.get("index", -1))
            with self.lock:
                ok = 0 <= g < len(self.engines) and g not in self.busy
                if ok:
                    self.busy.add(g)
            if not ok:
                conn.send({"error": f"engine {g} unknown or already connected"})
                conn.close()
                return
            conn.send({"ok": True, "info": self.infos[g]})
            self.bridge(conn, g)
        else:
            conn.close()

    def serve(self):
        key = bytes.fromhex(os.environ["RL_SERVICE_KEY"])
        listener = Listener((self.a.bind, self.a.port), backlog=64)    # authenticated per connection in handle()
        log(f"serving {len(self.engines)} engines on {self.a.bind}:{self.a.port}")

        def accept_loop():
            while not self.stop.is_set():
                try:
                    conn = listener.accept()
                except OSError as ex:
                    log(f"accept failed: {ex!r}")
                    time.sleep(0.5)
                    continue
                threading.Thread(target=self.handle, args=(conn, key), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        while not self.stop.is_set():
            if any(e.proc.poll() is not None for e in self.engines):
                dead = [g for g, e in enumerate(self.engines) if e.proc.poll() is not None]
                log(f"engines {dead} exited; the service stays up to report it (coordinators see closed connections)")
                time.sleep(5)
            self.stop.wait(2.0)
        log("shutdown requested: closing the engines")
        self.detach_bridges()
        self.close_all()
        listener.close()


def watch_parent(svc, parent, ppid0):
    """Never outlive the trainer: once the parent is gone (this process was re-parented, or an explicit --parent-pid
    that is not our parent no longer exists), kill the engines and exit. Runs from the start (checked only after the
    engines were ready, a trainer that died while they loaded would leave the service and engines running)."""
    while True:
        if parent == ppid0:
            gone = os.getppid() != parent
        else:
            try:
                os.kill(parent, 0)
                gone = False
            except ProcessLookupError:
                gone = True
            except PermissionError:
                gone = False
        if gone:
            log("parent trainer is gone: killing the engines")
            svc.kill_all()
            os._exit(1)
        time.sleep(1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-rank", type=int, required=True)
    ap.add_argument("--bind", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--engines", type=int, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--vllm-python", required=True)
    ap.add_argument("--cli-json", required=True, help="JSON list: the engines' vllm_worker arguments")
    ap.add_argument("--env-script", default="")
    ap.add_argument("--worker", default="", help="tests: a stand-in for vllm_worker.py")
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--stagger-s", type=float, default=5.0, help="pause between engine starts")
    a = ap.parse_args(argv)
    os.makedirs(a.log_dir, exist_ok=True)
    svc = Service(a)

    def on_term(signum, _frame):
        svc.kill_all()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    ppid0 = os.getppid()
    parent = a.parent_pid or ppid0
    if parent != 1:
        threading.Thread(target=watch_parent, args=(svc, parent, ppid0), daemon=True).start()
    try:
        svc.start_engines()
        svc.serve()
    except BaseException:
        import traceback
        log("fatal:\n" + traceback.format_exc())
        svc.kill_all()
        os._exit(1)


if __name__ == "__main__":
    main()
