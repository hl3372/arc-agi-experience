"""Host-memory sampling for the RL trainer: pinned host memory (engine weight backups, offloaded trainer state) plus a
checkpoint's transients can exhaust host RAM, and a node without swap then livelocks. Rank 0 samples MemAvailable every
few seconds and reports per-phase values and the step minimum as metrics, so headroom trends are visible long before a
crunch. Pure python (reads /proc/meminfo)."""
from __future__ import annotations

import threading


def mem_available_gb(path: str = "/proc/meminfo") -> float:
    """MemAvailable in GiB (nan when unreadable)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    return float("nan")


class HostMemMonitor:
    """A daemon thread tracking the minimum MemAvailable since the last take(); mark(name) records the current
    value under a phase name. take() returns {"host/avail_min_gb": ..., "host/avail_<name>_gb": ...} and resets."""

    def __init__(self, interval: float = 2.0, path: str = "/proc/meminfo"):
        self.interval, self.path = interval, path
        self._lock = threading.Lock()
        self._min = float("inf")
        self._marks = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hostmem", daemon=True)
        self._thread.start()

    def _observe(self) -> float:
        v = mem_available_gb(self.path)
        if v == v:                                  # not nan
            with self._lock:
                self._min = min(self._min, v)
        return v

    def _run(self):
        while not self._stop.wait(self.interval):
            self._observe()

    def mark(self, name: str) -> float:
        v = self._observe()
        with self._lock:
            self._marks[name] = v
        return v

    def take(self) -> dict:
        self._observe()
        with self._lock:
            out = {f"host/avail_{k}_gb": v for k, v in self._marks.items()}
            if self._min != float("inf"):
                out["host/avail_min_gb"] = self._min
            self._min, self._marks = float("inf"), {}
        return out

    def stop(self):
        self._stop.set()
