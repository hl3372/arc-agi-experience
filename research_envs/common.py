"""Paths and machine-wide slots shared by both domains.

Where outputs go (every round, pilot and validation writes under ROOT/runs/):
- RE_ROOT, if set;
- otherwise the source checkout this package was installed from (the folder holding pyproject.toml), so runs/ sits
  next to the code and is git-ignored;
- otherwise the current working directory.
"""
import fcntl
import os
import time
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = PACKAGE_DIR / 'archive'


def _checkout():
    root = PACKAGE_DIR.parent
    return root if (root / 'pyproject.toml').is_file() and (root / 'research_envs').is_dir() else None


CHECKOUT = _checkout()
ROOT = Path(os.environ.get('RE_ROOT') or CHECKOUT or Path.cwd()).resolve()
EXAMPLES_DIR = (CHECKOUT / 'examples') if CHECKOUT else None


class FileSlots:
    """Machine-wide slots: flock on one of k lock files under ROOT/runs. Shared by every task process of every round
    that uses the same ROOT, so overlapping runs never oversubscribe the cores (wall-clock caps would quietly weaken
    the battery) or the model account. A lock is released when its holder exits, even if it crashes.

    One instance must not be shared by concurrent threads (it keeps the held lock on the instance); threads create
    their own instance, which is cheap."""

    def __init__(self, name, k):
        self.name, self.k = name, max(1, int(k))
        self.held = []

    def _paths(self):
        runs = ROOT / 'runs'
        return [runs / f'.slot-{self.name}-{i}' for i in range(self.k)]

    def __enter__(self):
        (ROOT / 'runs').mkdir(parents=True, exist_ok=True)
        while True:
            for path in self._paths():
                f = open(path, 'a')
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.held.append(f)
                    return self
                except BlockingIOError:
                    f.close()
            time.sleep(2)

    def __exit__(self, *exc):
        f = self.held.pop()
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
