"""Where the code lives and where its data goes.

The checkout holds code only. Everything a batch writes goes to the workspace:

  <workspace>/batches/   batch profiles, specs and allocations made by `bin/arcgen new`
  <workspace>/runs/      one folder per run (and per commit), plus `<name>-ops/` operation logs
  <workspace>/bank/      the task bank (registry, task files, dossiers), unless ARCGEN_BANK points elsewhere
  <workspace>/cache/     embeddings shared across runs, so the bank is embedded once
  <workspace>/gate/      the offline test gate: the code hash that last passed the test suite

ARCGEN_WORKSPACE moves the workspace (default: `workspace/` in the checkout, git-ignored).
ARCGEN_BANK points at a bank folder anywhere, for example a downloaded copy of a released bank.
"""
from pathlib import Path
import os

PACKAGE = Path(__file__).resolve().parent          # the arcgen package
REPO = PACKAGE.parent                              # the checkout
WORKSPACE = Path(os.environ.get('ARCGEN_WORKSPACE') or REPO / 'workspace').expanduser().resolve()
BANK = Path(os.environ.get('ARCGEN_BANK') or WORKSPACE / 'bank').expanduser().resolve()
RUNS = WORKSPACE / 'runs'
BATCHES = WORKSPACE / 'batches'
CACHE = WORKSPACE / 'cache'
GATE = WORKSPACE / 'gate'  # the offline test gate: which code passed the suite (bin/arcgen test)


def rel(path):
    """A path as stored in run records: relative to the workspace when inside it, else absolute.

    `WORKSPACE / rel(p)` is `p` in both cases, so stored paths round-trip."""
    path = Path(path).resolve()
    return str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)


def resolve(value):
    """A stored or configured path: absolute as given, else relative to the workspace, else to the checkout."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if (WORKSPACE / path).exists() or not (REPO / path).exists():
        return WORKSPACE / path
    return REPO / path


def isolated(root, error=ValueError):
    """A run folder resolved, refused when it overlaps the code or the bank (runs never write into either)."""
    root = Path(root).resolve()
    for protected in (PACKAGE, BANK):
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise error('run output overlaps the code or the bank: ' + str(root))
    return root
