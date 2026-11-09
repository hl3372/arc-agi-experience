"""A small synthetic bank for tests, and the run inputs frozen from it (arcgen.batch.bank_inputs)."""
from pathlib import Path
from . import commit_fixtures as fx


def make(root, n=40, seed=0):
    """A bank of n synthetic tasks at <root>/committed_tasks (created once)."""
    bank = Path(root) / 'committed_tasks'
    if not (bank / 'active_registry.json').exists():
        fx.make_bank(root, n, seed=seed, render=False, blob=64)
    return bank


def prepare(run_root, n=40, seed=0):
    """Freeze the run inputs of `run_root` from a synthetic bank beside it (never inside the run)."""
    from arcgen.batch import bank_inputs
    run_root = Path(run_root)
    bank = make(run_root.with_name(run_root.name + '-bank'), n, seed)
    return bank_inputs.prepare(run_root, bank=bank)
