"""Batch identity and shared helpers, bound to the batch config (see config.py)."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from arcgen.astra_usage_diversity_20260913 import common as base
from .routing import POLICY, DEFAULT_ROLES, ASTRA, SOL
from .config import CONFIG, SCIENCE, OPERATIONS, science_hash, ACCOUNT, codex_executable

ROOT = base.ROOT
HERE = Path(__file__).resolve().parent
REVISION = CONFIG['revision']
HOMES = dict(OPERATIONS['accounts'])  # {ACCOUNT: Codex home}: the one configured account
REVIEW_CALLS_PER_GROUP = int(round(SCIENCE['review_calls_per_task'] * OPERATIONS['group_size']))
CALL_HOMES = dict(HOMES)  # every model call (ideas, selection, development) uses this account
CARDS = HERE / 'cards'
SOURCE, component = base.SOURCE, base.component
SETTINGS = deepcopy(base.SETTINGS)
IDEA_MODELS = tuple(SCIENCE['models'][k] for k in SCIENCE['idea_efforts'])  # every model with idea efforts
SETTINGS.update(accounts=sorted(HOMES), allowed_accounts=sorted(HOMES), idea_models=list(IDEA_MODELS))
SETTINGS['role_overrides'] = deepcopy(DEFAULT_ROLES)
SETTINGS['author_pinning']['count'] = SCIENCE['author_pinning_count']
SETTINGS['selector']['accepted_neighbors_per_candidate'] = SCIENCE['selector']['accepted_neighbors_per_candidate']
# The selector shows bank neighbours only: the reference corpus holds accepted bank tasks and nothing else.
SETTINGS['selector']['real_neighbors_per_candidate'] = 0
SETTINGS['selector']['prompt_token_budget'] = SCIENCE['selector']['prompt_token_budget']
# Floor for a window whose selector prompt is over budget (supply.selector_inputs); without it k never drops.
SETTINGS['selector']['minimum_accepted_neighbors'] = SCIENCE['selector'].get('minimum_accepted_neighbors',
                                                                         SCIENCE['selector']['accepted_neighbors_per_candidate'])
read, ref, sha, frozen, atomic, digest, canonical, verify_refs = (
    getattr(base, n) for n in ('read','ref','sha','frozen','atomic','digest','canonical','verify_refs'))
IntegrityError, BudgetError, Budget = base.IntegrityError, base.BudgetError, base.Budget
isolated_root = base.isolated_root


PACKAGE = HERE.parent


def code_manifest():
    """The whole consolidated package (library, layers, prompts, cards); configs are bound separately."""
    return [ref(p) for p in sorted(PACKAGE.rglob('*')) if p.is_file()
            and p.suffix in ('.py', '.md', '.json', '.yaml')
            and not any(v in p.relative_to(PACKAGE).parts for v in ('evidence', '__pycache__', '.pytest_cache', 'profiles', 'tests', 'testsupport'))]


def code_identity():
    return digest(code_manifest())


TIMEOUT_SECONDS = OPERATIONS['timeout_seconds']


@lru_cache(None)
def _executable():
    return codex_executable(OPERATIONS)


def executable():
    """{'path', 'sha256'} of the Codex executable, found once per process; the launch manifest binds it."""
    found = _executable()
    return {'path': found['path'], 'sha256': found['sha256']}


def verify_executable():
    """The Codex executable exists, matches a configured codex_sha256 and still has the bytes this process found."""
    found = executable()
    if found['sha256'] is None:
        raise IntegrityError('Codex executable not found: ' + found['path'] + ' (install the Codex CLI or set operations.codex_executable)')
    if _executable()['pinned_sha256'] and _executable()['pinned_sha256'] != found['sha256']:
        raise IntegrityError('Codex executable does not match operations.codex_sha256: ' + found['path'])
    verify_refs(found)
    return found


def config_for(root, account=ACCOUNT, *, seed=2026091452, mock=False):
    if account not in HOMES: raise IntegrityError('account is not configured for this batch: ' + str(account))
    conf = base.config_for(root, account, seed=seed, mock=mock)
    if not mock:
        verify_executable()
    conf['transport']['executable'] = executable()['path']
    conf['transport']['timeout_seconds'] = TIMEOUT_SECONDS
    conf['_paths']['prompts'] = HERE / 'prompts'
    conf['paths']['prompts'] = str(HERE / 'prompts')
    for purpose, binding in DEFAULT_ROLES.items(): conf['role_profile'][purpose].update(binding)
    conf['_profile_hash'] = digest(conf['role_profile'])
    conf['run']['replicate'] = REVISION
    conf['_run_identity']['replicate'] = REVISION
    conf['_experiment_profile'].update(id=REVISION, runtime_policy=deepcopy(POLICY))
    conf['scheduling']['escalation_calls_per_slate'] = REVIEW_CALLS_PER_GROUP
    conf['_experiment_profile']['review_budget_policy'] = 'shared-conditional-packages-v1'
    conf['_experiment_profile']['slate_barriers'] = OPERATIONS['slate_barriers']
    conf['_config_sha256'] = digest({'base':conf['_config_sha256'], 'revision':REVISION, 'executable':executable(), 'timeout_seconds':TIMEOUT_SECONDS,
                                  'routing':POLICY, 'roles':conf['role_profile'],
                                  'review_budget':REVIEW_CALLS_PER_GROUP})
    conf['_run_identity']['config_sha256'] = conf['_config_sha256']
    return conf


def require_launch(root, budget=None, window=None, *, reference_snapshot_hash=None):
    if budget is None:
        runtime=root; root,budget,window=runtime.budget.root,runtime.budget,runtime.window
    value = read(Path(root)/'launch_manifest.private.json')
    if (value['revision'] != REVISION or value['accounts'] != sorted(CALL_HOMES)
            or value['code_hash'] != code_identity() or value['budget_hash'] != digest(budget.limits)
            or value['runtime_policy'] != POLICY or value['launch_authorized'] is not True
            or window is not None and window not in value['window_ids']
            or reference_snapshot_hash is not None and reference_snapshot_hash != value['reference_snapshot_hash']
            or value.get('config_science_hash') not in (None, science_hash(CONFIG))
            or value.get('executable') not in (None, executable())):
        raise IntegrityError('frozen code/account/budget/context/science config or Codex executable changed')
    return value


def assert_launch(root):
    from .budget import Budget
    return require_launch(root, Budget(root))


def verify_sources(root):
    """The bank registries recorded at preparation are unchanged (bank_inputs.py): a run keeps its bank snapshot."""
    count = 0
    for r in read(Path(root) / 'inputs/protected.private.json'):
        if not Path(r['path']).is_file() or sha(Path(r['path'])) != r['sha256']:
            raise IntegrityError('the bank changed since this run was prepared: ' + r['path'])
        count += 1
    return count


def account_pool(path):
    """One durable worker pool per account, shaped by the batch config."""
    from arcgen.lib.e18_pre_e20_pool import DedicatedCallPool
    shape = OPERATIONS['pool']
    total = shape['downstream_workers'] + shape['upstream_workers']
    # Worker counts may change between sessions by recorded amendment (entry.py amend / bin/arcgen workers).
    ramp = OPERATIONS.get('ramp_up') or {}  # optional start-up pacing (config.py)
    return DedicatedCallPool(path, downstream_workers=shape['downstream_workers'],
                             upstream_workers=shape['upstream_workers'], global_ceiling=total, codex_cap=total,
                             allow_resize=True, ramp_gap_seconds=ramp.get('gap_seconds', 0),
                             ramp_window_seconds=ramp.get('window_seconds', 0))
