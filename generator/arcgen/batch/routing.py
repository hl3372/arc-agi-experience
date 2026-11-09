"""Explicit call binding from the batch config; no shared profile mutation or model classifier."""
from .config import CONFIG, SCIENCE, OPERATIONS, binding as _configured

ASTRA = SCIENCE['models']['astra']
SOL = SCIENCE['models']['sol']
IDEA_EFFORTS = {SCIENCE['models'][k]: tuple(v) for k, v in SCIENCE['idea_efforts'].items()}


def binding(model, effort): return {'model': model, 'reasoning_effort': effort}


DEFAULT_ROLES = {purpose: _configured(pair) for purpose, pair in SCIENCE['roles'].items()}
PLANNERS = {name: _configured(pair) for name, pair in SCIENCE['planners'].items()}
CALLS = {name: _configured(pair) for name, pair in SCIENCE['call_overrides'].items()}
POLICY = {'id': CONFIG['policy_id'], 'accounts': sorted(OPERATIONS['accounts']), 'planning_account': 'owner',
          'planning_service_tier': None, 'planners': PLANNERS, 'research_enabled': False, 'default_roles': DEFAULT_ROLES,
          'idea_efforts': {model: list(efforts) for model, efforts in IDEA_EFFORTS.items()},
          'technical_conditional': CALLS['technical_conditional'], 'mechanical_repair': CALLS['mechanical_repair'],
          'repaired_primary_blind': CALLS['repaired_primary_blind'], 'maximum_attempts': 2,
          'editorial_mode': 'bypass', 'unsolved_retention': True, 'automatic_commit': False}


def resolve(purpose, call_id, *, repair_type=None, default=None, planner=None):
    if purpose == 'editorial_board': raise ValueError('editorial boards remain bypassed')
    if purpose == 'planning':
        if not call_id.endswith('-concrete-plan') or planner not in PLANNERS:
            raise ValueError('planning requires its frozen per-task planner')
        return dict(PLANNERS[planner])
    if purpose == 'technical_verification':
        if call_id.endswith('-technical-primary'): return dict(DEFAULT_ROLES['technical_verification'])
        if call_id.endswith(('-technical-second', '-technical-after-repair', '-technical-after-ambiguity-repair')):
            return dict(CALLS['technical_conditional'])
        raise ValueError('unknown technical route')
    if purpose == 'primary_blind':
        if not call_id.endswith('-blind-primary'): raise ValueError('unknown primary blind route')
        return dict(DEFAULT_ROLES['primary_blind'])
    if purpose == 'conditional_blind':
        if call_id.endswith('-blind-second'): return dict(CALLS['second_blind'])
        if call_id.endswith('-blind-after-ambiguity-repair'): return dict(CALLS['repaired_primary_blind'])
        # structure lines (2026-10-22, STRUCTURE_SPEC D): the recorded default-reading call, low-effort Sol
        if call_id.endswith('-default-reading') and 'default_reading' in CALLS: return dict(CALLS['default_reading'])
        raise ValueError('unknown conditional blind route')
    if purpose == 'targeted_repair':
        if repair_type not in ('mechanical', 'ambiguity'): raise ValueError('repair requires an explicit type')
        if not call_id.endswith('-' + repair_type + '-repair'): raise ValueError('repair type and call ID disagree')
        return dict(CALLS[repair_type + '_repair'])
    if purpose.startswith('idea_'):
        if not default or default.get('reasoning_effort') not in IDEA_EFFORTS.get(default.get('model'), ()):
            raise ValueError('unfrozen author model/effort')
        return binding(default['model'], default['reasoning_effort'])
    if purpose == 'transport_probe':
        if not default or (default['model'], default['reasoning_effort']) not in {(m, e) for m in {ASTRA, SOL, *IDEA_EFFORTS} for e in ('high', 'xhigh', 'max')}:
            raise ValueError('unfrozen probe binding')
        return binding(default['model'], default['reasoning_effort'])
    if purpose not in DEFAULT_ROLES: raise ValueError('unbound routine/escalation purpose: ' + purpose)
    return dict(DEFAULT_ROLES[purpose])
