"""Scale rungs (2026-10-12).

A rung is a binding scale line for the planner (prompts/v1/rungs/) with floors from science.rungs:
- output_cells: the largest test output's area;
- escalation: the test input over the mean training input (or the test output over the mean training output, whichever
  is larger), when above zero;
- computed_cells: cells the solver computes, measured on the finished grids only (properties.py), since a plan's prose
  does not state it.
The plan-stage check reads the first two from the plan's structured dimensions. A plan below them gets one re-plan with
a one-line note (runtime.py); a second miss, or a re-plan that aborts, keeps the plan that was made and records the
rung it reached.
"""
from .config import SCIENCE, RUNGS, ALL_RUNGS

REPLAN_SUFFIX = '-replan'  # re-plan call IDs: '<task>-replan-concrete-plan' (routing and accounting need the suffix)


def task_of_plan_call(call_id):
    """The task ID of a planning call, first plan or re-plan."""
    return call_id[:-len('-concrete-plan')].removesuffix(REPLAN_SUFFIX)


def floors(rung):
    return SCIENCE['rungs'][rung]


def plan_of(value):
    """The concrete plan inside a planner response (ordinary plans are returned bare, routed ones wrapped), or None."""
    if not isinstance(value, dict):
        return None
    plan = value if 'public_contract' in value else value.get('concrete_plan')
    if not plan or plan.get('status') != 'ready' or not plan.get('public_contract') or not plan.get('private_execution'):
        return None
    return plan


def _area(d):
    return d['height'] * d['width']


def measures(plan):
    """Planned sizes from the plan's structured dimensions."""
    public, private = plan['public_contract'], plan['private_execution']
    train_in = [_area(p['input_dimensions']) for p in public['training_pairs']]
    train_out = [_area(p['output_dimensions']) for p in public['training_pairs']]
    test_in = [_area(p['input_dimensions']) for p in public['test_inputs']]
    test_out = [_area(p['output_dimensions']) for p in private['test_outputs']]
    largest = max(private['test_outputs'], key=lambda p: _area(p['output_dimensions']))['output_dimensions']
    escalation = max(max(test_in) / (sum(train_in) / len(train_in)), max(test_out) / (sum(train_out) / len(train_out)))
    return {'test_output_cells': max(test_out), 'test_output_shape': [largest['height'], largest['width']],
            'escalation': round(escalation, 3), 'training_pairs': len(train_in)}


def reached(m):
    """The highest rung whose plan-stage floors these measures meet, or None."""
    best = None
    for rung in ALL_RUNGS:  # R4 has floors only in a weakness run
        f = SCIENCE.get('rungs', {}).get(rung)
        if f and m['test_output_cells'] >= f['output_cells'] and (not f['escalation'] or m['escalation'] >= f['escalation']):
            best = rung
    return best


def shortfalls(rung, m):
    f = floors(rung); out = []
    if m['test_output_cells'] < f['output_cells']:
        h, w = m['test_output_shape']
        out.append(f"its largest test output was {h}×{w} = {m['test_output_cells']} cells, under this idea's floor of "
                   f"{f['output_cells']} cells")
    if f['escalation'] and m['escalation'] < f['escalation']:
        out.append(f"its test was {m['escalation']:.2f} times the size of the average training pair, under the required "
                   f"{f['escalation']:g}")
    return out


def feedback(rung, m):
    """The one note a re-plan receives, appended to the unchanged planner prompt."""
    return ('Scale check on an earlier plan for this idea: ' + '; '.join(shortfalls(rung, m)) + '. Plan the idea again so '
            'that its test meets the scale line above, through more consequences of the same laws, each visible where it '
            'happens, never through padding, repeats or new conventions. Keep what makes the idea work. If the idea cannot '
            'reach that scale this way, abort; the earlier plan will then be used.')
