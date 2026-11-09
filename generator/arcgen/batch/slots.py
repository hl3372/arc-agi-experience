"""Property slots that carry code (2026-10-23): ONE, a single demonstration pair.

Held-out ARC sets have single-training-pair tasks; recent batches produce almost none (bank 1.4%), because the teaching
line, the weakness and structure lines and the base planner's "one pair is allowed" all lean on cross-pair evidence. ONE
is an ordinary property slot (idea text prompts/v1/ideas/modifiers/ONE.md, binding planner line prompts/v1/slots/
ONE.planner.md), plus the little code a slot has needed for the first time:
- exclusions enforced in config (config._validate_one: the profile's modifier_exclusions must keep ONE off every arm,
  family and slot value below, and off every weakness line it has) and at admission (no weakness line, structure line,
  teaching draw, departure draw, weakness overlay or MANY overlay; fixed colours and no pose), so that no text drawn with
  it asks for two or more pairs, variation between pairs, or a test case the pairs do not show;
- a plan check (the planned training pair count) sharing the one re-plan of the rung and line checks (runtime.py), with a
  receipt at run_dir/slot_check/<task>.json and record['slot_check'];
- a review policy for the adjudicator (prompts/v1/slots/ONE.adjudicator.md) when the episode has one training pair, and a
  repair note, attached the way the lines' are (gate.py, runtime.call_review);
- the final pair count in blind_performance.one_pair, for ONE ideas and their matched control ({"slot_control": "ONE"} on
  a spec row: the same arm and size without the slot, with the same exclusions and presentation).
This module imports no batch module at load time (config.py imports it); what needs the run's config imports it lazily.
"""
from pathlib import Path
import re

ONE = 'ONE'
SLOT_DIR = Path(__file__).resolve().parent / 'prompts/v1/slots'
CHECKED = frozenset({ONE})  # slots with a plan check, receipt and recorded pair count
# The levels a ONE idea (or its control) may take when its call fixes a size in a weakness run (admission._structure_size;
# a 'large' call takes the largest). R4 is a weakness-line rung only.
ONE_RUNGS = frozenset({'R1', 'R2', 'R3'})

# Names whose text needs two or more training pairs, variation between pairs, or a test case the pairs do not show (the
# audit of prompts/v1/bridges, families, slots and ideas/cards, 2026-10-23). The values are the reasons, for the build log.
ONE_EXCLUDED_ARMS = {
    'SPREAD': 'six to nine training pairs, each showing only part of the rule',
    'STAGES': 'a chain of states: each training output is the next training input',
    'ACROSS': 'at least three training pairs showing different kinds of picture',
    'EXEMPLAR': 'what the worked instance shows must differ between pairs',
    'CONTRAST': 'a controlled comparison across pairs (its one-grid form duplicates the repeated instances ONE asks for)',
    'J': 'joint inference: any one pair must be insufficient',
    'D1': 'already the single-pair arm, with its own escape from the pair count; ONE puts the property on other arms',
    'T': 'the test derives a consequence no training pair instantiates',
    'DISC': 'the obvious reading fails on comparing the examples, and the test needs it in a situation they do not show',
    'INTERACT': 'the test needs an insight about the meeting that the examples did not literally show',
    'AIM': 'the pairs teach the condition and what may change between them',
}
ONE_EXCLUDED_FAMILIES = {
    'LEGEND': 'the key changes between pairs so that no two pairs show the same key',
    'SYMSEQ': 'the structure\'s position or form differs in each pair',
}
ONE_EXCLUDED_MODIFIERS = {
    'KEY': 'key content varies across the training pairs',
    'CHAINED': 'each training output is the next training input',
    'MINIMAL': 'a controlled comparison, typically across pairs',
    'DISTRIBUTED': 'six to nine pairs, each showing only part of the rule',
    'INSTRUCT': 'the instruction varies across the pairs and each kind of entry acts in two pairs',
    'EXTENT': 'output sizes differ across the training pairs',
    'SCENE': 'the output\'s size differs across the pairs',
    'PAYLOAD': 'colour content varies across the training pairs',
    'IGNORE': 'irrelevant content varies independently of what decides the answer',
    'DERIVED': 'the test meets an outcome no training output shows',
    'DIGEST': 'one small summary output per pair: a single pair shows its form only once',
    'JOINED': 'two or three joined rules, where ONE allows one or two laws',
    # skill lines (skill runs only)
    'QUANTITY': 'the pairs take different values of the quantity',
    'SYMMETRY': 'the axis or centre is placed differently across the pairs',
    'MATCH': 'matches in several orientations or sizes across the pairs',
    'CUED': 'every cue value acts in at least two pairs',
    'EMIT': 'every kind of meeting acts in at least two pairs',
    'GROWTH': 'the start or the change varies across pairs',
    'NOISY': 'what counts as an imperfection is shown across the pairs',
    # composite batch c1 (2026-10-25): of the four new skill lines, two need evidence across pairs. RANK and GRAPH do not:
    # their cases (each ranked thing, each node or part of a network) repeat inside one scene, and their texts ask for
    # variety across pairs only as breadth, which ONE's binding line holds over for a single-pair idea.
    'RECURSE': 'a different starting shape in each pair separates the self-similar rule from one picture',
    'AUTOMATON': 'every case of the local rule acts in two pairs started from different arrangements, which also settle the '
                 'number of steps or the stopping point',
}
# Never drawn with ONE, enforced at admission rather than by exclusion lists (weakness lines, by name, also join the
# profile's exclusions: config._validate_one): the weakness and structure lines (cross-pair evidence by design), the
# teaching line (each law in two pairs), the departure package (an unshown test situation), the weakness overlays (EDGE
# asks for cut things in two or more pairs; ROLES varies colour roles between pairs) and the MANY overlay (five to eight
# pairs). ONE ideas keep one colouring and no pose instruction: with one pair, a role colour or pose first seen at test is
# a kind of case the pair does not show.


def one_idea(study):
    """Whether an idea carries the ONE slot."""
    return ONE in ((study or {}).get('idea_modifiers') or ())


def bound(study):
    """A ONE idea or its matched control: both take ONE's exclusions and presentation."""
    return one_idea(study) or (study or {}).get('slot_control') == ONE


def checked(study):
    """The checked slots an idea carries (a plan check, a receipt, the recorded pair count)."""
    return tuple(sorted(CHECKED & set((study or {}).get('idea_modifiers') or ())))


def required_exclusions(modifiers, lines=()):
    """What a profile's modifier_exclusions must keep ONE apart from, given its idea_modifiers: every excluded arm and family,
    the excluded slot values it has, and the weakness lines it has (exclusions may name only known names, so values the
    profile lacks are left out)."""
    mods = set(modifiers or ())
    return sorted(set(ONE_EXCLUDED_ARMS) | set(ONE_EXCLUDED_FAMILIES) | (set(ONE_EXCLUDED_MODIFIERS) & mods)
                  | (set(lines) & mods))


def exclusion_problems(sci, lines=()):
    """The names a profile with ONE among its idea_modifiers fails to keep apart from it (empty: fine). A modifier is apart
    when it shares ONE's slot or either side's exclusion list names the other."""
    mods = sci.get('idea_modifiers') or {}
    if ONE not in mods:
        return []
    excl = sci.get('modifier_exclusions') or {}
    slot_of = {m: k for k, values in (sci.get('modifier_slots') or {}).items() for m in values}
    mine = set(excl.get(ONE) or ())
    return [n for n in required_exclusions(mods, lines)
            if n not in mine and ONE not in (excl.get(n) or ())
            and not (n in mods and slot_of.get(n) is not None and slot_of.get(n) == slot_of.get(ONE))]


# ------------------------------------------------------------------ plan check and the shared re-plan


def _plan(value):
    if not isinstance(value, dict):
        return None
    plan = value if 'public_contract' in value else value.get('concrete_plan')
    return plan if isinstance(plan, dict) and plan.get('status', 'ready') == 'ready' and plan.get('public_contract') else None


def planned_pairs(plan):
    """The training pair count a ready plan commits to (its storyboard), or None."""
    plan = _plan(plan)
    if not plan:
        return None
    pairs = plan['public_contract'].get('training_pairs')
    return len(pairs) if isinstance(pairs, list) else None


def plan_issues(study, plan):
    """What a ready plan of a checked slot lacks, as short notes for the one re-plan."""
    out = []
    if one_idea(study):
        n = planned_pairs(plan)
        if n is not None and n != 1:
            out.append(f'it planned {n} training pairs, where this idea is taught by exactly one')
    return out


def replan_note(issues):
    return ('Pair check on an earlier plan for this idea: ' + '; '.join(issues) + '. Plan the idea again with exactly one '
            'training pair, as the one-demonstration line above says: every law, convention and kind of case the test meets '
            'acts at least twice inside that pair. Keep what makes the idea work; the earlier plan is used if the new one aborts.')


def plan_check(study, first, second=None, replanned=False, used='first'):
    """The plan-stage receipt of a checked slot (run_dir/slot_check/<task>.json and record['slot_check'])."""
    chosen = second if used == 'second' else first
    return {'slots': list(checked(study)), 'first_pairs': planned_pairs(first), 'first_issues': plan_issues(study, first),
            'replanned': replanned, 'used': used, 'second_pairs': planned_pairs(second) if second is not None else None,
            'final_pairs': planned_pairs(chosen), 'final_issues': plan_issues(study, chosen)}


# ------------------------------------------------------------------ review policy and repair note


def _version(name):
    head = (SLOT_DIR / name).read_text().split('-->', 1)[0]
    found = re.search(r'prompt-version:\s*(\S+)', head)
    return found.group(1) if found else None


def body(name):
    return (SLOT_DIR / name).read_text().split('-->', 1)[1].strip()


POLICIES = {ONE: 'ONE.adjudicator.md'}
NOTES = {ONE: 'ONE.repair_note.md'}
POLICY_VERSION = _version(POLICIES[ONE])


def review(record):
    """The slot review policies that apply to this episode version: (names, None), ([], reason) or (None, None) for an idea
    without one. ONE's applies when the current episode has exactly one training pair (otherwise the plan or a repair
    added pairs, and the ordinary review stands)."""
    study = (record or {}).get('study') or {}
    names = [s for s in checked(study) if s in POLICIES]
    if not names:
        return None, None
    episode = record.get('complete_task')
    if not episode:
        return [], 'no_episode'
    if ONE in names and len(episode.get('train') or []) != 1:
        names = [n for n in names if n != ONE]
        if not names:
            return [], f"pairs:{len(episode.get('train') or [])}"
    return names, None


def policy_text(names):
    return '\n\n'.join(body(POLICIES[n]) for n in names or ())


def repair_notes(record):
    """Notes appended to every repair instruction of an idea carrying a slot with one (ONE: keep the one pair)."""
    study = (record or {}).get('study') or {}
    return [body(NOTES[s]) for s in checked(study) if s in NOTES]


# ------------------------------------------------------------------ the recorded pair count


def fields(record):
    """blind_performance['one_pair'] for a ONE idea or its matched control: the final episode's training pair count,
    whether it is exactly one, and the plan check's counts (absent elsewhere)."""
    study = (record or {}).get('study') or {}
    if not bound(study):
        return {}
    episode = record.get('complete_task') or {}
    n = len(episode['train']) if episode.get('train') is not None else None
    check = record.get('slot_check') or {}
    return {'one_pair': {'slot': ONE if one_idea(study) else None, 'control': study.get('slot_control') == ONE,
                         'train_pairs': n, 'exactly_one': n == 1 if n is not None else None,
                         'planned_pairs': check.get('final_pairs'), 'first_planned_pairs': check.get('first_pairs'),
                         'replanned': bool(check.get('replanned')), 'repairs': int(record.get('repair_count') or 0)}}
