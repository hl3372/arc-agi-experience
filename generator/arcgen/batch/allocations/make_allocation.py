"""Build an allocation table (author calls with stable IDs) from a compact batch spec.

Spec (JSON):
  {"revision": "...", "seed": 123, "windows": {"<id>": {"account": "main", "wave": "A", "context": "c01"}, ...},
   "rows": [["<window>", "<ARM>", "<INVITATION or null>", "U|BA|BI", "astra|sol", "<effort>"], ...],
   "planner_comparison": {...optional...}}
A row may end with a seventh element, {"family": "<FAMILY>", "rung": "<RUNG>"} (targeted runs, 2026-10-12): both join the
call's identity, so rows without them keep the IDs they always had.
The seventh element may also hold {"skill": "<MODIFIER>"} (skill runs, 2026-10-18): the call carries that idea modifier
for certain (allocation.assign_modifiers places it before the drawn shares); it joins the identity too. In a weakness run
(2026-10-20) "skill" may list several modifiers (a line and a second property, ["MARKS", "IGNORE"]), {"control": true}
marks a matched-control call (no line), {"size": "small" | "large" | "R2" ...} fixes the call's size arm or rung, and
{"fence": "<relaxation>" | "strict" | "relaxed"} its relaxed fence ("relaxed": each idea draws one of the line's relaxed forms);
{"decoy": true} with a JUDGE pin makes a planted negative for the consensus procedure (study.judge_decoy, DESIGN §6.9).
Structure lines (2026-10-22, STRUCTURE_SPEC.md): {"structure": "<STRUCTURE>"} draws that structure line on the call, on top
of its content line or arm (study.structure_call; the host matrix applies at admission); never with {"control": true}.
{"structure_control": true} marks the structure screen's matched control: the same host (line or arm) and size, no
structure line (study.structure_control).
ONE (2026-10-23, batch/slots.py): {"skill": "ONE"} pins the single-demonstration slot on an arm call (never with a line,
structure, control, fence or decoy, and at most R3); {"slot_control": "ONE"} marks its matched control: the same arm and
size, no slot (study.slot_control), with the slot's exclusions and presentation.
Composite batch c1 (2026-10-25): {"slot_control": "<SKILL>"} marks a skill line's matched control the same way (the same
arm and size, no skill; assign_modifiers never draws that skill onto it), and in a weakness run any arm row's
{"size": ...} is taken at admission (R1 to R3; R4 only with a weakness line pin or {"control": true}).
Usage: python3 -m arcgen.batch.allocations.make_allocation SPEC.json OUT.json
IDs are digests of each row with the revision, so a new revision always gets new IDs.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import sys

BANKS = {'U': 'unconditioned', 'BA': 'bank_aware', 'BI': 'bank_inspired', 'L': 'loop'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def _line_pin(skill):
    from arcgen.batch.config import LINE_MODIFIERS
    return bool(set([skill] if isinstance(skill, str) else skill or ()) & set(LINE_MODIFIERS))


def _weakness_row_problem(target):
    """A weakness spec row's fence, decoy and control flags against the line registry and the run's profile, so that a
    wrong row fails here rather than at admission, after its idea call is paid (review N7)."""
    from arcgen.batch.config import LINE_MODIFIERS, LINE_REGISTRY, LINE_CONTROL, SCIENCE
    skill = target.get('skill')
    line = next((m for m in ([skill] if isinstance(skill, str) else skill or ()) if m in LINE_MODIFIERS), None)
    relax = (LINE_REGISTRY['lines'].get(line) or {}).get('relax') or {}
    fence = target.get('fence')
    if fence is not None and fence != 'strict' and (not line or fence != 'relaxed' and (fence not in relax or relax[fence].get('decoy'))):
        return f'fence {fence} is not a drawable relaxation of {line or "an unpinned row"}'
    if fence == 'relaxed' and not [k for k, r in relax.items() if not r.get('decoy')]:
        return f'{line} has no relaxed forms'
    if target.get('decoy') and not any(r.get('decoy') for r in relax.values()):
        return f'{line} has no decoy form'
    weakness = SCIENCE.get('weakness')
    if target.get('control') and weakness is not None and LINE_CONTROL not in weakness.get('lines', {}):
        return 'a control row needs the control line in science.weakness.lines'
    if line and weakness is not None and line not in weakness.get('lines', {}):
        return f'{line} is not among science.weakness.lines'
    if 'structure_control' in target and (target['structure_control'] is not True or 'structure' in target or target.get('control')):
        return 'structure_control is true, on a row without a structure line and not the matched control line'
    structure = target.get('structure')
    if structure is not None:
        entry = (LINE_REGISTRY.get('structures') or {}).get(structure) if isinstance(structure, str) else None
        if entry is None:
            return f'{structure} is not a structure line'
        if target.get('control'):
            return 'a matched-control row draws no structure line'
        if weakness is not None and structure not in (weakness.get('structures') or {}):
            return f'{structure} is not among science.weakness.structures'
        if line in (entry.get('not_on') or ()):
            return f'{structure} is never drawn on {line} (the host matrix)'
        size = target.get('size')
        if size in ('R1', 'R2', 'R3', 'R4') and size not in entry['rungs']:
            return f'{structure} allows only {", ".join(entry["rungs"])}'
    return None


def _slot_row_problem(target):
    """A ONE row or its matched control (2026-10-23): checked here, before its idea call is paid, against the exclusions
    admission enforces (slots.py). A skill line's matched control (2026-10-25, composite batch c1: {"slot_control": SKILL},
    the same arm and size without the skill) pins no skill or weakness line and takes no structure, control, fence or
    decoy."""
    from arcgen.batch.config import SKILL_MODIFIERS
    from arcgen.batch.slots import ONE, ONE_RUNGS
    skill = target.get('skill')
    pins = [skill] if isinstance(skill, str) else list(skill or ())
    control = target.get('slot_control')
    if control in SKILL_MODIFIERS:
        clash = sorted({'structure', 'structure_control', 'control', 'fence', 'decoy'} & set(target))
        if set(pins) & (set(SKILL_MODIFIERS) | {ONE}) or _line_pin(skill):
            return f'a {control} control row pins no skill line, weakness line or ONE'
        if clash:
            return f'a {control} control row takes no {", ".join(clash)}'
        if target.get('size') == 'R4':
            return f'a {control} control row takes R1 to R3 only, not R4'
        return None
    if control is not None and control != ONE:
        return f'slot_control names a slot with a matched control (ONE or a skill line), not {control}'
    if ONE not in pins and control is None:
        return None
    who = 'a ONE row' if ONE in pins else 'a ONE control row'
    if ONE in pins and control is not None:
        return 'a ONE row is not its own matched control'
    if _line_pin(skill):
        return f'{who} carries no weakness line'
    clash = sorted({'structure', 'structure_control', 'control', 'fence', 'decoy'} & set(target))
    if clash:
        return f'{who} takes no {", ".join(clash)}'
    size = target.get('size')
    if size in ('R1', 'R2', 'R3', 'R4') and size not in ONE_RUNGS:
        return f'{who} takes R1 to R3 only, not {size}'
    return None


def build(spec, models, arm_labels, valid_invitation):
    windows = spec['windows']; order = list(windows)
    calls = []
    for ordinal, (window, arm, invitation, bank, model, effort, *rest) in enumerate(spec['rows'], 1):
        w = windows[window]
        target = rest[0] if rest else {}
        skill = target.get('skill')
        if (len(rest) > 1 or set(target) - {'family', 'rung', 'skill', 'control', 'size', 'fence', 'decoy', 'structure', 'structure_control',
                                            'slot_control'}
                or len({'family', 'rung'} & set(target)) == 1
                or ('skill' in target and not (isinstance(skill, str) or (isinstance(skill, list) and skill
                                                                           and all(isinstance(s, str) for s in skill)
                                                                           and len(set(skill)) == len(skill))))
                or ('control' in target and (target['control'] is not True or 'skill' in target and _line_pin(skill)))
                or ('size' in target and target['size'] not in ('small', 'large', 'R1', 'R2', 'R3', 'R4'))
                or ('fence' in target and not isinstance(target['fence'], str))
                or ('decoy' in target and (target['decoy'] is not True or not _line_pin(skill) or 'fence' in target))):
            raise ValueError(f'row {ordinal}: a seventh element must be {{"family": ..., "rung": ...}} and/or {{"skill": ...}}, '
                             '{"control": true}, {"size": "small" | "large" | a rung}, {"fence": relaxation or "strict"} and, with a line, '
                             '{"decoy": true}')
        problem = (_weakness_row_problem(target) if {'fence', 'decoy', 'control', 'structure', 'structure_control'} & set(target)
                   or _line_pin(skill) else None) or _slot_row_problem(target)
        if not problem and target.get('size') == 'R4' and not _line_pin(skill) and not target.get('control'):
            # the full-scale rung is a weakness line's (or the line control's) only; an arm idea takes R1 to R3 (admission)
            problem = 'R4 needs a weakness line pin or the matched control line'
        if problem:
            raise ValueError(f'row {ordinal}: {problem}')
        if invitation is not None and not valid_invitation(arm, invitation): raise ValueError(f'invitation {invitation} is not hosted by {arm}')
        row = dict(revision=spec['revision'], ordinal=ordinal, window=window, context=w['context'], wave=w['wave'],
                   account=w['account'], arm=arm, arm_label=arm_labels.get(arm, arm), invitation=invitation,
                   bank_setting=BANKS[bank], model=models[model], effort=effort,
                   seed=spec['seed'] + order.index(window) * 101, ideas=6 if w.get('commit') in ('kept', 'explore') else 4,
                   **target)
        if w.get('commit') in ('kept', 'explore'):  # up to six bank-checked drafts per call (explore: after a second round)
            row['commit'] = w['commit']
        row['private_id'] = 'call-' + digest(row)[:20]
        row['candidate_ids'] = ['k' + digest({'call': row['private_id'], 'i': i})[:18] for i in range(row['ideas'])]
        calls.append(row)
    ids = [k for c in calls for k in c['candidate_ids']]
    if len(ids) != len(set(ids)): raise ValueError('repeated candidate IDs')
    return dict(revision=spec['revision'], status='draft_not_launched', launch_authorized=False, windows=windows, calls=calls,
                **({'saved_windows': spec['saved_windows']} if spec.get('saved_windows') else {}),
                **({'saved_idea_windows': spec['saved_idea_windows']} if spec.get('saved_idea_windows') else {}),
                planner_comparison=spec.get('planner_comparison'),
                coverage=dict(idea_calls=len(calls), candidate_ideas=len(ids), arms=len({c['arm'] for c in calls}),
                              invited_calls=sum(c['invitation'] is not None for c in calls),
                              invitations_this_batch=sorted({c['invitation'] for c in calls if c['invitation']}),
                              arm_calls=dict(Counter(c['arm'] for c in calls)),
                              **({'family_rung_calls': dict(Counter(c['family'] + '/' + c['rung'] for c in calls if 'family' in c))}
                                 if any('family' in c for c in calls) else {}),
                              bank_mode_calls=dict(Counter(c['bank_setting'] for c in calls)),
                              model_effort_calls=dict(Counter(c['model'] + '/' + c['effort'] for c in calls)),
                              account_calls=dict(Counter(c['account'] for c in calls))))


if __name__ == '__main__':
    from arcgen.batch.supply import ARMS, valid_invitation
    from arcgen.batch.config import SCIENCE
    spec = json.loads(Path(sys.argv[1]).read_text())
    value = build(spec, SCIENCE['models'], {k: v['label'] for k, v in ARMS.items()}, valid_invitation)
    Path(sys.argv[2]).write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps(value['coverage'], indent=1))
