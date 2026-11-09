"""Reuse r4 authoring; add only informed adjudication and a public repair challenge."""
from copy import deepcopy
from arcgen.lib.factory_runtime import StageBinding
from arcgen.lib.schemas import SchemaError
from arcgen.lib.astra_authoring_contract_v3 import public_diagnostics, validate_public_realization
from arcgen.lib.util import content_hash, load_prompt, render_prompt
from arcgen.astra_evidence_runtime_revision_20260911.contracts import Bindings as R4Bindings
from arcgen.astra_evidence_runtime_revision_20260911 import evidence
from .common import CARDS, component
from .supply import ARMS
from arcgen.lib import astra_authoring_schemas_v3 as schemas
from arcgen.astra_usage_diversity_20260913 import adjudication, recheck
from . import repair, repair_v1, lines
from . import slots as _slots  # ONE (2026-10-23); planning_templates has a 'slots' parameter
from .common import HERE
from .config import SCIENCE, FAMILIES, RUNGS, ALL_RUNGS, LINE_NAMES, WEAKNESS_OVERLAYS, STRUCTURES

PROMPT_SET = SCIENCE.get('prompt_set', 'r23')
# the teaching line's pair count: round 2's 'fewest pairs' (teach.plain) or round 1's 'three pairs' (teach.three)
TEACH_LINE = 'teach.three.planner.md' if (SCIENCE.get('teach_draw') or {}).get('pairs') == 'three' else 'teach.plain.planner.md'
COLOUR_ROLES = SCIENCE.get('colour_roles', 'v1')  # 'choice': neutral per-role decision; 'vary': vary where the rule allows;
# 'mix': each idea is drawn 'vary' (at colour_vary_rate) or 'v1', recorded as study.colour_roles
VARY_PLANNER = 'v1/planner_colour_vary.md'
# diversity-v1 templates (prompts/v1/); the r23 templates remain only where v1 composes with them.
V1_OVERRIDES = {
    'concrete_planner': ('v1/planner.md', schemas.PLAN_STAGE),
    'v44_quality_plan': ('v1/quality.md', schemas.PLAN_STAGE),
    'task_realizer': ('v1/realizer.md', schemas.REALIZER_STAGE),
    'technical_verifier': ('v1/technical.md', evidence.STAGE),
    'targeted_repair': ('v1/bounded_repair.md', repair_v1.STAGE),
    'ambiguity_adjudicator': ('v1/difficulty_adjudicator.md', adjudication.STAGE),
    'difficulty_technical_recheck': ('v1/difficulty_technical_recheck.md', recheck.STAGE),
    'logical_leap_planner_addendum': ('v1/logical_leap_addendum.md', schemas.LOGICAL_STAGE),
    'evidence_scale_addendum': ('v1/overlay_addendum.md', schemas.SCALE_STAGE),
}
# v1 presentation overlays reuse the scale route (decision field expanded = used, kept_compact = kept ordinary).
OVERLAY_INVITATIONS = {
    'scale': 'Invitation: a substantially larger canvas. Build a version with a much larger canvas (for instance more than about 200 cells in at least one input) because the space carries something: distance, distribution, more instances in different local circumstances, global structure, or incidental material the solver must set aside. Padding and near-identical repeats do not count.',
    'COMPACT': 'Invitation: compact grids. Build a version whose grids are each at most about 100 cells, with the concept intact and every cue legible. Keep the ordinary plan if compression would need a decoder or lose a training purpose.',
    'MANY': 'Invitation: many examples. Build a version with five to eight training pairs that vary circumstances, show contrasts or confirm the rule in different situations. Pairs need not each be indispensable; rerolls do not count.',
    'XFER': 'Invitation: varied tests. Build a version with two or three test inputs that apply the same rule in meaningfully different ways, each supported by training. A recolour, enlargement or repeat of the same work does not count.',
    # A plain second test (eval-style study F-03, 2026-10-14): another instance of the same rule, never a case the pairs do
    # not show; whether a test may go beyond the pairs is the departure package's question, not this overlay's.
    'SECOND': 'Invitation: a second test. Build a version with two test inputs. The second applies the same rule as the first to another instance, a different arrangement, size or set of objects that the training pairs support. It is not a recolour, a copy or a larger repeat of the first test, and it brings in no case the training pairs do not show.',
    'RICH': 'Invitation: a richer scene. Build a version whose grids carry more organized visual content for the solver to read, relate or build: varied instances, context that changes how the rule applies, or incidental material the solver must set aside. Keep one rule; do not add a second rule or unrelated filler.',
}
# v1 moves presentation archetypes to planner overlays: ideas from these arms get no arm bridge.
V1_NO_BRIDGE = frozenset({'RICH', 'MANY', 'XFER', 'EXP', 'COMPACT', 'REF_STYLE', 'C', 'I1'})
# v1 bridges for new or reworked concept arms (files in prompts/v1/bridges/).
V1_BRIDGES = {'OPEN': 'OPEN.planner.md', 'DIRECT': 'DIRECT.planner.md', 'BOARD': 'BOARD.planner.md', 'RHYTHM': 'RHYTHM.planner.md', 'SIGNAL': 'SIGNAL.planner.md', 'FORM': 'FORM.planner.md', 'SEM': 'SEM.planner.md', 'PER': 'PER.planner.md', 'GEO': 'GEO.planner.md',
              'WORLD': 'WORLD.planner.md',
              'RESTORE': 'RESTORE.planner.md',
              'GEN': 'GEN.planner.md',
              'EXEMPLAR': 'EXEMPLAR.planner.md',
              'INTERACT': 'INTERACT.planner.md',
              'SEE': 'SEE.planner.md',
              'TEMPLATE': 'TEMPLATE.planner.md',
              **{a: a + '.planner.md' for a in ('LEVELS', 'INVERSE', 'AIM', 'ACROSS', 'STAGES', 'CONTRAST', 'VIEWS', 'SPREAD')}}
# Property slots with a planner line (prompts/v1/slots/<VALUE>.planner.md): an idea from a call that drew the value gets
# the line after its arm's bridge, so the planner keeps the property the idea was invented under.
SLOT_DIR = HERE / 'prompts/v1/slots'
SLOT_BRIDGES = frozenset(p.name.split('.')[0] for p in SLOT_DIR.glob('*.planner.md'))


# Targeted families and scale rungs (2026-10-12): a family's planner line (prompts/v1/families/<FAMILY>.planner.md) and the
# binding rung line for the family, or ANY for a general run's drawn rung (prompts/v1/rungs/<FAMILY|ANY>.<RUNG>.planner.md).
# An idea with a rung gets quality_scale.md in place of quality.md: the same text without "a small clean episode is the
# right answer" and with many visible events allowed as work rather than bookkeeping.
FAMILY_DIR, RUNG_DIR = HERE / 'prompts/v1/families', HERE / 'prompts/v1/rungs'
QUALITY_SCALE = 'v1/quality_scale.md'


def planner_slots(study):
    """The idea's drawn modifiers that carry a planner line, in a fixed order."""
    return tuple(sorted(m for m in study.get('idea_modifiers') or () if m in SLOT_BRIDGES))
V1_ROUTE_GUIDANCE = 'Plan the strongest fair episode this kernel supports, keeping what makes it distinct.'


def v1_planner_values(record, values):
    """v1 planner inputs: the selector's note on what makes the idea distinct, and a neutral planning note."""
    brief = values['EDITORIAL_BRIEF']
    note = (record.get('selection_note') or {}).get('useful_contribution') or brief.get('preserve')
    out = deepcopy(values)
    out['EDITORIAL_BRIEF'] = {'candidate_id': brief['candidate_id'], 'development_priority': brief.get('development_priority'),
                              'what_makes_it_distinct': note, 'development_notes': brief.get('develop'),
                              'main_authoring_risk': brief.get('avoid'),
                              'optional_development_suggestion': brief.get('optional_development_suggestion')}
    out['ROUTE_GUIDANCE'] = V1_ROUTE_GUIDANCE
    if record.get('private_route') == 'evidence_preserving_scale':
        out['OVERLAY_INVITATION'] = OVERLAY_INVITATIONS[record['study'].get('overlay', 'scale')]
    return out


def known_primary(primary):
    return primary in ARMS or (PROMPT_SET == 'v1' and primary in V1_BRIDGES)

ADJ_FIELDS = {'TASK_ID', 'EPISODE_HASH', 'AUTHOR_SPECIFICATION', 'COMPLETE_TASK',
              'BLIND_RESULTS', 'CONCERNS', 'TECHNICAL_EVIDENCE'}
ADJ_PREPARED_FIELDS = ADJ_FIELDS | {'REQUIRED_CONCERN_IDS'}
# v1 adjudications also carry REVIEW_POLICY: the departure package's review policy (2026-10-16), empty for every other
# episode (gate.departure_review), rendered at the end of prompts/v1/difficulty_adjudicator.md.
V1_ADJ_FIELDS = ADJ_FIELDS | {'REVIEW_POLICY'}
TECH_FIELDS = {'TASK_ID', 'VERIFIER_ID', 'PUBLIC_TASK', 'PUBLIC_CONTRACT', 'DIAGNOSTICS', 'REPAIR_CHALLENGE'}
REPAIR_FIELDS = {'CANDIDATE_ID', 'REPAIR_TYPE', 'REPAIR_INSTRUCTION', 'AUTHOR_SPECIFICATION',
                 'EVALUATOR_EVIDENCE', 'ALLOW_MECHANICAL_TEST_INPUT_RESTORATION'}


class Bindings(R4Bindings):
    def __init__(self):
        super().__init__()
        self.prompt_root = HERE / 'prompts'
        self.prompt_overrides['ambiguity_adjudicator'] = ('difficulty_adjudicator.md', adjudication.STAGE)
        self.prompt_overrides['difficulty_technical_recheck'] = ('difficulty_technical_recheck.md', recheck.STAGE)
        self.prompt_overrides['targeted_repair'] = ('bounded_repair.md', repair.STAGE)
        self.stage_bindings.pop(schemas.REPAIR_STAGE)
        self.stage_bindings[repair.STAGE] = StageBinding(repair.SCHEMA, repair.parse, ('targeted_repair',))
        self.stage_bindings[adjudication.STAGE] = StageBinding(adjudication.SCHEMA, adjudication.parse, ('ambiguity_adjudication',))
        self.stage_bindings[recheck.STAGE] = StageBinding(recheck.SCHEMA, recheck.parse, ('technical_verification',))
        self.prompt_set = PROMPT_SET
        self.territory = CARDS / 'prompts/ordinary_territory.planner.md'
        if PROMPT_SET == 'v1':
            self.prompt_overrides.update(V1_OVERRIDES)
            self.stage_bindings[repair_v1.STAGE] = StageBinding(repair_v1.SCHEMA, repair_v1.parse, ('targeted_repair',))
            if COLOUR_ROLES in ('choice', 'vary', 'mix'):  # 'vary': the planner varies colours wherever the rule allows
                if COLOUR_ROLES != 'mix':  # mix: chosen per idea in planning_templates
                    planner = 'v1/planner_colour_choice.md' if COLOUR_ROLES == 'choice' else VARY_PLANNER
                    self.prompt_overrides['concrete_planner'] = (planner, schemas.PLAN_STAGE)
                # repairs keep the neutral wording so a repair never recolours pairs it was not asked to change
                self.prompt_overrides['targeted_repair'] = ('v1/bounded_repair_colour_choice.md', repair_v1.STAGE)
            self.territory = HERE / 'prompts/v1/territory.planner.md'
            if SCIENCE.get('weakness'):  # weakness run (2026-10-20): realizer and repair stages that also record shortcut grids
                self.stage_bindings.update(lines.stage_bindings())

    def template(self, name):
        return load_prompt(self.prompt_root / self.prompt_overrides[name][0])

    def realizer_templates(self, study):
        """A weakness idea's realizer prompt: the unchanged realizer template, then the addenda asking for its shortcut
        grids or decision ledger (lines.realizer_components)."""
        names = lines.realizer_components(study)
        if not names or self.prompt_set != 'v1':
            raise SchemaError('$', 'realizer addenda outside a weakness idea')
        return [self.template('task_realizer')] + [lines.component(n) for n in names]

    def planning_templates(self, primary, route_names, colour=None, pose=False, slots=(), family=None, rung=None, canvas=None,
                           teach=None, depart=None, line=None, overlay=None, fence=None, structure=None):
        if not known_primary(primary):
            raise SchemaError('$', 'unknown local planning bridge')
        # R4 (weakness batch) is a line idea's scale line only: no family text, no general draw
        if (family or rung) and (self.prompt_set != 'v1' or rung not in (ALL_RUNGS if line and not family else RUNGS)
                                 or family not in (None, *FAMILIES)):
            raise SchemaError('$', 'planner family or rung outside the v1 targeted program')
        if slots and (self.prompt_set != 'v1' or set(slots) - SLOT_BRIDGES or list(slots) != sorted(set(slots))):
            raise SchemaError('$', 'planner slot lines outside the v1 property slots')
        planner = self.prompt_overrides['concrete_planner'][0]
        if colour is not None:  # the colour mix's per-idea instruction
            if self.prompt_set != 'v1' or COLOUR_ROLES != 'mix' or colour not in ('vary', 'v1'):
                raise SchemaError('$', 'per-idea colour instruction outside the colour mix')
            planner = VARY_PLANNER if colour == 'vary' else planner
        quality = QUALITY_SCALE if rung else self.prompt_overrides['v44_quality_plan'][0]
        rows = [load_prompt(self.prompt_root / planner), load_prompt(self.prompt_root / quality)]
        if self.prompt_set == 'v1':
            if primary in V1_BRIDGES:
                rows.append(component(HERE / 'prompts/v1/bridges' / V1_BRIDGES[primary]))
            elif primary not in V1_NO_BRIDGE and ARMS[primary].get('planner_bridge'):
                rows.append(component(CARDS / ARMS[primary]['planner_bridge']))
            rows.append(component(self.territory))  # the v1 formal line applies to every arm
            if pose:  # the per-idea pose instruction
                rows.append(component(HERE / 'prompts/v1/pose.planner.md'))
            rows += [component(SLOT_DIR / f'{s}.planner.md') for s in slots]  # the idea call's property slots
            if family:
                rows.append(component(FAMILY_DIR / f'{family}.planner.md'))
            if rung:  # last before any route addendum: it is binding and overrides size suggestions
                rows.append(component(RUNG_DIR / f'{family or "ANY"}.{rung}.planner.md'))
            if canvas:  # after the rung line: canvas shape and open background, which the rung's floors do not count as padding
                if canvas not in ('square', 'margin', 'both'):
                    raise SchemaError('$', 'unknown canvas line')
                rows.append(component(HERE / f'prompts/v1/canvas.{canvas}.planner.md'))
            if teach not in (None, 'plain', 'control'):
                raise SchemaError('$', 'unknown teaching line')
            if teach == 'plain':  # the binding teaching line (b17 review): pairs, plain laws, breadth over bookkeeping
                rows.append(component(HERE / 'prompts/v1' / TEACH_LINE))
            if depart not in (None, 'test', 'control'):
                raise SchemaError('$', 'unknown departure line')
            if depart == 'test':  # last of the binding lines: it overrides the clauses above for one situation only
                rows.append(component(HERE / 'prompts/v1/depart.planner.md'))
            # weakness run (2026-10-20): the EDGE or ROLES overlay line, then the idea's line, last of the binding lines
            if overlay is not None:
                if overlay not in WEAKNESS_OVERLAYS:
                    raise SchemaError('$', 'unknown weakness overlay')
                rows.append(component(lines.LINE_DIR / f'{overlay}.planner.md'))
            if line is not None:
                if line not in LINE_NAMES:
                    raise SchemaError('$', 'unknown weakness line')
                if rung == 'R4' and lines.spec(line).get('r4') is False:
                    raise SchemaError('$', 'this weakness line never takes the full-scale rung')
                rows.append(component(lines.LINE_DIR / f'{line}.planner.md'))
            if fence is not None:  # a relaxed fence's addendum, part of the line's binding text
                relax = ((lines.spec(line) or {}).get('relax') or {}).get(fence)
                if not relax:
                    raise SchemaError('$', 'relaxed fence outside its line')
                rows.append(lines.component(relax['text']))
            # structure lines (2026-10-22, STRUCTURE_SPEC): the second axis, composed after the content line (and its
            # fence), its text saying it yields to the line above
            if structure is not None:
                if structure not in STRUCTURES:
                    raise SchemaError('$', 'unknown structure line')
                rows.append(component(lines.LINE_DIR / f'{structure}.planner.md'))
            return rows + [load_prompt(self.prompt_root / self.prompt_overrides[name][0]) for name in route_names]
        bridge = ARMS[primary].get('planner_bridge')
        if bridge:
            rows.append(component(CARDS / bridge))
        if primary != 'FORM': rows.append(component(self.territory))
        return rows + [load_prompt(self.prompt_root / self.prompt_overrides[name][0]) for name in route_names]

    def prepare_values(self, prompt_name, values):
        if prompt_name == 'targeted_repair':
            if set(values) != REPAIR_FIELDS or type(values['ALLOW_MECHANICAL_TEST_INPUT_RESTORATION']) is not bool:
                raise SchemaError('$', 'repair received unapproved context')
            prepared = deepcopy(values)
            return prepared, {'profile_id': self.profile_id, 'prompt_name': prompt_name, 'values': prepared}
        if prompt_name == 'ambiguity_adjudicator':
            if set(values) != (V1_ADJ_FIELDS if self.prompt_set == 'v1' else ADJ_FIELDS):
                raise SchemaError('$', 'adjudication received unapproved context')
            prepared = deepcopy(values)
            prepared['REQUIRED_CONCERN_IDS'] = [r['concern_id'] for r in prepared['CONCERNS']]
        elif prompt_name == 'difficulty_technical_recheck':
            original = deepcopy(values); challenge = original.pop('REPAIR_CHALLENGE')
            prepared, _ = super().prepare_values('technical_verifier', original)
            prepared['REPAIR_CHALLENGE'] = deepcopy(challenge)
        else:
            return super().prepare_values(prompt_name, values)
        return prepared, {'profile_id': self.profile_id, 'prompt_name': prompt_name, 'values': deepcopy(prepared)}

    def validate_request(self, stage, prompt, context):
        if stage == lines.DEFAULT_STAGE:  # the default-reading call: public pairs and test inputs, the fixed template
            if (not isinstance(context, dict) or context.get('profile_id') != self.profile_id
                    or context.get('prompt_name') != 'default_reading' or set(context.get('values') or {}) != lines.DEFAULT_FIELDS):
                raise SchemaError('$', 'default reading requires its canonical context')
            public = context['values']['PUBLIC_TASK']
            if any('output' in pair for pair in public.get('test', [])):
                raise SchemaError('$', 'hidden output entered the default-reading call')
            public_diagnostics(public)
            rows = lines.default_templates()
            if (context.get('q2_template_hashes') != [row[2] for row in rows]
                    or prompt != render_prompt('\n\n'.join(row[0] for row in rows), context['values'])):
                raise SchemaError('$', 'default-reading prompt differs from its exact local components')
            return
        if stage == lines.REALIZER_STAGE:  # a weakness idea's realizer: the realizer template plus its addenda
            if (not isinstance(context, dict) or context.get('profile_id') != self.profile_id
                    or context.get('prompt_name') != 'task_realizer'):
                raise SchemaError('$', 'realization requires its canonical context')
            study = {'line': context.get('q2_line'),
                     'weak_overlays': {context['q2_overlay']: 'drawn'} if context.get('q2_overlay') else {}}
            if context.get('q2_structure'):
                study['structure'] = context['q2_structure']
            rows = self.realizer_templates(study)
            if (context.get('q2_template_hashes') != [row[2] for row in rows]
                    or prompt != render_prompt('\n\n'.join(row[0] for row in rows), context['values'])
                    or context.get('rendered_prompt_hash', content_hash(prompt)) != content_hash(prompt)):
                raise SchemaError('$', 'realizer prompt differs from its exact local components')
            from arcgen.lib.astra_authoring_contract_v3 import validate_plan
            validate_plan(context['values']['CONCRETE_PLAN'])
            return
        if stage in (repair.STAGE, repair_v1.STAGE, lines.REPAIR_STAGE):
            if (not isinstance(context, dict) or context.get('profile_id') != self.profile_id
                    or context.get('prompt_name') != 'targeted_repair' or set(context['values']) != REPAIR_FIELDS):
                raise SchemaError('$', 'bounded repair requires its original context')
            if type(context['values']['ALLOW_MECHANICAL_TEST_INPUT_RESTORATION']) is not bool:
                raise SchemaError('$', 'unbound mechanical restoration policy')
            template = self.template('targeted_repair')[0]
            if prompt != render_prompt(template, context['values']):
                raise SchemaError('$', 'repair prompt differs from its exact context')
            return
        routes = {schemas.PLAN_STAGE: (), schemas.LOGICAL_STAGE: ('logical_leap_planner_addendum',),
                  schemas.JOINT_STAGE: ('v43_episode_joint_plan',),
                  schemas.SCALE_STAGE: ('evidence_scale_addendum',)}
        if stage in routes:
            if (not isinstance(context, dict) or context.get('profile_id') != self.profile_id
                    or context.get('prompt_name') != 'concrete_planner'
                    or not known_primary(context.get('q2_primary'))):
                raise SchemaError('$', 'planning requires its explicit local bridge and canonical context')
            primary = context['q2_primary']
            specialist = {schemas.LOGICAL_STAGE: 'T', schemas.JOINT_STAGE: 'J'}
            if (stage in specialist and primary != specialist[stage]
                    or primary in ('J', 'T') and stage not in specialist):
                raise SchemaError('$', 'specialist source and planning route differ')
            rows = self.planning_templates(primary, routes[stage], context.get('q2_colour_roles'), context.get('q2_pose', False),
                                           tuple(context.get('q2_slots', ())), context.get('q2_family'), context.get('q2_rung'),
                                           context.get('q2_canvas'), context.get('q2_teach'), context.get('q2_depart'),
                                           context.get('q2_line'), context.get('q2_overlay'), context.get('q2_fence'),
                                           context.get('q2_structure'))
            expected = render_prompt('\n\n'.join(row[0] for row in rows), context['values'])
            # a re-plan below its rung's floors (rungs.py), missing its weakness line's fields (lines.py) or with more than
            # the one training pair a ONE idea plans (slots.py): one appended note
            if context.get('q2_replan') is not None:
                if (not (context.get('q2_rung') or context.get('q2_line') or context.get('q2_overlay') or context.get('q2_structure')
                         or _slots.CHECKED & set(context.get('q2_slots') or ()))
                        or not isinstance(context['q2_replan'], str)):
                    raise SchemaError('$', 're-plan note without a rung')
                expected += '\n\n' + context['q2_replan']
            if (context.get('q2_template_hashes') != [row[2] for row in rows]
                    or context.get('rendered_prompt_hash', content_hash(prompt)) != content_hash(prompt)
                    or prompt != expected):
                raise SchemaError('$', 'planning prompt differs from its exact local components')
            return
        if stage not in (adjudication.STAGE, recheck.STAGE):
            return super().validate_request(stage, prompt, context)
        name = 'ambiguity_adjudicator' if stage == adjudication.STAGE else 'difficulty_technical_recheck'
        if not context or context.get('profile_id') != self.profile_id or context.get('prompt_name') != name:
            raise SchemaError('$', 'new review requires its original versioned request context')
        values = context['values']
        if stage == adjudication.STAGE:
            fields = ADJ_PREPARED_FIELDS | ({'REVIEW_POLICY'} if self.prompt_set == 'v1' else set())
            if set(values) != fields or content_hash(values['COMPLETE_TASK']) != values['EPISODE_HASH']:
                raise SchemaError('$', 'adjudicator context or episode identity changed')
            if values['CONCERNS'] != adjudication.concerns(values['BLIND_RESULTS']):
                raise SchemaError('$', 'material concerns were removed or changed')
            if values['REQUIRED_CONCERN_IDS'] != [r['concern_id'] for r in values['CONCERNS']]:
                raise SchemaError('$', 'required concern checklist changed')
        else:
            if set(values) != TECH_FIELDS:
                raise SchemaError('$', 'technical recheck crossed its public boundary')
            validate_public_realization(values['PUBLIC_CONTRACT'], values['PUBLIC_TASK'])
            if values['DIAGNOSTICS'] != evidence.grouped_diagnostics(public_diagnostics(values['PUBLIC_TASK'])):
                raise SchemaError('$', 'technical diagnostics must be derived from public grids')
            challenge = values['REPAIR_CHALLENGE']
            if set(challenge) != {'policy', 'original_episode_hash', 'changed_training_pairs', 'concerns'} or challenge['policy'] != adjudication.POLICY:
                raise SchemaError('$', 'repair challenge includes unapproved fields')
            if not challenge['concerns'] or not challenge['changed_training_pairs']:
                raise SchemaError('$', 'repair has no identified public teaching change')
            ids = []
            for row in challenge['concerns']:
                from arcgen.astra_evidence_runtime_revision_20260911.structure import check
                check(row, adjudication.CHALLENGE)
                ids.append(row['concern_id'])
            if len(ids) != len(set(ids)):
                raise SchemaError('$', 'duplicate repair target')
            valid = {'train_' + str(i + 1) for i in range(len(values['PUBLIC_TASK']['train']))}
            if not set(challenge['changed_training_pairs']) <= valid:
                raise SchemaError('$', 'repair cites unknown changed demonstrations')
        template = self.template(name)[0]
        if prompt != render_prompt(template, values) or context.get('rendered_prompt_hash', content_hash(prompt)) != content_hash(prompt):
            raise SchemaError('$', 'review prompt differs from its exact versioned context')

    def validate_response(self, stage, value, context):
        if stage == lines.DEFAULT_STAGE:
            values = (context or {}).get('values') or {}
            if (value.get('task_id') != values.get('TASK_ID') or value.get('solver_id') != values.get('SOLVER_ID')
                    or len(value.get('predicted_test_outputs') or []) != len(values['PUBLIC_TASK']['test'])):
                raise SchemaError('$', 'default reading returned another identity or test count')
            return
        if stage == lines.REALIZER_STAGE:  # the ordinary realization is checked as always; its shortcut grids are recorded
            if not context or context.get('profile_id') != self.profile_id:
                raise SchemaError('$', 'realization response lacks its original context')
            from arcgen.lib.astra_authoring_contract_v3 import validate_realization
            validate_realization(context['values']['CONCRETE_PLAN'], lines.split(value)[0])
            return
        if stage in (repair.STAGE, repair_v1.STAGE, lines.REPAIR_STAGE):
            if not context or context.get('profile_id') != self.profile_id:
                raise SchemaError('$', 'repair response lacks its original context')
            if stage == lines.REPAIR_STAGE:
                repair_v1.expand(lines.split(value)[0], context['values'])
            else:
                (repair_v1 if stage == repair_v1.STAGE else repair).expand(value, context['values'])
            return
        if stage not in (adjudication.STAGE, recheck.STAGE):
            return super().validate_response(stage, value, context)
        if not context or context.get('profile_id') != self.profile_id:
            raise SchemaError('$', 'review response lacks its original context')
        (adjudication if stage == adjudication.STAGE else recheck).validate_response(value, context['values'])


def bindings(*, palette_allocator=None):
    b = Bindings()
    def palette(plan, realization, target_background, palette_config):
        result = (palette_allocator.materialize(b.materialize_palette, plan, realization, target_background, palette_config)
                  if palette_allocator is not None else
                  b.materialize_palette(plan, realization, target_background=target_background, palette_config=palette_config))
        return result['plan'], result['realization'], result
    # New review stages use the adapter's explicit render path; the frozen
    # factory's authoring-only prompt whitelist remains unchanged.
    overrides = {k: v for k, v in b.prompt_overrides.items()
                 if k not in ('ambiguity_adjudicator', 'difficulty_technical_recheck')}
    return {'profile_id': b.profile_id, 'prompt_overrides': overrides,
            'stage_bindings': {k: v for k, v in b.stage_bindings.items()
                               if k not in (adjudication.STAGE, recheck.STAGE)}, 'prepare_values': b.prepare_values,
            'validate_request': b.validate_request, 'validate_response': b.validate_response,
            'materialize_palette': palette}
