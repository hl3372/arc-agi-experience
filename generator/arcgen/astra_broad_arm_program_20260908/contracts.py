"""Exact q2 prompt compositions around the frozen canonical authoring contract."""
from copy import deepcopy
from arcgen.lib.astra_contract_bindings_v3 import AstraContractBindings
from arcgen.lib import astra_authoring_schemas_v3 as s
from arcgen.lib.schemas import SchemaError
from arcgen.lib.util import load_prompt, render_prompt, content_hash
from .common import CARDS, SOURCE, read, component

ARMS = {a['id']:a for a in read(CARDS/'catalog.proposed.json')['arms']}

class Q2Bindings(AstraContractBindings):
    def __init__(self):
        super().__init__(SOURCE/'prompts')
        self.prompt_overrides = deepcopy(self.prompt_overrides)
        self.prompt_overrides['concrete_planner'] = ('broad_q2_planner.md',s.PLAN_STAGE)
        self.prompt_overrides['v44_quality_plan'] = ('broad_q2_quality.md',s.PLAN_STAGE)

    def planning_templates(self, primary, route_names):
        if primary not in ARMS: raise SchemaError('$','unknown frozen planning bridge')
        rows = [load_prompt(self.prompt_root/self.prompt_overrides[n][0])
                for n in ('concrete_planner','v44_quality_plan')]
        card = ARMS[primary].get('planner_bridge')
        if card: rows.append(component(CARDS/card))
        rows += [load_prompt(self.prompt_root/self.prompt_overrides[n][0]) for n in route_names]
        return rows

    def validate_request(self, stage, rendered_prompt, request_context):
        if stage not in (s.PLAN_STAGE,s.LOGICAL_STAGE,s.JOINT_STAGE,s.SCALE_STAGE):
            return super().validate_request(stage,rendered_prompt,request_context)
        c = request_context
        if not isinstance(c,dict) or c.get('profile_id') != self.profile_id or c.get('prompt_name') != 'concrete_planner':
            raise SchemaError('$','planning requires bound canonical context')
        if c.get('rendered_prompt_hash',content_hash(rendered_prompt)) != content_hash(rendered_prompt):
            raise SchemaError('$','planning request hash mismatch')
        routes = {s.PLAN_STAGE:(),s.LOGICAL_STAGE:('logical_leap_planner_addendum',),
            s.JOINT_STAGE:('v43_episode_joint_plan',),s.SCALE_STAGE:('evidence_scale_addendum',)}
        primary = c.get('q2_primary')
        if primary not in ARMS: raise SchemaError('$','missing explicit frozen bridge identity')
        if stage==s.LOGICAL_STAGE and primary!='T' or stage==s.JOINT_STAGE and primary!='J':
            raise SchemaError('$','specialist route lost its source semantics')
        if primary in ('J','T') and stage not in (s.LOGICAL_STAGE,s.JOINT_STAGE):
            raise SchemaError('$','specialist source cannot acquire ordinary or scale bridge')
        rows = self.planning_templates(primary,routes[stage])
        if c.get('q2_template_hashes') != [r[2] for r in rows]:
            raise SchemaError('$','frozen planning components changed')
        if rendered_prompt != render_prompt('\n\n'.join(r[0] for r in rows), c['values']):
            raise SchemaError('$','rendered request differs from exact q2 components and values')

def bindings():
    q = Q2Bindings()
    def palette(plan, realization, target_background, palette_config):
        result = q.materialize_palette(plan,realization,target_background=target_background,palette_config=palette_config)
        return result['plan'],result['realization'],result
    return {'profile_id':q.profile_id,'prompt_overrides':q.prompt_overrides,
        'stage_bindings':q.stage_bindings,'prepare_values':q.prepare_values,
        'validate_request':q.validate_request,'validate_response':q.validate_response,'materialize_palette':palette}
