"""Answer-withheld recheck which cannot silently drop prior technical concerns."""
from copy import deepcopy
from arcgen.lib.schemas import SchemaError, parse_json_object
from arcgen.lib.util import content_hash
from arcgen.lib.v4_schemas import _transport_schema
from arcgen.lib.factory_runtime import StageBinding
from arcgen.astra_evidence_runtime_revision_20260911 import evidence
from arcgen.astra_evidence_runtime_revision_20260911.structure import check

STAGE = 'sol_astra_technical_recheck_v1'
TEXT = {'type':'string','minLength':1,'maxLength':6000}
STRICT_SCHEMA = deepcopy(evidence.STRICT_SCHEMA)
STRICT_SCHEMA['properties']['prior_concern_resolutions'] = {'type':'array','items':{
    'type':'object','additionalProperties':False,'required':['concern_id','resolution','evidence'],
    'properties':{'concern_id':TEXT,'resolution':{'type':'string','enum':['resolved','confirmed_defect','unresolved']},'evidence':TEXT}}}
STRICT_SCHEMA['required'].append('prior_concern_resolutions')
SCHEMA = _transport_schema(STRICT_SCHEMA)


def parse(data):
    v = parse_json_object(data) if isinstance(data,(str,bytes,bytearray)) else deepcopy(data)
    strict=deepcopy(v)
    for row in strict.get('diagnostic_flag_assessments',[]): row.pop('justified',None)
    check(strict,STRICT_SCHEMA)
    resolutions=v.pop('prior_concern_resolutions')
    v=evidence.parse(v);v['prior_concern_resolutions']=resolutions
    ids=[r['concern_id'] for r in resolutions]
    if len(set(ids))!=len(ids): raise SchemaError('$','repeated technical concern')
    if v['status']=='pass' and any(r['resolution']!='resolved' for r in resolutions):
        raise SchemaError('$','technical pass leaves a prior concern unresolved')
    return v


def concerns(evaluation):
    response=evaluation['response']
    # These fields came from a verifier which never saw author test outputs.
    # Predicted arrays and match-location feedback are deliberately excluded.
    fields=('status','mechanically_correct','test_supported','one_answer_supported','plan_fidelity',
            'unapproved_realizer_decisions','evidence_sufficiency','redundancy_assessment','pair_economy',
            'composition_preserved','serious_concrete_shortcuts','canvas_use','arbitrary_exceptions',
            'unsupported_test_semantics','diagnostic_flag_assessments','repair_instruction','confidence',
            'derivation_failure_reason')
    return [{'concern_id':'prior-technical-'+content_hash(response)[:16],
             'observations':{k:deepcopy(response[k]) for k in fields if k in response},
             'independent_rederivation_required':True}]


def validate_response(value, values):
    value=parse(value)
    plain=deepcopy(value);plain.pop('prior_concern_resolutions')
    public_values={k:v for k,v in values.items() if k!='TECHNICAL_CONCERNS'}
    evidence.validate_response(plain,public_values)
    if value['task_id']!=values['TASK_ID'] or value['verifier_id']!=values['VERIFIER_ID']:
        raise SchemaError('$','technical recheck identity changed')
    if {x['concern_id'] for x in value['prior_concern_resolutions']} != {x['concern_id'] for x in values['TECHNICAL_CONCERNS']}:
        raise SchemaError('$','technical recheck must address every supplied concern')


def stage_binding(): return StageBinding(SCHEMA,parse,('technical_verification',))
