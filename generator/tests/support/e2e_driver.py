"""End-to-end driver: one batch of the reference design through the whole pipeline with scripted model responses.

  python tests/support/e2e_driver.py setup     a sample bank, a stand-in Codex executable, and `bin/arcgen new e2e` with a
                                               two-window design (waves A and B, eight explore calls)
  python tests/support/e2e_driver.py run       prepare, qualify/ready, authorize and run the batch (mock transport),
                                               then write its summary

Run each step in its own process with ARCGEN_WORKSPACE and ARCGEN_BANK set (the test does); `run` also needs
ARCGEN_CONFIG pointing at the batch profile. No network: sockets are refused, embeddings are the local hashing provider,
and every model call is answered here. The responses follow the clear path: every idea window keeps one idea, and
each kept idea is planned (plain or presentation-overlay route), realized, checked and solved by the blind solver on
the first try. (The two kept ideas get the same fixture grids, so the batch's exact-duplicate audit keeps only one.)
"""
import json
import os
import socket
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
sys.path[:0] = [str(REPO), str(TESTS)]
NAME = 'e2e'


def no_network(*args, **kwargs):
    raise RuntimeError('end-to-end smoke test attempted a network connection')


socket.socket.connect = no_network
socket.create_connection = no_network


def setup():
    from arcgen.paths import WORKSPACE, BANK
    from support import commit_fixtures as fx
    assert BANK == WORKSPACE / 'committed_tasks', 'set ARCGEN_BANK to <workspace>/committed_tasks'
    fx.make_bank(WORKSPACE, 12, render=True, blob=64)
    codex = WORKSPACE / 'bin/codex'  # never run: the batch runs on the mock transport
    codex.parent.mkdir(parents=True, exist_ok=True)
    codex.write_text('#!/bin/sh\necho "stand-in Codex executable for the offline smoke test" >&2\nexit 1\n')
    codex.chmod(0o755)
    from arcgen.batch import cli
    profile = json.loads(cli.TEMPLATE_PROFILE.read_text())
    profile['operations'].update(codex_executable=str(codex), workers=8)
    profile['science']['embedding'] = {'provider': 'local', 'dimensions': 64}
    # eight calls cannot hold the reference design's bare-call reservations; every other property share is kept
    profile['science'].update(modifier_bare={}, modifier_arm_caps={})
    template = WORKSPACE / 'e2e-template.json'
    template.write_text(json.dumps(profile, indent=1))
    spec = {'revision': 'arcgen-e2e', 'seed': 1,
            'windows': {'supply-01': {'account': 'main', 'wave': 'A', 'reference_wave': 'A', 'context': 'c01', 'commit': 'explore'},
                        'supply-02': {'account': 'main', 'wave': 'A', 'reference_wave': 'B', 'context': 'c02', 'commit': 'explore'}},
            'rows': [['supply-01', 'SEE', None, 'L', 'sol', 'high'], ['supply-01', 'ABS', None, 'L', 'sol', 'max'],
                     ['supply-01', 'LEVELS', None, 'L', 'sol', 'xhigh'], ['supply-01', 'OPEN', None, 'L', 'sol6', 'xhigh'],
                     ['supply-02', 'STAGES', None, 'L', 'sol', 'high'], ['supply-02', 'CONTRAST', None, 'L', 'sol', 'max'],
                     ['supply-02', 'SPREAD', None, 'L', 'sol', 'high'], ['supply-02', 'GEO', None, 'L', 'sol6', 'xhigh']]}
    spec_path = WORKSPACE / 'e2e-spec.json'
    spec_path.write_text(json.dumps(spec, indent=1))
    cli.new(NAME, spec=spec_path, profile=template, workers=8)


def responder():
    """FactoryModelInvoker._mock_response for every stage of the clear path, plus the transport receipt each call
    leaves on the real transport (the commit checks it)."""
    import threading
    from copy import deepcopy
    from arcgen.lib.util import content_hash
    from arcgen.lib import astra_authoring_schemas_v3 as s
    from arcgen.astra_evidence_runtime_revision_20260911 import evidence
    from arcgen.astra_usage_diversity_20260913 import selection
    from arcgen.batch import idea_v1, runtime
    from arcgen.batch.config import ACCOUNT
    from support.fixtures import episode, technical, blind
    runtimes, lock = {}, threading.Lock()
    original = runtime.make_runtime

    def capture(*args, **kwargs):
        rt = original(*args, **kwargs)
        with lock:
            runtimes[str(rt.calls_dir)] = rt
        return rt
    runtime.make_runtime = capture
    from arcgen.batch import pipeline  # it calls make_runtime through its own import
    pipeline.make_runtime = capture

    def receipt(invoker, stage, cid, model):
        directory = invoker.calls_dir / stage / cid
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'program_transport_identity.json').write_text(json.dumps(
            {'expected': {'model_name': model}, 'account': ACCOUNT, 'observations': [],
             'verified': True}))

    def kernels(prefix, n):
        return [{'id': f'{prefix}{i:02d}', 'title': f'{prefix} idea {i}', 'idea': f'A relation {prefix}{i}.',
                 'recipe': f'Do step {prefix}{i} to every region.'} for i in range(1, n + 1)]

    def answer(invoker, stage, cid):
        directory = invoker.calls_dir / stage / cid
        request = json.loads((directory / 'request.json').read_text())
        receipt(invoker, stage, cid, request['model']['name'])
        if stage == 'factory_transport_probe':
            return json.dumps({'probe_id': cid, 'ok': True})
        if stage == idea_v1.STAGE_SKETCH:
            return json.dumps({'sketches': kernels('s', 14)})
        if stage == idea_v1.STAGE_EXPLORE:
            return json.dumps({'sketches': kernels('n', 6)})
        if stage == idea_v1.STAGE_COMMIT_KEPT:
            # keep two drafts; make each relation unique to its call so no two ideas are exact duplicates
            verdicts = [{'id': f's{i:02d}', 'verdict': 'keep' if i <= 2 else 'drop', 'closest_existing': 'none'} for i in range(1, 15)]
            verdicts += [{'id': f'x{i:02d}', 'verdict': 'drop', 'closest_existing': 'none'} for i in range(1, 7)]
            return json.dumps({'sketch_verdicts': verdicts, 'kernels': [
                {'source': f's{i:02d}', 'central_relation': f'Relation {i} of call {cid}.', 'recipe': f'Recipe {i}.',
                 'what_is_new': 'A new insight.', 'why_coherent': 'Coherent.', 'possible_test_generalization': 'A new situation.',
                 'main_authoring_risk': 'Ambiguity.'} for i in (1, 2)]})
        if stage == selection.STAGE:
            ids = request['parent_ids']; keep = ids[:1]  # one idea per window goes on to development
            return json.dumps({'selector_id': cid, 'window_observations': None,
                'candidate_assessments': [{'candidate_id': k, 'eligibility': 'plausible' if k in keep else 'ineligible',
                    'conceptual_potential': 'solid', 'coherence_potential': 'solid', 'fair_realization_potential': 'plausible',
                    'redundant_with': [], 'same_procedure_family_with': [], 'bank_comparisons': [],
                    'useful_contribution': 'Fresh insight ' + k, 'reason': 'Coherent synthetic relation.'} for k in ids],
                'ranked_plausible': [{'candidate_id': k, 'development_priority': i + 1, 'development_brief': 'Realize fairly.',
                    'optional_development_suggestion': None} for i, k in enumerate(keep)]})
        rt = runtimes[str(invoker.calls_dir)]
        tid = request['parent_ids'][0]
        plan, realized = episode(tid)
        if stage == s.PLAN_STAGE:
            return json.dumps(plan)
        if stage == s.SCALE_STAGE:  # an idea drawn for a presentation overlay
            return json.dumps({'candidate_id': tid, 'scale_decision': 'kept_compact',
                               'scale_reason': 'The compact episode already shows the relation.', 'concrete_plan': plan})
        if stage == s.REALIZER_STAGE:
            return json.dumps(realized)
        record = rt.state['semantic_records'][tid]
        outputs = deepcopy(record['realization']['authoring_metadata']['expected_test_outputs'])
        if stage == evidence.STAGE:
            value = technical(tid); suffix = cid[len(tid) + 1:]
            value.update(verifier_id='verifier-' + content_hash({'task': tid, 'suffix': suffix})[:12],
                         derived_test_outputs=outputs, derivation_failure_reason='')
            groups = evidence.grouped_diagnostics(record['technical_diagnostics'])['flags']
            value['diagnostic_flag_assessments'] = [dict(flag_code=g['code'], diagnostic_observation='confirmed',
                retention_decision='acceptable', evidentially_necessary=False, assessment='Checked in this fixture.',
                reviewed_observation_ids=[o['observation_id'] for o in g['observations']]) for g in groups]
            return json.dumps(value)
        if stage == 'v4_blind_solver':
            suffix = cid[len(tid) + 1:]; value = blind(tid, rt.factory_config['run']['id'])
            value.update(task_id='blind-' + content_hash({'seed': rt.config['seed'], 'task': tid})[:12],
                         solver_id='solver-' + content_hash({'task': tid, 'suffix': suffix})[:12], predicted_test_outputs=outputs,
                         serious_different_output_alternative=False, test_requires_unsupported_guessing=False,
                         strongest_plausible_alternatives=[])
            return json.dumps(value)
        raise AssertionError('no scripted response for stage ' + stage)
    return answer


def run():
    import arcgen.lib.factory_context as context
    context._tokenizer = lambda model: (None, {'tokenizer_available': False})
    from arcgen.lib.factory_runtime import FactoryModelInvoker
    FactoryModelInvoker._mock_response = responder()
    FactoryModelInvoker._codex_response = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('real Codex call attempted'))
    from arcgen.paths import WORKSPACE
    from arcgen.batch import next_batch, cli
    root = WORKSPACE / 'runs' / NAME
    os.environ['ARCGEN_SKIP_TEST_GATE'] = '1'
    print(json.dumps(next_batch.prepare(root)['status']))
    print(json.dumps(next_batch.ready(root)['status']))
    print(json.dumps(next_batch.authorize(root, authorization='Offline end-to-end smoke test.', mock=True)['status']))
    result = next_batch.run(root, mock=True)
    print(json.dumps({'run_status': result.get('status')}))
    print(json.dumps({'summary': cli.summary(NAME, write_files=True)['verified_tasks']}))


if __name__ == '__main__':
    {'setup': setup, 'run': run}[sys.argv[1]]()
