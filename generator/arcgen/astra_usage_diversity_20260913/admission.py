"""Native admission with palette decisions deferred to the shared allocator."""
from copy import deepcopy
from arcgen.lib.factory_authoring_screen import initial_record
from arcgen.astra_targeted_followup_20260911.admission import assign_routes
from .common import frozen, digest, IntegrityError


def prepare(runtime, window, allocation):
    rows = deepcopy(allocation['rows'])
    routed = assign_routes(runtime.factory_config, rows) if rows else []
    result = []
    for row in routed:
        row['slate_index'] = 1
        tid = 'd'+digest({'window': window['id'], 'source': row['source_candidate_id']})[:19]
        initial = initial_record({'task_id': tid, 'source_candidate_id': row['source_candidate_id'],
            'lineage_id': row['lineage_id'], 'arm': 'single', 'private_source': row['private_source'],
            'private_conditioned': row['private_conditioned'], 'private_source_call_id': row['private_source_call_id'],
            'private_idea_effort': row['private_effort'], 'private_route': row['private_route'],
            'slate_index': 1, 'source_development_priority': row['study']['dispatch_ordinal'],
            'search_mode': row['search_mode'], 'kernel': row['kernel'],
            'editorial_brief': row['editorial_brief'], 'novelty': row['novelty']})
        initial['study'] = deepcopy(row['study'])
        runtime.budget.reserve_development(tid, runtime.phase, window['id'], digest(initial), window['maximum_developments'])
        runtime.palette_allocator.register(tid, window['account'], window['id'], digest(initial))
        result.append({'task_id': tid, 'initial': initial, 'selected': row})
    # None is an explicit deferred target, not a request to skip the new hook.
    manifest = frozen(runtime.run_dir/'admission/manifest.private.json', {
        'window': window, 'rows': result, 'assignments': {r['task_id']: None for r in result},
        'palette_allocation': 'batch-background-mixture-v1/deferred',
        'palette_context_hash': runtime.palette_allocator.context_hash,
        'allocation_hash': digest(allocation), 'source_manifest_hash': digest(runtime.source_manifest)})
    if set(runtime.state['semantic_records']) - {r['task_id'] for r in result}:
        raise IntegrityError('foreign candidate in window runtime')
    for row in result:
        previous = runtime.state['semantic_records'].get(row['task_id'])
        if previous:
            for key in ('task_id', 'source_candidate_id', 'private_source', 'private_source_call_id', 'study'):
                if previous[key] != row['initial'][key]:
                    raise IntegrityError('resumed candidate attribution changed')
        else:
            runtime.state['semantic_records'][row['task_id']] = deepcopy(row['initial'])
    runtime.save()
    return manifest
