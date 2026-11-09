"""Board-independent yield, duplicate checks, watch cases and separate-answer gallery."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from html import escape
import shutil
from arcgen.lib.util import content_hash
from arcgen.lib.v41_palette import palette_equivalent_hash
from arcgen.lib.e18_evidence_gate import evidence_is_clear
from .common import read, ref, digest, atomic, frozen, IntegrityError, isolated_root, verify_refs
from .budget import Budget
from .downstream import verify, verify_bypass


def consistency_watch(record):
    """Observable patterns only; no claim that prose or either field is correct."""
    hits = []
    for i, review in enumerate(record.get('technical_results') or []):
        response = review.get('response') or {}
        if response.get('status') == 'pass' and review.get('unresolved_pair_economy_flags'):
            hits.append({'stage': 'technical', 'review_index': i,
                         'pattern': 'pass_with_unresolved_diagnostic',
                         'flags': review['unresolved_pair_economy_flags'],
                         'assessments': response.get('diagnostic_flag_assessments', [])})
    adjudication = record.get('ambiguity_adjudication') or {}
    if adjudication.get('decision') == 'clear' and adjudication.get('comparably_simple_different_output_rule') is True:
        hits.append({'stage': 'adjudication', 'pattern': 'clear_with_competing_rule_field',
                     'response': adjudication})
    return hits


def duplicate_dispositions(tasks, bank_tasks):
    """Canonical order; held candidates cannot displace a usable representative."""
    exact = {r['exact_hash']: 'bank:'+r['registry_task_id'] for r in bank_tasks}
    palette = {r['palette_hash']: 'bank:'+r['registry_task_id'] for r in bank_tasks}
    result = {}
    for row in tasks:
        h, p = row.get('exact_hash'), row.get('palette_hash')
        e, c = exact.get(h), palette.get(p)
        usable = row['verified_before_union'] and not e and not c
        result[row['task_id']] = {'exact_duplicate_of': e, 'palette_duplicate_of': c,
                                  'usable_after_exact_palette_union': usable}
        if usable:
            exact[h] = 'batch:'+row['task_id']; palette[p] = 'batch:'+row['task_id']
    return result


def _gallery(root, tasks):
    colors = ['#000000','#0074d9','#ff4136','#2ecc40','#ffdc00','#aaaaaa','#f012be','#ff851b','#7fdbff','#870c25']
    def grid(value):
        h, w = len(value), len(value[0]); cells=[]
        for y,row in enumerate(value):
            for x,color in enumerate(row):
                cells.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{colors[color]}"/>')
        return f'<svg role="img" aria-label="{h} by {w} grid" viewBox="0 0 {w} {h}" width="{min(300,w*16)}" height="{min(300,h*16)}">'+''.join(cells)+'</svg>'
    public, answers = [], []
    for row in tasks:
        if not row['realized']: continue
        task = read(row['task_file'])
        title = escape(row['task_id'])
        label=escape(row.get('blind_performance',{}).get('label') or '')
        body = ''.join('<div class="pair">'+grid(p['input'])+'<span>→</span>'+grid(p['output'])+'</div>' for p in task['train'])
        body += '<p>Test inputs</p>'+''.join(grid(p['input']) for p in task['test'])
        public.append(f'<article id="{title}"><h2>{title}</h2><p>{label}</p>{body}<p><a href="answers.html#{title}">Open answers and rule</a></p></article>')
        answers.append(f'<article id="{title}"><h2>{title}</h2><p>{escape(row["rule"])}</p>'+''.join(grid(p['output']) for p in task['test'])+'</article>')
    prefix='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#fafafa;color:#171717}article{border-top:1px solid #aaa;padding:16px}svg{margin:8px;image-rendering:pixelated}.pair{display:flex;align-items:center;gap:16px;flex-wrap:wrap}</style>'
    gallery=root/'gallery';gallery.mkdir(exist_ok=True)
    (gallery/'index.html').write_text(prefix+'<title>Task gallery</title><h1>Task gallery</h1><p>Training pairs and test inputs. Answers are in a separate file.</p>'+''.join(public))
    (gallery/'answers.html').write_text(prefix+'<title>Answers and rules</title><h1>Answers and rules</h1>'+''.join(answers))
    return {'public':ref(gallery/'index.html'), 'answers':ref(gallery/'answers.html')}


def planner_comparison(tasks):
    """Within-kernel Astra xhigh vs GPT-6.1 Sol xhigh vs Sol max planning outcomes; descriptive."""
    from .routing import PLANNERS
    fields = ('task_id', 'status', 'planned', 'realized', 'evidence_clear', 'verified_before_union',
              'usable_after_exact_palette_union', 'repaired', 'hold_kind', 'confirmed_rival',
              'original_primary_result', 'review_routes')
    pairs = {}
    for t in tasks:
        if t.get('planner_paired'): pairs.setdefault(t['planner_pair_id'], {})[t['planner']] = {k: t.get(k) for k in fields}
    rows = [{'kernel_id': k, **{p: v.get(p) for p in PLANNERS}} for k, v in sorted(pairs.items())]
    def count(planner, key):
        return sum(bool((r[planner] or {}).get(key)) for r in rows)
    outcome = lambda r, p: bool((r[p] or {}).get('verified_before_union'))
    others = [p for p in PLANNERS if p != 'astra']
    return {'design': 'triplet_random_half_within_window', 'planners': PLANNERS, 'pairs': len(rows), 'rows': rows,
            'complete_sets': sum(all(r[p] is not None for p in PLANNERS) for r in rows),
            'by_planner': {p: {k: count(p, k) for k in ('planned', 'realized', 'evidence_clear', 'verified_before_union', 'repaired')}
                           for p in PLANNERS},
            'discordant_verified_vs_astra': {p: {'astra_only': sum(outcome(r, 'astra') and not outcome(r, p) for r in rows),
                                                 p + '_only': sum(outcome(r, p) and not outcome(r, 'astra') for r in rows)}
                                             for p in others},
            'blind_solve_rate_ranks_planners': False,
            'pending': ['blinded side-by-side review', 'planner token/time accounting from call ledgers']}


def research_index(root, phase, tasks):
    """Make disabled research explicit; no fabricated judgments or production edits."""
    index = {r['task_id']: {'arm':r['arm'], 'invitation':r['invitation'], 'cohort':r['cohort'],
        'bank_setting':r['bank_setting'], 'admission':r['admission'], 'natural_eligible':r['natural_eligible'],
        'rule':r['rule'], 'research':{}} for r in tasks if r['realized']}
    for row in index.values(): row['research_status']='disabled_by_design'
    path=root/'analysis/research_index.private.json'; frozen(path,index)
    return {'status':'disabled_by_design','index':ref(path),'group_statuses':{},
        'fully_reviewed_tasks':0,'unreviewed_or_incomplete_tasks':[],
        'research_disabled_tasks':list(index),'scheduled_calls':0}


def queued_sources(root,plan):
    """Include frozen saved eligibility even before any runtime has started."""
    rows={}
    for w in plan['windows']:
        directory=root/'accounts'/w['account']/w['id']
        paths=[directory/'allocation.private.json',directory/'eligible_queue.private.json']
        if w.get('kind')=='saved_development':
            paths.insert(0,root/'inputs/saved'/w['id']/'allocation.private.json')
        for path in paths:
            if path.exists():
                for row in read(path).get('rows',[]):rows[row['source_candidate_id']]=row
    return rows


def link_kernels(kernels, tasks):
    """Each kernel's development: its Astra (or unrecorded-planner) task where the r23 triplets planned one kernel three
    times, else its task under any planner (v1 plans with Sol only; v1-b14 reported all 472 as queued_unstarted)."""
    task_by_kernel={}
    for t in sorted(tasks,key=lambda t:(t.get('planner') not in (None,'astra'),t['task_id'])):
        task_by_kernel.setdefault(t['kernel_id'],t)
    for k in kernels:
        t=task_by_kernel.get(k['kernel_id'])
        k.update(quality_eligible=k['admitted'],development_started=t is not None,
            task_id=t['task_id'] if t else None,
            queued_unstarted=k['admitted'] and t is None)
    return kernels


LEDGERS=('budget.sqlite3','budget_limits.private.json','palette.sqlite3','inputs/palette_context.private.json')


def snapshot_ledgers(root, out):
    """Copies of the shared ledgers: opening one takes a write lock, and a stopped run is only read."""
    if any(root.glob('*-wal')):raise IntegrityError('a ledger has an open write-ahead log; the run may be alive')
    rows=[]
    for name in LEDGERS:
        target=out/'ledger_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(root/name,target)
        rows.append({'source':ref(root/name),'copy':ref(target)})
    if any(r['source']['sha256']!=r['copy']['sha256'] for r in rows):raise IntegrityError('a ledger changed while it was copied')
    return rows


def report(root, *, out=None):
    """With out: a stopped run's finished windows, reported into out from ledger copies; nothing is written under root."""
    root=isolated_root(root); dest=Path(out).resolve() if out else root
    if out and (dest.is_relative_to(root) or root.is_relative_to(dest)):raise IntegrityError('a partial report must be written outside the run')
    ledgers=snapshot_ledgers(root,dest) if out else None; ledger=dest/'ledger_snapshot' if out else root
    plan=deepcopy(read(root/'design/batch.private.json'))
    for name in ('all',):
        path=root/'design/execution'/(name+'.private.json')
        if path.exists():plan['windows'].extend(read(path)['groups'])
    from .scheduler import load_assignments
    assigned=load_assignments(root)
    for w in plan['windows']:
        if w.get('account') is None and w['id'] in assigned: w['account']=assigned[w['id']]
    plan['windows']=[w for w in plan['windows'] if w.get('account')]  # never-started dynamic groups have no state
    inputs=read(root/'inputs/revision_inputs.private.json')
    tasks, kernels, windows, sources = [], [], [], []
    queue_rows = queued_sources(root,plan)
    for w in plan['windows']:
        directory=root/'accounts'/w['account']/w['id']
        source_path=directory/'source_rows.private.json'
        source=read(source_path) if source_path.exists() else {}
        if source_path.exists(): sources.append(ref(source_path))
        allocation_path=directory/'allocation.private.json'
        allocation=read(allocation_path) if allocation_path.exists() else {'rows':[]}
        admitted=queue_rows
        packed_path=directory/'selector_inputs.private.json'
        packed=read(packed_path) if packed_path.exists() else {}
        selector_state=directory/'selector/state.json'
        selector=read(selector_state) if selector_state.exists() else {}
        response={}
        if w.get('kind') == 'saved_development':
            saved = read(root/'inputs/saved'/w['id']/'manifest.private.json')
            source = read(saved['source_rows']['path'])
            response = read(saved['selector_response']['path'])
            packed = {'exact_conflicts': saved['fresh_exact_conflicts']}
        for call in selector.get('call_ledger',{}).values():
            if call.get('purpose')=='full_window_selection' and call.get('status')=='complete':
                p=directory/'selector/model_calls'/call['stage']/call['call_id']/'parsed_response.json'
                response=read(p); sources.append(ref(p))
        assessments={r['candidate_id']:r for r in response.get('candidate_assessments',[])}
        ranks={r['candidate_id']:r['development_priority'] for r in response.get('ranked_plausible',[])}
        report_calls = w['calls']
        if w.get('kind') == 'saved_ideas':
            if not source:  # not prepared yet: the saved file holds the same rows
                from .saved_ideas import load
                source = load(w)['source_rows']
            report_calls = [{'opaque_kernel_ids': [kid], 'arm': raw['study']['primary_arm'],
                'invitation': raw['study']['invitation'], 'bank_setting': raw['study']['bank_setting'],
                'effort': raw['private_effort'], 'recipe_id': raw['study']['recipe_id'],
                'idea_model': raw['study']['generation_model'], 'comparison_block': raw['study'].get('comparison_block'),
                'comparison_role': raw['study'].get('comparison_role')} for kid, raw in sorted(source.items())]
        if w.get('kind') == 'saved_development':
            report_calls = [{'opaque_kernel_ids': [kid], 'arm': raw['study']['primary_arm'],
                'invitation': raw['study']['invitation'], 'bank_setting': raw['study']['bank_setting'],
                'effort': raw['private_effort'], 'recipe_id': raw['study']['recipe_id'],
                'pinning':raw['study'].get('pinning','ordinary'), 'idea_model':'gpt-6-astra'} for kid, raw in source.items()]
        for call in report_calls:
            for kid in call['opaque_kernel_ids']:
                kernels.append({'kernel_id':kid, 'window':w['id'], 'account':w['account'],
                    'cohort': w.get('cohort', 'fresh_model_comparison'),
                    'reused_generation': w.get('kind') in ('saved_development', 'saved_ideas'),
                    'generation_provenance': source.get(kid, {}).get('study'),
                    'arm':call['arm'], 'invitation':call['invitation'], 'bank_setting':call['bank_setting'],
                    'effort':call['effort'], 'recipe_id':call['recipe_id'], 'generated':kid in source,
                    'generation_model':call['idea_model'], 'comparison_block':call.get('comparison_block'),
                    'comparison_role':call.get('comparison_role'),
                    'pinning':source.get(kid,{}).get('study',{}).get('pinning',call.get('pinning','unconditioned' if call['bank_setting']=='unconditioned' else 'pinned')), 'pin_pair_id':call.get('pin_pair_id'), 'context':w['context'],
                    'exact_conflict':packed.get('exact_conflicts',{}).get(kid),
                    'assessment':assessments.get(kid), 'real_selector_rank':ranks.get(kid),
                    'admitted':kid in admitted, 'saved_kernel':source.get(kid,{}).get('kernel')})
        state_path=directory/'downstream/state.json'; terminal_path=directory/'downstream/window_terminal.private.json'
        window_terminal=terminal_path.exists()
        windows.append({'window':w['id'],'terminal':window_terminal})
        if not state_path.exists():continue
        state=read(state_path);sources.append(ref(state_path))
        from arcgen.astra_usage_diversity_20260913.palette import Allocator
        runtime=SimpleNamespace(run_dir=state_path.parent,state=state,window=w['id'],palette_allocator=Allocator(ledger),budget=Budget(ledger),
                                factory_config={'_output_directory':str(directory)})
        if window_terminal:verify(runtime,read(terminal_path))
        for tid,r in sorted(state['semantic_records'].items()):
            episode=r.get('complete_task'); study=r['study']
            row={'task_id':tid,'kernel_id':r['source_candidate_id'],'window':w['id'],'account':w['account'],
                 'cohort': w.get('cohort', 'fresh_model_comparison'),
                 'natural_eligible': study.get('natural_eligible', False),
                 'merit_admitted': study.get('merit_admitted', False),
                 'generation_provenance': study.get('original_source'),
                 'arm':study['primary_arm'],'invitation':study['invitation'],'bank_setting':study['bank_setting'],
                 'generation_model':study.get('generation_model','gpt-6-astra'),
                 'comparison_block':study.get('comparison_block'), 'comparison_role':study.get('comparison_role'),
                 'pinning':study.get('pinning','ordinary'), 'pin_pair_id':study.get('pin_pair_id'), 'context':w['context'],
                 'planner':study.get('planner'),'planner_paired':study.get('planner_paired'),
                 'planner_pair_id':study.get('planner_pair_id'),'planner_version':study.get('planner_version'),
                 **({'family':study.get('family'),'rung':study['rung']} if study.get('rung') else {}),
                 'effort':r['private_idea_effort'],'admission':study['admission'],
                 'real_selector_rank':study.get('real_selector_rank'),'status':r['status'],
                 'author_target_account':((r.get('concrete_plan') or {}).get('private_execution') or {}).get(
                     'cross_pair_concrete_audit', {}).get('relevant_and_incidental_variation'),
                 'author_target_account_is_quality_evidence':False,
                 'planned':bool(r.get('concrete_plan')),'realized':bool(episode),
                 'evidence_clear':evidence_is_clear(r),
                 'verified_before_union':window_terminal and r.get('pipeline_usable_before_union_audit') is True,
                 'editorial_mode':r.get('editorial_mode'),'intrinsic_quality':r.get('intrinsic_quality'),
                 'evidence_consistency_watch':consistency_watch(r),
                 'exact_hash':content_hash(episode) if episode else None,
                 'palette_hash':palette_equivalent_hash(episode) if episode else None,
                 'task_file':str(state_path.parent/'tasks'/(tid+'.arc.json')) if episode and window_terminal else None,
                 'rule':'','state':ref(state_path)}
            from arcgen.astra_usage_diversity_20260913 import gate,adjudication
            blind=r.get('blind_results',[])
            current=r.get('difficulty_versions',{}).get(row['exact_hash'],{})
            row.update(blind_performance=gate.performance(tid,r,list(state['call_ledger'].values())),
                original_primary_result='unavailable' if not blind else 'matched' if blind[0].get('primary_matches_hidden') else 'mismatched',
                difficulty_versions=deepcopy(r.get('difficulty_versions',{})),
                review_routes=[v.get('route') for v in r.get('difficulty_versions',{}).values() if v.get('route')],
                hold_kind=r.get('answer_evidence',{}).get('hold_kind'),
                review_reason=r.get('answer_evidence',{}).get('resolution'),
                repaired=bool(r.get('repair_count')),confirmed_rival=adjudication.confirmed_rival(current.get('adjudication')),
                response_count=len(blind))
            if r.get('study',{}).get('gate'):  # gate study 2026-10-15: right-twice fields (gate.py)
                from .gate import _gate_study_fields
                row['blind_performance'].update(_gate_study_fields(tid,r,list(state['call_ledger'].values()),r.get('answer_evidence') or {}))
            from . import lines
            if lines.tagged(r.get('study')):  # weakness run 2026-10-20: line checks, size arm and form (lines.fields)
                row['blind_performance'].update(lines.fields(tid,r,list(state['call_ledger'].values())))
            if episode:
                from arcgen.lib.factory_contracts import display_value
                row['rule']=display_value(r['realization']['authoring_metadata']['intended_rule'],r['concrete_plan'])
            tasks.append(row)
    duplicates=duplicate_dispositions(tasks,read(inputs['bank_identities'])['tasks'])
    for row in tasks:row.update(duplicates[row['task_id']])
    link_kernels(kernels,tasks)
    categories=Counter(c['category'] for c in Budget(ledger).summary()['calls'])
    if categories['actual_boards']:raise IntegrityError('boardless batch spent an editorial call')
    accounting=Budget(ledger).summary(); embeddings=read(root/'embedding_requests.private.json')
    terminal_path=root/'phases'/(plan['phase']+'.terminal.private.json')
    status=('stopped_partial' if out
            else 'complete_pending_semantic_synthesis' if all(w['terminal'] for w in windows) and terminal_path.exists()
            else 'prepared_not_launched' if not (root/'launch_manifest.private.json').exists() and not accounting['logical_calls']
            else 'in_progress')
    for k in kernels:
        k['resource_deferred'] = k['queued_unstarted']
    value={'status':status,
           'windows':windows,'kernels':kernels,'tasks':tasks,
           'counts':{'input_slots':len(kernels),'generated_kernels':sum(k['generated'] for k in kernels),
                     'quality_eligible':sum(k['quality_eligible'] for k in kernels),
                     'queued_unstarted':sum(k['queued_unstarted'] for k in kernels),
                     'developments':len(tasks),'plans':sum(r['planned'] for r in tasks),'grids':sum(r['realized'] for r in tasks),
                     'evidence_clear':sum(r['evidence_clear'] for r in tasks),
                     'boardless_verified_before_union':sum(r['verified_before_union'] for r in tasks),
                     'usable_after_exact_palette_union':sum(r['usable_after_exact_palette_union'] for r in tasks),
                     'evidence_consistency_watch_cases':sum(bool(r['evidence_consistency_watch']) for r in tasks)},
           'call_accounting':{k:accounting[k] for k in ('logical_calls','physical_attempts','developments','by_phase_category','by_provider_account')},
           'ordinary_embedding_requests':len(embeddings['physical_requests']),
           'reused_call_accounting': [read(root/'inputs/saved'/w['id']/'manifest.private.json')['reused_calls']
                                     for w in plan['windows'] if w.get('kind') == 'saved_development'],
           'new_editorial_calls':0,'new_kernel_research_calls':0,'tasks_committed':0,
           'source_records_unchanged_by_analysis':True,
           'phase_terminal':ref(terminal_path) if terminal_path.exists() else None,
           'research_status':'disabled_by_design',
           'next_action':'Joint saved-artifact review: repetition and author exposure, surviving procedures, coverage and concrete guidance refresh; no automatic bank commit.'}
    # Aggregate complete recipe cells; never pool control/treatment by family.
    value['by_recipe']=[]
    cell = lambda k: (k['cohort'], k['arm'], k['invitation'] or '', k['bank_setting'], k['effort'], k['pinning'], k['generation_model'])
    for key in sorted({cell(k) for k in kernels}):
        ks=[k for k in kernels if cell(k)==key]
        ts=[t for t in tasks if cell(t)==key]
        natural = [t for t in ts if t['natural_eligible']]
        value['by_recipe'].append({'cohort':key[0], 'arm':key[1],'invitation':key[2] or None,'bank_setting':key[3],'effort':key[4],
            'pinning':key[5], 'generation_model':key[6],
            'kernels':sum(k['generated'] for k in ks),'admitted':sum(k['admitted'] for k in ks),
            'grids':sum(t['realized'] for t in ts),'evidence_clear':sum(t['evidence_clear'] for t in ts),
            'usable':sum(t['usable_after_exact_palette_union'] for t in ts),
            'blind_and_review': {'primary_attempts':sum(t['original_primary_result']!='unavailable' for t in ts),
                'primary_misses':sum(t['original_primary_result']=='mismatched' for t in ts),
                'reviewed':sum(bool(t['review_routes']) for t in ts),
                'verified_unsolved':sum(bool(t['blind_performance'].get('label')) for t in ts),
                'repaired':sum(t['repaired'] for t in ts),'confirmed_rival':sum(t['confirmed_rival'] for t in ts),
                'holds':dict(Counter(t['hold_kind'] for t in ts if t['hold_kind']))},
            'by_admission': dict(Counter(t['admission'] for t in ts)),
            'natural_only': {'attempts':len(natural), 'grids':sum(t['realized'] for t in natural),
                'usable':sum(t['usable_after_exact_palette_union'] for t in natural)}})
    value['counts'].update(new_kernel_slots=sum(not k['reused_generation'] for k in kernels),
        reused_input_kernels=sum(k['reused_generation'] for k in kernels),
        newly_generated_kernels=sum(k['generated'] and not k['reused_generation'] for k in kernels))
    value['planner_comparison']=planner_comparison(tasks)
    value['model_comparison_blocks']=[]
    for block in plan['comparison_blocks']:
        members=[k for k in kernels if k['comparison_block']==block['id']]
        value['model_comparison_blocks'].append({**block,'kernels':[{field:k[field] for field in
            ('kernel_id','generation_model','comparison_role','generated','admitted','task_id','assessment')} for k in members],
            'interpretation':'Compare A1+A2 against A1+S at equal call volume where the dedicated AA baseline exists; kernel acceptance alone is not completed-task complementarity.'})
    from .accounting import collect
    from arcgen.astra_usage_diversity_20260913.profile_report import summarize
    value['profile_comparison']=summarize(kernels,tasks)
    value['model_timing_accounting']=None if out else collect(root,terminal=value['status']=='complete_pending_semantic_synthesis')
    if value['status']=='complete_pending_semantic_synthesis':
        from .gallery import render
        value['gallery']=render(root,tasks)
        value['research_coverage']=research_index(root,plan['phase'],tasks)
        value['sources']=sources
        frozen(root/'analysis/terminal_report.private.json',value)
    elif out:  # no gallery or timing accounting: both write under the run
        value.update(ledger_snapshot=ledgers,research_coverage=research_index(dest,plan['phase'],tasks),sources=sources)
        frozen(dest/'analysis/terminal_report.private.json',value)
    else:atomic(root/'analysis/status_report.private.json',value)
    return value
