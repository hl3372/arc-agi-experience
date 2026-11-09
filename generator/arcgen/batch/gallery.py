"""The run gallery: one browsable page of every realized task.

The v1 gallery is written at the end of every run (gallery/index.html and gallery/metadata.json): every realized
task with its grids (test answers collapsed), the rule, arm, idea source, route and presentation overlay, the idea
and what the author says is new, the selector's note, repairs and the final status. To rebuild it for a finished
run elsewhere (the run folder is left untouched):

  python -m arcgen.batch.gallery runs/<run> --out <directory>
"""
import html
import json
from pathlib import Path
from .common import ROOT, HERE, read, ref, atomic
from .config import SCIENCE

V1 = SCIENCE.get('prompt_set', 'r23') == 'v1'
ARM_NAMES = {
    'T': 'Derived consequence', 'J': 'Joint inference across examples', 'ABS': 'Necessary abstraction',
    'DISC': 'Discovery-led difficulty', 'H_DISC': 'Hard conceptual discovery', 'H_REASON': 'Hard dependent reasoning',
    'H_COMPOSE': 'Sustained compositional execution', 'JUDG': 'Contextual judgment', 'D1': 'Exactly one demonstration',
    'ACC': 'Relatively accessible', 'EXEC': 'Execution-led difficulty', 'FORM': 'Logic puzzle',
    'SEM': 'Meaning learned in the task', 'PER': 'What counts as a unit or role', 'GEO': 'Spatial geometry',
    'WORLD': 'Things acting on one another', 'RESTORE': 'Restoring from regularity', 'GEN': 'Construction directed by structure',
    'EXEMPLAR': 'A worked instance in each input', 'INTERACT': 'Visual processes meeting', 'SEE': 'Seeing how a picture is organised',
    'TEMPLATE': 'Copies placed by the picture', 'OPEN': 'Open call, no direction',
    'DIRECT': 'Marks that direct', 'BOARD': 'A reference to consult',
    'RHYTHM': 'Finding the true repetition', 'SIGNAL': 'Structure under interference',
    'LEVELS': 'Two levels both needed', 'INVERSE': 'Working back to a cause', 'AIM': 'A condition of success selects',
    'ACROSS': 'One relation, different pictures', 'STAGES': 'Pairs as successive stages', 'CONTRAST': 'Comparisons that discriminate',
    'VIEWS': 'Two readings at once', 'SPREAD': 'Parts spread across many pairs'}
ROUTES = {'ordinary_current': 'Ordinary plan', 'evidence_preserving_scale': 'Presentation overlay',
          'episode_joint_inference': 'Joint-inference plan', 'logical_leap': 'Logical-leap plan'}


def render(root, tasks):
    return render_v1(root, tasks)


def _status(task, record):
    if task['usable_after_exact_palette_union']:
        return 'verified', 'Verified'
    if task['verified_before_union']:
        return 'duplicate', 'Duplicate of ' + str(task.get('exact_duplicate_of') or task.get('palette_duplicate_of'))
    outcome = (record.get('answer_evidence') or {}).get('difficulty_outcome') or {}
    return 'held', 'Held: ' + str(outcome.get('reason') or record.get('status') or task['status']).replace('_', ' ')


def v1_rows(root, tasks):
    """One row per realized task, from the run's saved states, selector queues and frozen design only."""
    queue = {}
    for path in sorted(Path(root).glob('accounts/*/*/eligible_queue.private.json')):
        queue.update({r['source_candidate_id']: r for r in read(path)['rows']})
    states = {}
    rows = []
    for task in tasks:
        if not task['realized']:
            continue
        path = task['state']['path']
        if path not in states:
            states[path] = read(path)['semantic_records']
        record = states[path][task['task_id']]
        source = queue.get(record['source_candidate_id'], {})
        kernel = source.get('kernel') or {}
        plan = record['planning_result']
        study = record['study']
        route = record.get('private_route')
        overlay = None
        if route == 'evidence_preserving_scale':
            overlay = {'kind': study.get('overlay') or 'scale', 'decision': (plan.get('scale_decision') or 'none').replace('_', ' ')}
        episode = read(task['task_file']) if task.get('task_file') else record['complete_task']
        grids = [p['input'] for p in episode['train'] + episode['test']]
        status, status_text = _status(task, record)
        rows.append({
            'task_id': task['task_id'], 'status': status, 'status_text': status_text,
            'arm': task['arm'], 'arm_name': ARM_NAMES.get(task['arm'], task['arm']),
            'idea_source': f"{task['generation_model']} {task['effort']}",
            'route': ROUTES.get(route, route), 'overlay': overlay,
            'train_pairs': len(episode['train']), 'tests': len(episode['test']),
            'max_side': max(max(len(g), len(g[0])) for g in grids),
            'repairs': int(record.get('repair_count') or 0),
            'blind': (task.get('blind_performance') or {}).get('label') or (
                'solved' if (task.get('blind_performance') or {}).get('any_all_test_success') else 'not solved'),
            'rule': task['rule'],
            'planner_rule': record['concrete_plan']['public_contract']['semantic_core']['central_rule'],
            'idea': kernel.get('central_relation'), 'what_is_new': kernel.get('what_is_new'),
            'selector_note': (source.get('selector_assessment') or {}).get('useful_contribution'),
            'idea_origin': 'replacement after the bank check' if kernel.get('loop_source') == 'replacement' else 'kept draft',
            'colour_instruction': {'vary': 'vary colours where the rule allows', 'v1': 'standard'}.get(study.get('colour_roles')),
            'idea_modifiers': ', '.join(study.get('idea_modifiers') or []) or None, 'pose': study.get('pose'),
            'task': episode})
    order = {'verified': 0, 'duplicate': 1, 'held': 2}
    rows.sort(key=lambda r: (order[r['status']], r['arm'], r['task_id']))
    return rows


def render_v1(root, tasks, dest=None):
    root = Path(root)
    dest = Path(dest) if dest else root / 'gallery'
    dest.mkdir(parents=True, exist_ok=True)
    rows = v1_rows(root, tasks)
    title = f"{root.name} · task gallery"
    counts = {s: sum(r['status'] == s for r in rows) for s in ('verified', 'duplicate', 'held')}
    page = (PAGE.replace('__TITLE__', html.escape(title))
                .replace('__SUMMARY__', html.escape(f"{len(rows)} realized tasks: {counts['verified']} verified, "
                                                    f"{counts['duplicate']} bank duplicates, {counts['held']} held. Not committed to the bank."))
                .replace('__DATA__', json.dumps(rows, separators=(',', ':')).replace('</', '<\\/')))
    (dest / 'index.html').write_text(page)
    atomic(dest / 'metadata.json', {'tasks': [{k: v for k, v in r.items() if k != 'task'} for r in rows]})
    return {'public': ref(dest / 'index.html'), 'metadata': ref(dest / 'metadata.json'), 'tasks': len(rows)}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f6f3;--panel:#fff;--fg:#1f2320;--muted:#636a62;--line:#dcdfd8;--ok:#2e6b3a;--dup:#8a5a00;--held:#9b2c2c;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#151715;--panel:#1d201d;--fg:#e5e8e3;--muted:#9aa396;--line:#30352f;--ok:#8fcf8f;--dup:#e7b85a;--held:#f08b8b;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:20px 16px 48px}
main{max-width:1100px;margin:0 auto}h1{font-size:1.4rem;margin:0 0 4px}.muted{color:var(--muted)}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0 18px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:1}
select,input{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg)}
article{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
.head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}.head h2{font-size:1rem;margin:0}
.badge{font-size:.75rem;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid currentColor}
.verified{color:var(--ok)}.duplicate{color:var(--dup)}.held{color:var(--held)}
dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:10px 0;font-size:.9rem}dt{color:var(--muted)}dd{margin:0;min-width:0}
.pairs{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}.pair{display:flex;gap:6px;align-items:center}
figure{margin:0}figcaption{font-size:.75rem;color:var(--muted)}canvas{display:block;image-rendering:pixelated;border:1px solid var(--line)}
details{font-size:.85rem}summary{cursor:pointer;color:var(--muted)}.mono{font-family:ui-monospace,monospace;font-size:.85em}
</style></head><body><main>
<h1>__TITLE__</h1><p class="muted">__SUMMARY__ Arms describe the request to the idea author; the overlay is the presentation invitation the planner received and what it decided.</p>
<div class="bar"><label>Arm <select id="arm"><option value="">all</option></select></label>
<label>Status <select id="status"><option value="">all</option><option>verified</option><option>duplicate</option><option>held</option></select></label>
<label>Search <input id="q" placeholder="rule, idea, id"></label><span id="shown" class="muted"></span></div>
<div id="list"></div></main>
<script>
const ROWS=__DATA__;
const ARC=['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25'];
function grid(g){const s=Math.max(4,Math.min(14,Math.floor(150/Math.max(g.length,g[0].length))));const c=document.createElement('canvas');
c.width=g[0].length*s;c.height=g.length*s;const x=c.getContext('2d');
g.forEach((row,i)=>row.forEach((v,j)=>{x.fillStyle=ARC[v];x.fillRect(j*s,i*s,s,s)}));
if(s>=6){x.strokeStyle='rgba(128,128,128,.35)';for(let i=0;i<=g.length;i++){x.beginPath();x.moveTo(0,i*s);x.lineTo(c.width,i*s);x.stroke()}
for(let j=0;j<=g[0].length;j++){x.beginPath();x.moveTo(j*s,0);x.lineTo(j*s,c.height);x.stroke()}}return c}
function fig(g,label){const f=document.createElement('figure');f.appendChild(grid(g));const t=document.createElement('figcaption');t.textContent=label+' '+g.length+'×'+g[0].length;f.appendChild(t);return f}
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e}
const list=document.getElementById('list');const cards=[];
for(const r of ROWS){const a=el('article');a.id=r.task_id;
const h=el('div','head');h.appendChild(el('h2',null,r.arm+' · '+r.arm_name));h.appendChild(el('span','badge '+r.status,r.status_text));h.appendChild(el('span','muted mono',r.task_id));a.appendChild(h);
const dl=el('dl');const add=(k,v)=>{if(v==null||v==='')return;dl.appendChild(el('dt',null,k));dl.appendChild(el('dd',null,v))};
add('Rule',r.rule);add('Idea',r.idea);add('What is new',r.what_is_new);add('Selector',r.selector_note);
add('Idea source',r.idea_source+' · '+r.idea_origin);add('Route',r.route+(r.overlay?' · overlay '+r.overlay.kind+': '+r.overlay.decision:''));
add('Episode',r.train_pairs+' training pairs · '+r.tests+' test'+(r.tests>1?'s':'')+' · largest side '+r.max_side);
add('Checks','blind solver: '+r.blind+' · repairs: '+r.repairs);add('Colour instruction',r.colour_instruction);add('Idea modifiers',r.idea_modifiers);if(r.pose!=null)add('Pose instruction',r.pose?'yes':'no');a.appendChild(dl);
const p=el('div','pairs');r.task.train.forEach((t,i)=>{const d=el('div','pair');d.appendChild(fig(t.input,'Train '+(i+1)+' in'));d.appendChild(el('span','muted','→'));d.appendChild(fig(t.output,'out'));p.appendChild(d)});
r.task.test.forEach((t,i)=>{const d=el('div','pair');d.appendChild(fig(t.input,'Test '+(i+1)+' in'));if(t.output){const s=el('details');s.appendChild(el('summary',null,'answer'));s.appendChild(fig(t.output,'Test '+(i+1)+' out'));d.appendChild(s)}p.appendChild(d)});
a.appendChild(p);list.appendChild(a);cards.push([r,a])}
const arms=[...new Set(ROWS.map(r=>r.arm))].sort();const sel=document.getElementById('arm');for(const x of arms){const o=el('option',null,x);sel.appendChild(o)}
function filter(){const A=sel.value,S=document.getElementById('status').value,Q=document.getElementById('q').value.toLowerCase();let n=0;
for(const [r,a] of cards){const ok=(!A||r.arm===A)&&(!S||r.status===S)&&(!Q||[r.rule,r.idea,r.what_is_new,r.task_id].join(' ').toLowerCase().includes(Q));a.hidden=!ok;n+=ok}
document.getElementById('shown').textContent=n+' shown'}
for(const id of ['arm','status','q'])document.getElementById(id).addEventListener('input',filter);filter();
</script></body></html>"""


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Rebuild the v1 task gallery for a finished run into another directory.')
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = read(a.root.resolve() / 'analysis/terminal_report.private.json')
    print(json.dumps(render_v1(a.root.resolve(), report['tasks'], dest=a.out), indent=1))
