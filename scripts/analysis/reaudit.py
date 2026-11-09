"""Re-run the build and audit of a finished round's tasks with the current battery (no model calls)."""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.construct import pipeline

src = Path(sys.argv[1])
out = Path(sys.argv[2]); (out / 'tasks').mkdir(parents=True, exist_ok=True)
only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
pipeline.PROCESSES = min(18, os.cpu_count() or 4)
for d in sorted((src / 'tasks').glob('*/dossier.json')):
    t = json.loads(d.read_text())
    if only and t['id'] not in only or not t.get('witness_program'):
        continue
    t0 = time.time()
    try:
        audit, w = pipeline.build(t, out)
    except Exception as e:
        import traceback
        audit, w = {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1500:]}, None
    (out / 'tasks' / t['id']).mkdir(exist_ok=True, parents=True)
    (out / 'tasks' / t['id'] / 'build.json').write_text(json.dumps({'audit': audit, 'hint': t['hint']}, indent=1, default=str))
    old = (t.get('audit') or {})
    def brief(a):
        return {k: v for k, v in a.items() if k.startswith('battery_')}
    print(f"== {t['id']} ({t['demand']}) ref {audit.get('witness', {}).get('objective')} in {time.time()-t0:.0f}s", flush=True)
    print('   old audit', json.dumps(old.get('audit')), flush=True)
    print('   new audit', json.dumps(audit.get('audit')), audit.get('error', ''), flush=True)
    for b in ('battery_20000', 'battery_200000'):
        print(f'   {b}: old', json.dumps({m: (r.get("valid"), r.get("objective")) for m, r in (old.get(b) or {}).items()}), flush=True)
        print(f'   {b}: new', json.dumps({m: (r.get("valid"), r.get("objective"), r.get("time_capped")) for m, r in (audit.get(b) or {}).items()}), flush=True)
    print('   probes', json.dumps(audit.get('probes', {}).get('adequacy')), 'capped', audit.get('time_capped'), flush=True)
