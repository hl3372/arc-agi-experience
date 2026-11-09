"""Re-run the blind-restatement comparison, adjudication and classification of a round's finished tasks with the
current code, so every task is judged by the same rules. Model calls are reused from the records; the only new calls
are retries for blind specs that cannot be evaluated. Keeps the class at run time as class_at_run."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.construct import pipeline

rd = pipeline.ROOT / 'runs' / sys.argv[1]
changed = 0
build_errors = []
for f in sorted((rd / 'tasks').glob('*/dossier.json')):
    t = json.loads(f.read_text())
    if t.get('final') and (t.get('audit') or {}).get('error'):
        build_errors.append(t['id'])
    if not t.get('final') or not t.get('witness'):
        continue
    t.setdefault('class_at_run', t.get('class'))
    t.setdefault('reasons_at_run', t.get('reasons'))
    pr = t['audit'].get('probes') or {}
    if pr.get('adequacy') is not None and pr['adequacy'] < 0.9 and 'targeted' not in pr:
        from research_envs.construct import probes as P
        with pipeline._compute:
            ps = P.probe_set(t['spec'], [t['witness']], t['hint'])
            ta = P.targeted_adequacy(t['spec'], ps, t['hint'], t['sibling'])
        pr['plain_adequacy'] = pr['adequacy']
        pr['targeted'] = ta
        pr['adequacy'] = round(ta['score'], 3)
        t['audit']['probes'] = pr
    rs, blind = pipeline.blind_restatement(t, f.parent / 'calls')
    t['restate'], t['blind_spec'] = rs, blind
    t['flags'] = []
    t['outcomes'] = pipeline.adjudicate(t, t['audit'], t.get('pilots') or {}, blind)
    t['class'], t['reasons'] = pipeline.classify(t, t['audit'], rs, t['outcomes'])
    t['audit_mode'] = pipeline.audit_mode(t, t['audit'])
    t['rechecked_with'] = pipeline.code_hash()
    pipeline._write(f, t)
    moved = t['class'] != t['class_at_run']
    changed += moved
    print(f"{t['id']:14s} {t['class_at_run']:12s} -> {t['class']:12s} {t['reasons']} flags {t['flags']} "
          f"| adequacy {t['audit'].get('probes', {}).get('plain_adequacy', t['audit'].get('probes', {}).get('adequacy'))}"
          f"->{t['audit'].get('probes', {}).get('adequacy')} | sib {rs.get('disagree_sibling')} probes {rs.get('disagree_probes')}"
          f"{' UNUSABLE ' + rs['unusable'][:120] if rs.get('unusable') else ''}", flush=True)
print(f'{changed} classes changed')
print(f'build errors (repair pass: un-final them and rerun the round): {build_errors}')
