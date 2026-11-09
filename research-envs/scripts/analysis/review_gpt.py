"""The GPT half of the model review panel: one blind review per packet, with the shared rubric (docs/REVIEW_RUBRIC.md),
through the Codex CLI (model calls).

usage: review_gpt.py OUT [--workers N]
  Reads runs/OUT/index.json and runs/OUT/packets/ (from review_packets.py); writes runs/OUT/verdicts/gpt/<id>.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the checkout, if the package is not installed
from research_envs.common import ROOT
from research_envs.construct import llm

args = sys.argv[1:]
workers = 8
if '--workers' in args:
    i = args.index('--workers'); workers = int(args[i + 1]); del args[i:i + 2]
if not args:
    raise SystemExit(__doc__)
R = ROOT / 'runs' / args[0]
SCHEMA = {'type': 'object', 'properties': {
    'id': {'type': 'string'}, 'coherence': {'type': 'integer'}, 'fairness': {'type': 'string', 'enum': ['fair', 'unfair', 'unsure']},
    'fairness_reason': {'type': 'string'}, 'interest': {'type': 'integer'}, 'known_as': {'type': 'string'},
    'description_ok': {'type': 'boolean'}, 'solver_failure': {'type': 'string', 'enum': ['task', 'solver', 'n/a']},
    'verdict': {'type': 'string', 'enum': ['accept', 'revise', 'reject']}, 'main_issue': {'type': 'string'}}}
rubric = (Path(__file__).resolve().parents[2] / 'docs' / 'REVIEW_RUBRIC.md').read_text()
(R / 'verdicts' / 'gpt').mkdir(parents=True, exist_ok=True)


def one(entry):
    out = R / 'verdicts' / 'gpt' / f"{entry['id']}.json"
    if out.exists():
        return
    packet = (R / 'packets' / f"{entry['id']}.md").read_text()
    r = llm.call(rubric + '\n\n---\n\n' + packet + '\n\nReview this task with the rubric above.', SCHEMA,
                 R / 'calls' / 'gpt', entry['id'], effort='high', timeout=1800)
    out.write_text(json.dumps(r, indent=1))
    print(entry['id'], r.get('verdict'), r.get('fairness'), flush=True)


with ThreadPoolExecutor(workers) as ex:
    list(ex.map(one, json.loads((R / 'index.json').read_text())))
