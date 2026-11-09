"""Model calls through OpenRouter, with the same signature and records as construct/llm.py, for validation solvers
(the open-weight solver comparisons in domain A). Nothing else uses OpenRouter.

The key is read from the environment variable OPENROUTER_API_KEY, or else from the file named by the optional
OR_KEY_FILE variable. No other location is searched. The key is never logged, printed or written into records.
Small models are asked for native JSON fields (an array, an object) instead of the JSON-in-a-
string fields our Codex schemas use; the reply is parsed tolerantly and converted back to the schema's field names.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = 'https://openrouter.ai/api/v1/chat/completions'


class ORError(RuntimeError):
    pass


def _key():
    k = os.environ.get('OPENROUTER_API_KEY')
    if not k and os.environ.get('OR_KEY_FILE'):
        f = Path(os.environ['OR_KEY_FILE']).expanduser()
        if f.is_file():
            k = f.read_text().strip()
    if not k:
        raise ORError('no OpenRouter key: set OPENROUTER_API_KEY, or OR_KEY_FILE to a file holding the key')
    return k


def _fields(schema):
    """[(reply key, schema key, wants native JSON)] from a flat object schema."""
    out = []
    for k, v in schema.get('properties', {}).items():
        if k.endswith('_json'):
            out.append((k[:-5], k, True))
        else:
            out.append((k, k, False))
    return out


def _instructions(schema):
    lines = []
    for reply, key, native in _fields(schema):
        t = schema['properties'][key].get('type')
        lines.append(f'- "{reply}": ' + ('a JSON value (an array or object, not a string)' if native else f'a {t}'))
    return ('Reply with only one JSON object, with exactly these keys:\n' + '\n'.join(lines) +
            '\nNo text before or after the JSON object.')


def _extract(text):
    """The first JSON object in the reply (inside a code fence or bare)."""
    if not text:
        return None
    for fence in ('```json', '```'):
        if fence in text:
            seg = text.split(fence, 1)[1].split('```', 1)[0]
            try:
                return json.loads(seg)
            except json.JSONDecodeError:
                pass
    start = text.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    return None


def make_call(model, max_tokens=48000, temperature=0.2, reasoning_tokens=24000):
    def call(prompt, schema, record_dir, name, effort='high', timeout=600, retries=4):
        rec = Path(record_dir) / name if record_dir else None
        if rec:
            rec.mkdir(parents=True, exist_ok=True)
            if (rec / 'response.json').exists():
                return json.loads((rec / 'response.json').read_text())
            (rec / 'prompt.md').write_text(prompt)
        # reasoning models can spend the whole allowance thinking and return no answer: cap the reasoning, leave room
        body = {'model': model, 'temperature': temperature, 'max_tokens': max_tokens,
                'reasoning': {'max_tokens': reasoning_tokens},
                'messages': [{'role': 'user', 'content': prompt + '\n\n' + _instructions(schema)}]}
        last = None
        for attempt in range(1, retries + 2):
            t0 = time.time()
            try:
                req = urllib.request.Request(URL, data=json.dumps(body).encode(), method='POST',
                                             headers={'Authorization': f'Bearer {_key()}', 'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.load(r)
                text = (data.get('choices') or [{}])[0].get('message', {}).get('content') or ''
                usage = data.get('usage', {})
                parsed = _extract(text)
                meta = {'attempt': attempt, 'model': model, 'seconds': round(time.time() - t0, 1), 'usage': usage,
                        'parsed': parsed is not None, 'text_tail': text[-1500:]}
                if rec:
                    (rec / f'attempt-{attempt}.json').write_text(json.dumps(meta, indent=1))
                if parsed is None:
                    last = 'no JSON object in the reply'
                    continue
                out = {}
                for reply, key, native in _fields(schema):
                    v = parsed.get(reply, parsed.get(key))
                    out[key] = (json.dumps(v) if not isinstance(v, str) else v) if native else v
                if rec:
                    (rec / 'response.json').write_text(json.dumps(out, indent=1))
                return out
            except urllib.error.HTTPError as e:
                last = f'HTTP {e.code}'
                if rec:
                    (rec / f'attempt-{attempt}.json').write_text(json.dumps({'attempt': attempt, 'error': last}))
                time.sleep(min(60, 4 * 2 ** attempt) + random.random() * 3)
            except Exception as e:
                last = f'{type(e).__name__}: {e}'
                if rec:
                    (rec / f'attempt-{attempt}.json').write_text(json.dumps({'attempt': attempt, 'error': last}))
                time.sleep(5 * attempt)
        raise ORError(f'{model} {name}: {last}')
    return call
