"""Model calls through the Codex CLI, with structured JSON output and a full record of each call.

Every call writes prompt, schema, raw response and timing under the run directory, so any verdict can be traced.

One Codex account: the CLI runs with CODEX_HOME set to $CODEX_HOME if it is set, else ~/.codex (log in once with
`codex login`). Concurrent calls are limited machine-wide by RE_CODEX_CONCURRENCY (default 8), across every task
process of every round that shares the same runs root.

The model runs in Codex's read-only sandbox (`--sandbox read-only`) in an empty temporary directory, and its prompt
holds only the public task. Read-only mode does not hide the disk: if Codex's own sandbox can run shell commands on
your machine, the model could read any file your user can read, including runs/ (see README, "Sandboxes").

Settings (environment):
- RE_CODEX_MODEL: the model (default gpt-6.1-sol, the model used for every published run);
- RE_CODEX_CONCURRENCY: concurrent Codex calls (default 8);
- RE_CODEX_BIN: the Codex executable (default `codex` on PATH);
- RE_LLM_STUB: a Python file whose call() is used instead of Codex, for dry runs (no model is called).
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from ..common import FileSlots

MODEL = os.environ.get('RE_CODEX_MODEL', 'gpt-6.1-sol')
CONCURRENCY = int(os.environ.get('RE_CODEX_CONCURRENCY', 8))
CODEX = os.environ.get('RE_CODEX_BIN', 'codex')


def codex_home():
    """The one Codex account: $CODEX_HOME, else ~/.codex."""
    return Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')


class LLMError(RuntimeError):
    pass


def _strict(schema):
    """Structured-output schemas: every object closed and every property required."""
    if isinstance(schema, dict):
        if schema.get('type') == 'object':
            schema.setdefault('additionalProperties', False)
            schema['required'] = list(schema.get('properties', {}))
        for v in schema.values():
            _strict(v)
    elif isinstance(schema, list):
        for v in schema:
            _strict(v)
    return schema


def call(prompt, schema, record_dir, name, effort='high', timeout=1800, retries=2, model=None):
    """Run one structured call; returns the parsed JSON. Records everything under record_dir/name/.
    For dry runs, RE_LLM_STUB names a Python file whose call() is used instead (no model is called)."""
    if os.environ.get('RE_LLM_STUB'):
        import importlib.util
        sp_ = importlib.util.spec_from_file_location('re_llm_stub', os.environ['RE_LLM_STUB'])
        mod = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(mod)
        return mod.call(prompt, schema, record_dir, name, effort=effort)
    model = model or MODEL
    rec = Path(record_dir) / name
    rec.mkdir(parents=True, exist_ok=True)
    if (rec / 'response.json').exists():  # resume: a finished call is never repeated, and its record is kept as made
        if (rec / 'prompt.md').exists() and (rec / 'prompt.md').read_text() != prompt:
            (rec / 'prompt_on_resume.md').write_text(prompt)
        return json.loads((rec / 'response.json').read_text())
    (rec / 'prompt.md').write_text(prompt)
    schema = _strict(json.loads(json.dumps(schema)))
    (rec / 'schema.json').write_text(json.dumps(schema, indent=1))
    env = {**os.environ, 'CODEX_HOME': str(codex_home())}
    last = None
    for attempt in range(1, retries + 2):
        with tempfile.TemporaryDirectory(prefix='re-llm-') as tmp:
            out = Path(tmp) / 'response.json'
            sp = Path(tmp) / 'schema.json'
            sp.write_text(json.dumps(schema))
            cmd = [CODEX, 'exec', '--model', model, '--config', f'model_reasoning_effort="{effort}"',
                   '--sandbox', 'read-only', '--ephemeral', '--ignore-user-config', '--skip-git-repo-check',
                   '--cd', tmp, '--output-schema', str(sp), '--output-last-message', str(out), '--color', 'never', '-']
            with FileSlots('codex', CONCURRENCY):   # a fresh instance per call: safe across threads and processes
                t0 = time.time()
                try:
                    p = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout, env=env)
                    ok = p.returncode == 0 and out.is_file()
                    raw = out.read_text() if out.is_file() else ''
                except subprocess.TimeoutExpired:
                    ok, raw, p = False, '', None
                except FileNotFoundError:
                    raise LLMError(f'{name}: the Codex CLI ({CODEX!r}) is not installed or not on PATH')
            meta = {'attempt': attempt, 'model': model, 'effort': effort, 'seconds': round(time.time() - t0, 1),
                    'returncode': p.returncode if p else None, 'stderr_tail': (p.stderr[-1500:] if p else 'timeout')}
            (rec / f'attempt-{attempt}.json').write_text(json.dumps(meta, indent=1))
            if ok:
                try:
                    value = json.loads(raw)
                    (rec / 'response.json').write_text(json.dumps(value, indent=1))
                    return value
                except json.JSONDecodeError as e:
                    last = f'bad JSON: {e}'
            else:
                last = meta['stderr_tail']
        time.sleep(5 * attempt)
    raise LLMError(f'{name}: {last}')
