"""Test isolation, set before anything imports research_envs:
- a fresh runs root, so tests never write into (or wait on the compute slots of) real rounds;
- no model can be called: the Codex executable points nowhere and OpenRouter keys are removed, so a test that
  reached a real model call would fail instead of spending anything. Dry-run tests use RE_LLM_STUB."""
import os
import tempfile

os.environ['RE_ROOT'] = tempfile.mkdtemp(prefix='re-tests-')
os.environ['RE_CODEX_BIN'] = os.path.join(os.environ['RE_ROOT'], 'codex-disabled-in-tests')
for k in ('OPENROUTER_API_KEY', 'OR_KEY_FILE', 'RE_LLM_STUB'):
    os.environ.pop(k, None)
# run-mode flags are read at import time; tests set the ones they need themselves
for k in [k for k in os.environ if k.startswith(('RE_A_', 'RE_B_')) or k in ('RE_BUDGET', 'RE_IMPROVE', 'RE_ROUND_REGISTRY')]:
    os.environ.pop(k)
