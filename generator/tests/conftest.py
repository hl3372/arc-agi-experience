"""Offline test suite. Runs under the v1 reference profile with a throw-away workspace; any network connection, Codex
transport or tokenizer download fails the test."""
import atexit
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(REPO))
# Before any arcgen import: the profile, no run root, and a workspace that is never the checkout's own.
os.environ.setdefault('ARCGEN_CONFIG', str(REPO / 'arcgen/batch/profiles/v1.json'))
os.environ['ARCGEN_RUN_ROOT'] = ''
os.environ['ARCGEN_WORKSPACE'] = tempfile.mkdtemp(prefix='arcgen-test-workspace-')  # never a real workspace
atexit.register(shutil.rmtree, os.environ['ARCGEN_WORKSPACE'], True)
os.environ.pop('ARCGEN_BANK', None)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from arcgen.lib.factory_runtime import FactoryModelInvoker
    from arcgen.lib import factory_context
    native_tokenizer = factory_context._tokenizer

    def forbidden(*args, **kwargs):
        pytest.fail('offline check attempted network/provider dispatch')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(FactoryModelInvoker, '_codex_response', forbidden)
    # No tokenizer download: the context guard uses its native estimate.
    monkeypatch.setattr(factory_context, '_tokenizer', lambda model: (None, {'tokenizer_available': False}))
    yield native_tokenizer
