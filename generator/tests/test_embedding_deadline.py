"""The embedding client gives up on a response that trickles keep-alive bytes forever (the OpenRouter hang that stalled
a batch preparation and a study), and the new arms are fully registered."""
import http.server, socket, threading, time
import pytest

_REAL_CREATE, _REAL_CONNECT = socket.create_connection, socket.socket.connect  # captured before the offline guard


def _loopback_only(monkeypatch):
    """Re-allow connections to 127.0.0.1 only; everything else stays blocked by the offline guard."""
    def create(address, *args, **kwargs):
        if address[0] not in ('127.0.0.1', 'localhost'):
            pytest.fail('offline check attempted network/provider dispatch')
        return _REAL_CREATE(address, *args, **kwargs)
    def connect(self, address):
        if address[0] != '127.0.0.1':
            pytest.fail('offline check attempted network/provider dispatch')
        return _REAL_CONNECT(self, address)
    monkeypatch.setattr(socket, 'create_connection', create)
    monkeypatch.setattr(socket.socket, 'connect', connect)
from arcgen.lib.embeddings import OpenRouterEmbeddingProvider, EmbeddingError


class Trickle(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
        try:
            for _ in range(40):  # never a complete body; a byte every 0.25 s keeps the socket from going idle
                self.wfile.write(b' '); self.wfile.flush(); time.sleep(0.25)
        except OSError:
            pass

    def log_message(self, *args):
        pass


def test_trickling_response_hits_the_overall_deadline(monkeypatch):
    _loopback_only(monkeypatch)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    provider = OpenRouterEmbeddingProvider({'provider': 'openrouter', 'model': 'openai/text-embedding-3-large',
                                            'api_key_env': 'OPENROUTER_API_KEY', 'base_url': f'http://127.0.0.1:{server.server_port}',
                                            'batch_size': 4, 'timeout_seconds': 2, 'retries': 0, 'app_name': 'test'})
    start = time.monotonic()
    with pytest.raises(EmbeddingError, match='overall deadline'):
        provider.embed(['a text'])
    assert time.monotonic() - start < 6
    server.shutdown()


def test_new_arms_are_registered():
    from arcgen.batch import idea_v1, contracts, gallery
    for arm in ('DIRECT', 'BOARD', 'RHYTHM', 'SIGNAL'):
        assert arm in idea_v1.ARMS and contracts.known_primary(arm) and arm in gallery.ARM_NAMES
        assert (idea_v1.PROMPTS / 'cards' / f'{arm}.md').exists()
    assert 'BUSY' in idea_v1.MODIFIERS and (idea_v1.PROMPTS / 'modifiers/BUSY.md').exists()
