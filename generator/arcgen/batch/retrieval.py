"""Full-corpus cached search with per-candidate quotas and shared reference text."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import threading
import numpy as np
from arcgen.lib.factory_context import FactoryContextGuard
from .common import SETTINGS, IntegrityError, digest, read, isolated_root, require_launch, verify_refs
from .routing import SOL


class Retrieval:
    """The run's embedding provider: a per-run cache (<run>/retrieval_cache) seeded once from the shared workspace cache
    (inputs['historical_cache']), with a ledger of physical embedding requests (outside the text-call budget)."""
    def __init__(self, root, inputs):
        import shutil
        from arcgen.lib.embeddings import build_embedding_provider
        from arcgen.paths import WORKSPACE
        from .common import frozen, atomic, ref
        self.root = Path(root); self.lock = threading.RLock(); self._atomic = atomic
        context = read(inputs['selector_context'])
        self.provider = build_embedding_provider(context['embedding'], self.root / 'retrieval_cache')
        old = Path(inputs['historical_cache']); old = old if old.is_absolute() else WORKSPACE / old
        source = build_embedding_provider(context['embedding'], old)
        copied, missing = [], []; reuse_path = self.root / 'retrieval_cache_reuse.private.json'
        reuse = read(reuse_path) if reuse_path.exists() else None
        if reuse: verify_refs(reuse)
        for text in (() if reuse is not None else dict.fromkeys(r['central_relation'] for r in context['references'])):
            key = source._key(text); p = source._path(key); dest = self.provider._path(key)
            if p.exists():
                source._read(p, key=key, text=text)
                if not dest.exists(): dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
                self.provider._read(dest, key=key, text=text); copied.append({'key': key, 'source': ref(p), 'copy': ref(dest)})
            else:
                missing.append(key)
        if reuse is None: frozen(reuse_path, {'source_cache': str(old), 'copied': copied, 'missing': missing})
        self.path = self.root / 'embedding_requests.private.json'
        self.ledger = read(self.path) if self.path.exists() else {'physical_requests': [], 'cache_hits': 0, 'cache_misses': 0}
        if hasattr(self.provider.provider, '_request'):
            self.provider.provider._request_observer = self.observed

    def observed(self, attempt, text_count):
        import time
        self.ledger['physical_requests'].append({'at_epoch': time.time(), 'native_attempt': attempt, 'text_count': text_count,
            'model': getattr(self.provider.provider, 'model', None), 'scope': 'ordinary embedding request; excluded from text-call budget'})
        self._atomic(self.path, self.ledger)

    def embed(self, texts):
        with self.lock:
            hit = sum(self.provider._path(self.provider._key(t)).exists() for t in texts)
            self.ledger['cache_hits'] += hit; self.ledger['cache_misses'] += len(texts) - hit
            self._atomic(self.path, self.ledger); return self.provider.embed(texts)


def make_run_index(root, inputs, budget):
    """Native isolated cache/reuse ledger; a cache miss requires a new launch.

    The reference matrix is built once per run, then reused across windows.
    Ordinary embedding attempts remain in their own ledger, outside text calls.
    """
    root = isolated_root(root)
    if root != budget.root:
        raise IntegrityError('retrieval and text budget belong to different run roots')
    verify_refs(inputs)
    context = read(inputs['selector_context'])
    reference_hash = digest(sorted(context['references'], key=lambda r: r['reference_id']))
    native = Retrieval(root, inputs)

    class BoundRetrieval:
        def embed(self, texts):
            values = list(texts)
            if (root / 'STOP_DISPATCH.json').exists():
                raise IntegrityError('revision dispatch paused')
            if any(not native.provider._path(native.provider._key(t)).exists() for t in values):
                require_launch(root, budget, reference_snapshot_hash=reference_hash)
            return native.embed(values)

    return ReferenceIndex(context['references'], BoundRetrieval(), provider_identity={
        'provider_config_hash': native.provider._config_hash,
        'embedding_config_hash': digest(context['embedding']),
        'cache_root': str(root / 'retrieval_cache'),
        'request_ledger': str(root / 'embedding_requests.private.json')})


class CachedVectors:
    """Read-only qualification provider. A cache miss never falls through to HTTP."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.vectors = {}
        self.config_hash = None
        for path in sorted(self.directory.glob('*/*.json')):
            row = json.loads(path.read_text())
            if row['key'] != path.stem or row['provider_config_hash'] != self.directory.name:
                raise IntegrityError('embedding cache identity mismatch')
            if self.config_hash not in (None, row['provider_config_hash']):
                raise IntegrityError('mixed embedding configurations')
            self.config_hash = row['provider_config_hash']
            vector = np.asarray(row['vector'], dtype=np.float64)
            if row['version'] != 1 or vector.ndim != 1 or len(vector) != row['dimension'] or not np.isfinite(vector).all():
                raise IntegrityError('invalid cached vector')
            h = row['text_sha256']
            if h in self.vectors and not np.array_equal(self.vectors[h], vector):
                raise IntegrityError('conflicting vectors for the same text')
            self.vectors[h] = vector
        if not self.vectors:
            raise IntegrityError('empty embedding cache')
        self.identity = {'provider_config_hash': self.config_hash, 'mode': 'read_only_existing_cache'}
        self.cache_hits = 0

    def embed(self, texts):
        result = []
        for text in texts:
            key = hashlib.sha256(text.encode()).hexdigest()
            if key not in self.vectors:
                raise IntegrityError('missing cached vector; no network fallback: ' + key)
            result.append(self.vectors[key])
        self.cache_hits += len(texts)
        return result


def unit(vectors):
    value = np.asarray(vectors, dtype=np.float64)
    if value.ndim != 2 or not value.shape[1] or not np.isfinite(value).all():
        raise IntegrityError('invalid embedding matrix')
    lengths = np.linalg.norm(value, axis=1)
    if (lengths <= 0).any():
        raise IntegrityError('zero embedding vector')
    return value / lengths[:, None]


def public_reference(row):
    text = row['central_relation']
    if row['summary'] not in text:
        text += '\n' + row['summary']
    return {'reference_id': row['reference_id'], 'source': row['source'], 'procedural_detail': text}


class ReferenceIndex:
    """One immutable matrix per corpus. Embedding provider owns cache/accounting."""
    def __init__(self, references, provider, *, provider_identity):
        self.records = {}
        for row in references:
            rid = row['reference_id']
            if not row.get('central_relation') or not row.get('summary'):
                raise IntegrityError('reference lacks procedural text')
            if rid in self.records and self.records[rid] != row:
                raise IntegrityError('conflicting reference ID')
            if row['source'] not in ('accepted_generated_task', 'real_task'):
                raise IntegrityError('unrecognized reference source')
            self.records[rid] = deepcopy(row)
        self.ids = sorted(self.records)
        self.provider = provider
        self.provider_identity = deepcopy(provider_identity)
        self.snapshot_hash = digest([self.records[k] for k in self.ids])
        # An empty bank is a valid corpus: every search then returns no neighbours (a first batch from scratch).
        self.vectors = unit(provider.embed([self.records[k]['central_relation'] for k in self.ids])) if self.ids else None
        self._position = {k: i for i, k in enumerate(self.ids)}
        self.lock = threading.RLock()

    def rankings(self, queries):
        # Freeze a full ordering locally; changing k never requires another embedding.
        if self.vectors is None:
            return [[] for _ in queries]
        with self.lock:
            q = unit(self.provider.embed(queries))
        if q.shape[1] != self.vectors.shape[1]:
            raise IntegrityError('candidate and bank embedding dimensions differ')
        scores = q @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind='stable')  # IDs already sorted for ties.
        return [[(self.ids[int(i)], float(scores[j, i])) for i in row] for j, row in enumerate(order)]

    def pack(self, candidates, *, accepted_k=None, real_k=None, rankings=None):
        """rankings: this index's rankings() of the candidates' relations, when the caller packs one window at several k."""
        settings = SETTINGS['selector']
        accepted_k = settings['accepted_neighbors_per_candidate'] if accepted_k is None else accepted_k
        real_k = settings['real_neighbors_per_candidate'] if real_k is None else real_k
        if any(type(k) is not int or k < 0 for k in (accepted_k, real_k)) or accepted_k == 0:
            raise IntegrityError('invalid reference quota')
        ids = list(candidates)
        if rankings is None:
            rankings = self.rankings([candidates[k]['central_relation'] for k in ids]) if ids else []
        elif len(rankings) != len(ids):
            raise IntegrityError('rankings do not match the candidates')
        mapping, shared, diagnostics = {}, {}, {}
        for cid, hits in zip(ids, rankings):
            mapping[cid] = {'accepted_reference_ids': [], 'real_reference_ids': []}
            diagnostics[cid] = []
            for kind, cap, field in [('accepted_generated_task', accepted_k, 'accepted_reference_ids'),
                                     ('real_task', real_k, 'real_reference_ids')]:
                selected = [(rid, score) for rid, score in hits if self.records[rid]['source'] == kind][:cap]
                for rank, (rid, score) in enumerate(selected, 1):
                    row = self.records[rid]
                    mapping[cid][field].append(rid)
                    # Full existing text, no arbitrary prefix cut and no source arm/grade.
                    shared[rid] = public_reference(row)
                    diagnostics[cid].append({'reference_id': rid, 'source_rank': rank, 'similarity': score})
        return {'references': [shared[k] for k in sorted(shared)], 'candidate_references': mapping,
                'private_retrieval': {'snapshot_hash': self.snapshot_hash, 'provider': self.provider_identity,
                    'query_hashes': {k: digest(candidates[k]['central_relation']) for k in ids},
                    'rankings': diagnostics, 'accepted_k': accepted_k, 'real_k': real_k}}


def _measure(prompt, references, schema=None):
    settings = SETTINGS['selector']
    guard = FactoryContextGuard(context_limit=settings['context_limit'],
                                response_headroom=settings['response_headroom'], model=SOL)
    # Same conservative schema/system allowance as the unchanged native Runtime.
    full = prompt + '\n' + json.dumps(schema or {}, ensure_ascii=False, separators=(',', ':')) + '\n' + ' system' * 2048
    report = guard.inspect(full, 'full_window_selection')
    reference_report = guard.inspect(json.dumps(references, ensure_ascii=False, separators=(',', ':')), 'selector_references')
    over = (report['estimated_prompt_tokens'] > settings['prompt_token_budget'] or report['requires_context_decision']
            or reference_report['estimated_prompt_tokens'] > settings['reference_token_budget'])
    return report, reference_report, over


def over_budget(prompt, references, schema=None):
    return _measure(prompt, references, schema)[2]


def check_size(prompt, references, schema=None):
    settings = SETTINGS['selector']
    report, reference_report, over = _measure(prompt, references, schema)
    if over and not report['requires_context_decision']:
        raise IntegrityError('bounded selector context exceeds its frozen budget; no truncation')
    if report['requires_context_decision']:
        raise IntegrityError('native selector context guard blocked the complete request')
    return {'complete_request': report, 'references': reference_report,
            'prompt_budget': settings['prompt_token_budget'], 'reference_budget': settings['reference_token_budget']}
