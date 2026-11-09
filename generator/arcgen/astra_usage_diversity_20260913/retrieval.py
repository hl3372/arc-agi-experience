"""Full-corpus cached search with per-candidate quotas and shared reference text."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import threading
import numpy as np
from arcgen.lib.factory_context import FactoryContextGuard
from .common import SETTINGS, IntegrityError, digest, read, isolated_root, require_launch, verify_refs


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
        if not self.ids:
            raise IntegrityError('empty reference corpus')
        self.provider = provider
        self.provider_identity = deepcopy(provider_identity)
        self.snapshot_hash = digest([self.records[k] for k in self.ids])
        self.vectors = unit(provider.embed([self.records[k]['central_relation'] for k in self.ids]))
        self._position = {k: i for i, k in enumerate(self.ids)}
        self.lock = threading.RLock()

    def rankings(self, queries):
        # Freeze a full ordering locally; changing k never requires another embedding.
        with self.lock:
            q = unit(self.provider.embed(queries))
        if q.shape[1] != self.vectors.shape[1]:
            raise IntegrityError('candidate and bank embedding dimensions differ')
        scores = q @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind='stable')  # IDs already sorted for ties.
        return [[(self.ids[int(i)], float(scores[j, i])) for i in row] for j, row in enumerate(order)]

    def pack(self, candidates, *, accepted_k=None, real_k=None):
        settings = SETTINGS['selector']
        accepted_k = settings['accepted_neighbors_per_candidate'] if accepted_k is None else accepted_k
        real_k = settings['real_neighbors_per_candidate'] if real_k is None else real_k
        if any(type(k) is not int or k < 0 for k in (accepted_k, real_k)) or accepted_k == 0:
            raise IntegrityError('invalid reference quota')
        ids = list(candidates)
        rankings = self.rankings([candidates[k]['central_relation'] for k in ids]) if ids else []
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


def check_size(prompt, references, schema=None):
    settings = SETTINGS['selector']
    guard = FactoryContextGuard(context_limit=settings['context_limit'],
                                response_headroom=settings['response_headroom'], model='gpt-5.6-sol')
    # Same conservative schema/system allowance as the unchanged native Runtime.
    full = prompt + '\n' + json.dumps(schema or {}, ensure_ascii=False, separators=(',', ':')) + '\n' + ' system' * 2048
    report = guard.inspect(full, 'full_window_selection')
    reference_report = guard.inspect(json.dumps(references, ensure_ascii=False, separators=(',', ':')), 'selector_references')
    if report['estimated_prompt_tokens'] > settings['prompt_token_budget'] or reference_report['estimated_prompt_tokens'] > settings['reference_token_budget']:
        raise IntegrityError('bounded selector context exceeds its frozen budget; no truncation')
    if report['requires_context_decision']:
        raise IntegrityError('native selector context guard blocked the complete request')
    return {'complete_request': report, 'references': reference_report,
            'prompt_budget': settings['prompt_token_budget'], 'reference_budget': settings['reference_token_budget']}
