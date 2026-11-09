"""In-batch references (2026-10-09): later idea windows see the ideas already selected in the same batch.

Idea calls and the selector compare each idea with the reference corpus frozen when the run was prepared, so ideas
of one batch never see each other: v1-b12 repeated templates across calls (12 of 23 AIM tasks moved or turned one
element) while scoring as fresh against the bank. With `science.batch_references`, each idea window carries a
`reference_wave` (A, B, ...). A window of a later wave starts only when every window of the earlier waves has frozen
its eligible queue; its idea loop and selector then search the corpus plus every kernel those windows selected,
labelled `batch:<candidate id>`. The added references are frozen per wave, so a resumed run sees the same ones.
Development never waits for a wave: only idea windows do.
"""
from .common import IntegrityError, read, ref, frozen, verify_refs


def wave(window):
    return window.get('reference_wave')


SOURCE_KINDS = ('fresh_supply', 'saved_ideas', 'saved_queue')  # windows whose eligible queue later waves see


def earlier(windows, window):
    """Idea windows (and saved windows given a reference wave) of earlier reference waves, in frozen window order."""
    mine = wave(window)
    if mine is None:
        return []
    return [w for w in windows if w.get('kind') in SOURCE_KINDS and wave(w) is not None and wave(w) < mine]


def queue_path(root, window):
    return root / 'accounts' / window['account'] / window['id'] / 'eligible_queue.private.json'


def blocked(root, windows, window):
    """True while an earlier wave still has an idea window without a frozen eligible queue."""
    return any(not queue_path(root, w).exists() for w in earlier(windows, window))


def references(root, windows, window):
    """The frozen in-batch references for this window's wave: one per kernel the earlier windows selected."""
    mine = wave(window)
    path = root / 'design/waves' / mine / 'batch_references.private.json'
    if path.exists():
        value = read(path); verify_refs(value['sources'])
        return value['references']
    sources, rows = [], []
    for w in earlier(windows, window):
        queue = queue_path(root, w)
        if not queue.exists():
            raise IntegrityError('in-batch references requested before an earlier wave froze its queue: ' + w['id'])
        sources.append(ref(queue))
        for row in read(queue).get('rows', []):
            rid = 'batch:' + row['source_candidate_id']
            if any(r['reference_id'] == rid for r in rows):
                continue  # planner versions of one idea share its kernel
            kernel = row['kernel']
            detail = ' '.join(f'{label}: {kernel[k]}' for label, k in (('Recipe', 'recipe'), ('What is new', 'what_is_new'))
                              if kernel.get(k))
            # central_relation is the text the index embeds (already embedded as a selector query); the summary adds the
            # recipe so authors and the selector see about as much as for a bank task
            rows.append({'reference_id': rid, 'source': 'accepted_generated_task', 'archive_tier': 'in_batch_selected',
                         'central_relation': kernel['central_relation'], 'summary': detail or kernel['central_relation']})
    return frozen(path, {'wave': mine, 'sources': sources, 'references': rows,
                         'policy': 'kernels selected by earlier reference waves of this batch'})['references']


def index(root, windows, window, base):
    """The reference index for this window: the run's index, extended for windows of a later wave."""
    from .retrieval import ReferenceIndex
    extra = references(root, windows, window) if earlier(windows, window) else []
    if not extra:
        return base
    value = ReferenceIndex(list(base.records.values()) + extra, base.provider, provider_identity=base.provider_identity)
    value.batch_references = extra
    return value
