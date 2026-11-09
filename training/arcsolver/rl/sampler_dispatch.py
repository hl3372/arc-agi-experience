"""Queue complete prompt groups across engines without changing the rollout policy."""
from collections import deque
import math
import threading
import time


def generate_dynamic(engines, executor, request, chunk_prompts):
    if chunk_prompts < 1 or not engines:
        raise ValueError('dynamic dispatch needs engines and a positive chunk size')
    prompts = request['prompts']
    if not prompts:
        return {'outputs': [], 'gen_tokens': 0, 'seconds': 0., 'dispatch': {}}
    # Keep every GPU usable for small batches (including smoke and evaluation).
    chunk = min(chunk_prompts, max(1, math.ceil(len(prompts)/len(engines))))
    batches = deque(list(range(i, min(i+chunk, len(prompts))))
                    for i in range(0, len(prompts), chunk))
    first = [batches.popleft() if batches else None for _ in engines]
    lock = threading.Lock()
    started = time.monotonic()

    def run_engine(e):
        items, gen, active, groups = [], 0, 0., 0
        indices = first[e]
        while indices is not None:
            limits = request.get('max_tokens')
            seeds = request.get('seeds')
            counts = request.get('n_per_prompt')
            sub = dict(prompts=[prompts[i] for i in indices], params=request['params'],
                       max_tokens=[limits[i] for i in indices] if isinstance(limits, list) else limits,
                       seeds=[seeds[i] for i in indices] if seeds else None,
                       n_per_prompt=[counts[i] for i in indices] if counts is not None else None,
                       topk_dir=request.get('topk_dir'),
                       tag=f"{request.get('tag','g')}e{e}b{indices[0]}")
            before = time.monotonic()
            result = engines[e].call('generate', **sub)
            active += time.monotonic()-before
            if len(result['outputs']) != len(indices):
                raise RuntimeError('engine returned the wrong number of prompt groups')
            for i, outputs in zip(indices, result['outputs']):
                expected = counts[i] if counts is not None else request['params']['n']
                if len(outputs) != expected:
                    raise RuntimeError('engine returned the wrong number of samples in a group')
                items.append((i, outputs))
            gen += result['gen_tokens']
            groups += len(indices)
            with lock:
                indices = batches.popleft() if batches else None
        return items, gen, active, groups, time.monotonic()-started

    futures = [executor.submit(run_engine, e) for e in range(len(engines))]
    outputs, gen, stats = [None]*len(prompts), 0, []
    for future in futures:
        items, tokens, active, groups, finished = future.result()
        for i, group in items:
            if outputs[i] is not None:
                raise RuntimeError('prompt group was dispatched twice')
            outputs[i] = group
        gen += tokens
        stats.append(dict(groups=groups, active_s=active, finished_s=finished))
    if any(group is None for group in outputs):
        raise RuntimeError('dynamic dispatch lost a prompt group')
    elapsed = time.monotonic()-started
    return dict(outputs=outputs, gen_tokens=gen, seconds=elapsed,
                dispatch=dict(chunk_prompts=chunk, engines=stats,
                              tail_s=elapsed-min(s['finished_s'] for s in stats if s['groups'])))


def generate_stream(engines, executor, request, target_seqs, order=None, kv_high=0.9, on_sample=None):
    """Continuous refill across engines: one global queue of single samples; every engine keeps
    about target_seqs of them decoding and asks for more as samples finish (vllm_worker "stream"), so no GPU waits
    for the slowest sample of a job while work is queued. Sample j of prompt i is the n=1 request with seed
    seeds[i] + j (vLLM's own seeding of an n > 1 request's children), so groups, seeds, completion caps and top-k
    files are the same whichever engine runs a sample. `order` optionally lists the prompt indices in dispatch
    order (default: as given). Returns the generate() result shape (outputs per prompt, samples in j order).
    on_sample(i, j, rec), if given, is called once per sample as it arrives (in that engine's dispatch thread,
    outside the lock): the RL loop scores samples there while the others still decode (--reward-stream). It must
    return quickly and must not modify rec."""
    if target_seqs < 1 or not engines:
        raise ValueError('streaming dispatch needs engines and a positive target')
    prompts = request['prompts']
    if not prompts:
        return {'outputs': [], 'gen_tokens': 0, 'seconds': 0., 'dispatch': {}}
    params = request['params']
    limits = request.get('max_tokens')
    seeds = request.get('seeds')
    counts = request.get('n_per_prompt') or [params['n']] * len(prompts)
    if len(counts) != len(prompts) or any(int(n) < 1 for n in counts):
        raise ValueError('n_per_prompt must have one positive sample count per prompt')
    # Sample-major: sample 0 of every prompt, then sample 1, ... Handing out a prompt's group consecutively would put
    # all samples of a hard prompt (often all near the token cap) on one engine, which then finishes long after the
    # others. Which engine runs a sample changes nothing else.
    prompt_order = list(order if order is not None else range(len(prompts)))
    queue = deque()
    for j in range(max(counts)):
        for i in prompt_order:
            if j < counts[i]:
                queue.append(dict(key=i, j=j, prompt=list(prompts[i]),
                                  seed=(seeds[i] + j) if seeds and seeds[i] is not None else None,
                                  max_tokens=limits[i] if isinstance(limits, list) else limits))
    total = len(queue)
    lock = threading.Lock()
    started = time.monotonic()
    outputs = [[None] * n for n in counts]

    def take(k, refill=True):
        # Fair share near the end: an engine whose samples finished early would ask for many of the last samples,
        # and those late starters set the tail. Once the queue is short, a refill gets at most its share of what is
        # left (first batches are not capped).
        with lock:
            k = min(k, len(queue))
            if refill:
                k = min(k, max(1, math.ceil(len(queue) / len(engines))))
            return [queue.popleft() for _ in range(k)]

    def run_engine(e):
        stats = dict(samples=0, gen=0)

        def on_message(m):
            if m['type'] == 'need':
                return {'items': take(m['n'])}
            if m['type'] == 'sample':
                with lock:
                    if outputs[m['key']][m['j']] is not None:
                        raise RuntimeError('sample returned twice')
                    outputs[m['key']][m['j']] = m['rec']
                stats['samples'] += 1
                stats['gen'] += len(m['rec']['token_ids'])
                if on_sample is not None:
                    on_sample(m['key'], m['j'], m['rec'])
                return None
            raise RuntimeError(f'unexpected engine message {m.get("type")}')

        extra = {'force': request['force']} if request.get('force') else {}   # --train/eval-force-inline
        end = engines[e].stream('stream', on_message, items=take(target_seqs, refill=False), params=params,
                                target_seqs=target_seqs, kv_high=kv_high, topk_dir=request.get('topk_dir'),
                                tag=request.get('tag', 'g'), **extra)
        stats.update(active_s=end['seconds'], finished_s=time.monotonic() - started,
                     kv_usage_max=end.get('kv_usage_max'),
                     **{k: end[k] for k in ('forced', 'forced_gen_tokens', 'forced_cached_tokens',
                                            'forced_prompt_tokens') if k in end})
        return stats

    futures = [executor.submit(run_engine, e) for e in range(len(engines))]
    stats = [f.result() for f in futures]
    missing = sum(o is None for group in outputs for o in group)
    if missing or sum(s['samples'] for s in stats) != total:
        raise RuntimeError(f'streaming dispatch lost {missing} of {total} samples')
    elapsed = time.monotonic() - started
    busy = [s for s in stats if s['samples']]
    return dict(outputs=outputs, gen_tokens=sum(s['gen'] for s in stats), seconds=elapsed,
                dispatch=dict(chunk_prompts=0, target_seqs=target_seqs, kv_high=kv_high, engines=stats,
                              tail_s=elapsed - min(s['finished_s'] for s in busy) if busy else 0.))


HOLD = object()      # more(): nothing to add right now, but a later call may add a round (not exhausted)


def generate_rounds(engines, executor, request, target_seqs, kv_high=0.9, on_sample=None, more=None, on_round=None,
                    heartbeat_s=2.0):
    """Dynamic sampling over rounds (experimental): engines with free slots start generating more prompts without
    waiting for the rest of the batch. generate_stream's continuous refill, but the work is a growing list of prompt
    rounds:
      - round 0 is `request`; whenever an engine has room and the queue is empty, more() is called (under the queue
        lock) and may return the next round's request (prompts, max_tokens, seeds, n_per_prompt), HOLD (nothing now,
        ask again later) or None (no more rounds this step). Global prompt keys run across rounds (round r's prompt i
        is key start_r + i), so seeds, top-k files and on_sample(key, j, rec) stay unique.
      - rounds are judged in dispatch order, each once all its samples are in: on_round(r, outputs_of_round) returns
        True to stop. Inclusion is per whole round, never per finished group, so the batch is not biased towards
        groups whose samples happen to be short (a group-level "first N informative to finish" rule would be).
      - on stop, every engine's next "need" (the engines ask at least every heartbeat_s while decoding) is answered
        with abort: the engine aborts its in-flight samples and ends at once.
    Returns the generate() shape for the included rounds only (rounds 0..stop_round, or every round if none
    stopped), plus dispatch stats: rounds, prompts dispatched / included, samples aborted, per-engine times."""
    if target_seqs < 1 or not engines:
        raise ValueError('streaming dispatch needs engines and a positive target')
    params = request['params']
    lock, judge = threading.Lock(), threading.Lock()
    queue, rounds, outputs = deque(), [], []
    st = dict(stopping=False, stop_round=None, judged=0, more_done=more is None, holds=0)
    started = time.monotonic()

    def add_round(req):                                   # under lock
        ps = req['prompts']
        if not ps:
            return
        counts = req.get('n_per_prompt') or [params['n']] * len(ps)
        if len(counts) != len(ps) or any(int(n) < 1 for n in counts):
            raise ValueError('n_per_prompt must have one positive sample count per prompt')
        limits, seeds = req.get('max_tokens'), req.get('seeds')
        start = len(outputs)
        outputs.extend([None] * n for n in counts)
        rounds.append(dict(start=start, n=len(ps), remaining=sum(counts), t_added=time.monotonic() - started))
        for j in range(max(counts)):                      # sample-major within the round (see generate_stream)
            for i in range(len(ps)):
                if j < counts[i]:
                    queue.append(dict(key=start + i, j=j, prompt=list(ps[i]),
                                      seed=(seeds[i] + j) if seeds and seeds[i] is not None else None,
                                      max_tokens=limits[i] if isinstance(limits, list) else limits))

    def round_of(key):
        for r in range(len(rounds) - 1, -1, -1):
            if key >= rounds[r]['start']:
                return r
        raise KeyError(key)

    def take(k, refill=True):
        """Up to k samples; [] with done=False means "nothing now" (hold), done=True means no more work ever."""
        with lock:
            if st['stopping']:
                return [], True
            if not queue and not st['more_done']:
                nxt = more()
                if nxt is None:
                    st['more_done'] = True
                elif nxt is HOLD:
                    st['holds'] += 1
                else:
                    add_round(nxt)
            k = min(k, len(queue))
            if refill and st['more_done']:                # fair share only once no further round can come
                k = min(k, max(1, math.ceil(len(queue) / len(engines))))
            items = [queue.popleft() for _ in range(k)]
            return items, (not items and st['more_done'])

    with lock:
        add_round(request)

    def judge_rounds():
        with judge:
            while not st['stopping'] and st['judged'] < len(rounds) and rounds[st['judged']]['remaining'] == 0:
                r = st['judged']
                rd = rounds[r]
                stop = bool(on_round(r, outputs[rd['start']:rd['start'] + rd['n']])) if on_round else False
                rd['t_judged'] = time.monotonic() - started
                st['judged'] += 1
                if stop:
                    with lock:
                        st['stopping'], st['stop_round'] = True, r
                        queue.clear()

    def run_engine(e):
        stats = dict(samples=0, gen=0, aborted=0)

        def on_message(m):
            if m['type'] == 'need':
                if m['n'] <= 0:                              # heartbeat while decoding
                    return {'items': [], 'abort': True} if st['stopping'] else {'items': [], 'hold': True}
                items, done = take(m['n'])
                if st['stopping']:
                    return {'items': [], 'abort': True}
                if items:
                    return {'items': items}
                return {'items': []} if done else {'items': [], 'hold': True}
            if m['type'] == 'sample':
                with lock:
                    key, j = m['key'], m['j']
                    if outputs[key][j] is not None:
                        raise RuntimeError('sample returned twice')
                    outputs[key][j] = m['rec']
                    rounds[round_of(key)]['remaining'] -= 1
                stats['samples'] += 1
                stats['gen'] += len(m['rec']['token_ids'])
                if on_sample is not None:
                    on_sample(m['key'], m['j'], m['rec'])
                judge_rounds()
                return None
            raise RuntimeError(f'unexpected engine message {m.get("type")}')

        first, _ = take(target_seqs, refill=False)
        extra = {'force': request['force']} if request.get('force') else {}   # --train/eval-force-inline
        end = engines[e].stream('stream', on_message, items=first, params=params, target_seqs=target_seqs,
                                kv_high=kv_high, topk_dir=request.get('topk_dir'), tag=request.get('tag', 'g'),
                                heartbeat_s=heartbeat_s, **extra)
        stats.update(active_s=end['seconds'], finished_s=time.monotonic() - started,
                     kv_usage_max=end.get('kv_usage_max'), aborted=end.get('aborted', 0),
                     **{k: end[k] for k in ('forced', 'forced_gen_tokens', 'forced_cached_tokens',
                                            'forced_prompt_tokens') if k in end})
        return stats

    futures = [executor.submit(run_engine, e) for e in range(len(engines))]
    stats = [f.result() for f in futures]
    judge_rounds()                                        # rounds that completed with the last samples
    last = st['stop_round'] if st['stop_round'] is not None else len(rounds) - 1
    n_incl = rounds[last]['start'] + rounds[last]['n']
    included = outputs[:n_incl]
    missing = sum(o is None for group in included for o in group)
    if missing:
        raise RuntimeError(f'dynamic sampling lost {missing} samples of the included rounds')
    elapsed = time.monotonic() - started
    busy = [s for s in stats if s['samples']]
    return dict(outputs=included, gen_tokens=sum(s['gen'] for s in stats), seconds=elapsed,
                dispatch=dict(chunk_prompts=0, target_seqs=target_seqs, kv_high=kv_high, engines=stats,
                              tail_s=elapsed - min(s['finished_s'] for s in busy) if busy else 0.,
                              dyn=dict(rounds=len(rounds), rounds_included=last + 1, stopped=st['stop_round'] is not None,
                                       prompts_dispatched=len(outputs), prompts_included=n_incl,
                                       samples_aborted=sum(s['aborted'] for s in stats), holds=st['holds'],
                                       round_times=[(round(r['t_added'], 1), round(r.get('t_judged', -1), 1))
                                                    for r in rounds])))
