"""Unequal engine speed must not misalign tasks, samples or behaviour log-probs (sampler_dispatch.generate_dynamic)."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from arcsolver.rl.sampler_dispatch import generate_dynamic


def request(n=12):
    return dict(prompts=[[i] for i in range(n)], params={'n': 2},
                n_per_prompt=[i%3+1 for i in range(n)], max_tokens=[100+i for i in range(n)],
                seeds=[1000+i for i in range(n)], topk_dir='/temporary/topk', tag='s17')


class Engine:
    def __init__(self, wait=None, release=None):
        self.calls, self.wait, self.release = [], wait, release

    def call(self, command, **kw):
        assert command == 'generate'
        self.calls.append(kw)
        if self.wait: assert self.wait.wait(timeout=5)
        groups = []
        for local_index, (prompt, count, limit, seed) in enumerate(zip(
                kw['prompts'], kw['n_per_prompt'], kw['max_tokens'], kw['seeds'])):
            assert limit == 100+prompt[0] and seed == 1000+prompt[0]
            groups.append([dict(task=prompt[0], sample=j, sampler_lp=[-.1-j],
                                topk_path=f"{kw['tag']}_{local_index}_{j}.npz") for j in range(count)])
        if self.release: self.release.set()
        return {'outputs': groups, 'gen_tokens': sum(map(len, groups)), 'seconds': 1.}


def test_faster_engine_takes_more_work_without_losing_group_alignment():
    release = threading.Event()
    slow, fast = Engine(wait=release), Engine()
    calls = 0
    original = fast.call
    def run(command, **kw):
        nonlocal calls
        result = original(command, **kw)
        calls += 1
        if calls == 5: release.set()
        return result
    fast.call = run
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = generate_dynamic([slow, fast], pool, request(), 2)
    assert len(fast.calls) == 5 and len(slow.calls) == 1
    assert [group[0]['task'] for group in result['outputs']] == list(range(12))
    assert [len(group) for group in result['outputs']] == [i%3+1 for i in range(12)]
    assert result['gen_tokens'] == 24
    assert all(sample['sampler_lp']==[-.1-sample['sample']]
               for group in result['outputs'] for sample in group)
    paths = [s['topk_path'] for group in result['outputs'] for s in group]
    assert len(paths)==len(set(paths))
    assert sum(e['groups'] for e in result['dispatch']['engines'])==12


def test_small_batch_uses_all_available_engines():
    engines = [Engine() for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = generate_dynamic(engines, pool, request(4), 4)
    assert all(len(e.calls)==1 for e in engines)
    assert result['dispatch']['chunk_prompts']==1


@pytest.mark.parametrize('failure', ['groups', 'samples', 'exception'])
def test_bad_engine_results_cannot_be_used_for_training(failure):
    class Broken(Engine):
        def call(self, command, **kw):
            if failure=='exception': raise RuntimeError('engine failure')
            result = super().call(command, **kw)
            if failure=='groups': result['outputs'].pop()
            else: result['outputs'][0].append({'extra_sample': True})
            return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(RuntimeError):
            generate_dynamic([Broken(), Engine()], pool, request(4), 1)
