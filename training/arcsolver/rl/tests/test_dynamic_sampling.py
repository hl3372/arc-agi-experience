"""Dynamic sampling: sampler_dispatch.generate_rounds against fake engines that run the same stream protocol as
vllm_worker's stream (need / hold / abort / heartbeat), and FrontierMixOrder.take_more."""
import concurrent.futures as cf
import random
import threading
import time

from arcsolver.rl import mixorder as M
from arcsolver.rl import sampler_dispatch as D


class FakeEngine:
    """vllm_worker.stream's control loop with a fake decoder: an item finishes after dur(key, j) ticks; at most
    target items decode at once. Messages go through on_message exactly like EngineClient.stream (a non-None return
    is the engine's next reply)."""

    def __init__(self, dur, tick=0.0005):
        self.dur, self.tick, self.calls = dur, tick, 0

    def stream(self, cmd, on_message, items, params, target_seqs, kv_high, topk_dir, tag, heartbeat_s=0.0):
        assert cmd == "stream"
        self.calls += 1
        inbox = []                                     # replies the client sent (strict request/reply)

        def send(m):
            r = on_message(m)
            if r is not None:
                inbox.append(r)
            return r

        pending, asked, exhausted, aborted, done, gen = {}, False, False, 0, 0, 0
        last_ask = time.time()
        t = 0

        def add(its):
            for it in its:
                pending[(it["key"], it["j"])] = [it, t + self.dur(it["key"], it["j"])]

        add(items)
        held = False
        while True:                                       # mirrors vllm_worker.stream's control loop
            want = target_seqs - len(pending)
            room = not exhausted and want > 0
            due = heartbeat_s > 0 and time.time() - last_ask >= heartbeat_s
            if not asked and ((room and (not held or due)) or (pending and due)):
                send({"type": "need", "n": want if room else 0})
                asked, last_ask = True, time.time()
            if asked and (not pending or inbox):
                assert len(inbox) == 1, inbox             # exactly one reply per need, never unsolicited
                reply = inbox.pop()
                asked = False
                if reply.get("abort"):
                    aborted = len(pending)
                    pending.clear()
                    break
                held = bool(reply.get("hold"))
                if reply.get("items"):
                    add(reply["items"])
                elif not held:
                    exhausted = True
                continue
            if not pending:
                if held and not exhausted:
                    time.sleep(0.0005)
                    continue
                break
            t += 1
            time.sleep(self.tick)
            for k in [k for k, (it, end) in pending.items() if end <= t]:
                it, _ = pending.pop(k)
                n = self.dur(it["key"], it["j"])
                done += 1
                gen += n
                send({"type": "sample", "key": it["key"], "j": it["j"],
                      "rec": {"token_ids": [0] * n, "finish": "stop", "seed": it["seed"]}})
        assert not inbox, inbox                           # nothing left unanswered
        return {"type": "end", "seconds": 0.0, "gen_tokens": gen, "samples": done, "aborted": aborted}


def reward(key, j):
    return 1.0 if (key * 7 + j * 3) % 5 == 0 else 0.0    # a deterministic mix of right / wrong


def informative(outs, start):
    return sum(max(reward(start + i, j) for j in range(len(g))) > min(reward(start + i, j) for j in range(len(g)))
               for i, g in enumerate(outs))


def run(target, round_prompts=4, cap=40, first=8, n=4, engines=4, seqs=3, hold_at=None, seed=0):
    rng = random.Random(seed)
    lens = {}

    def dur(key, j):
        return lens.setdefault((key, j), rng.randint(1, 60))

    eng = [FakeEngine(dur) for _ in range(engines)]
    st = dict(prompts=first, judged=0, inf=0, rounds=[])
    lock = threading.Lock()

    def req(lo, hi):
        return dict(prompts=[[1, 2, i] for i in range(lo, hi)], max_tokens=[100] * (hi - lo),
                    seeds=[1000 + 8 * i for i in range(lo, hi)], n_per_prompt=[n] * (hi - lo))

    def more():
        with lock:
            if hold_at is not None and st["prompts"] >= hold_at and st["judged"] < st["prompts"]:
                return D.HOLD
            k = min(round_prompts, cap - st["prompts"])
            if k <= 0:
                return None
            lo = st["prompts"]
            st["prompts"] += k
            return req(lo, lo + k)

    def on_round(r, outs):
        inf = informative(outs, st["judged"])
        st["judged"] += len(outs)
        st["inf"] += inf
        st["rounds"].append((len(outs), inf))
        return target and st["inf"] >= target

    base = req(0, first)
    request = dict(base, params={"n": n}, topk_dir=None, tag="t")
    with cf.ThreadPoolExecutor(max_workers=engines) as ex:
        res = D.generate_rounds(eng, ex, request, seqs, on_sample=None, more=more, on_round=on_round,
                                heartbeat_s=0.001)
    return res, st


def test_stops_on_whole_rounds_and_aborts_the_rest():
    res, st = run(target=5)
    dyn = res["dispatch"]["dyn"]
    assert dyn["stopped"] and st["inf"] >= 5
    # included = every prompt of the judged rounds, in dispatch order, complete
    assert len(res["outputs"]) == sum(n for n, _ in st["rounds"]) == dyn["prompts_included"]
    assert all(o is not None for g in res["outputs"] for o in g) and all(len(g) == 4 for g in res["outputs"])
    assert dyn["prompts_dispatched"] >= dyn["prompts_included"]
    # seeds follow the global prompt index (round 0 unchanged, extra rounds continue it)
    for i, g in enumerate(res["outputs"]):
        assert [s["seed"] for s in g] == [1000 + 8 * i + j for j in range(4)]


def test_without_a_stop_every_round_is_included():
    res, st = run(target=10 ** 6, cap=20)
    dyn = res["dispatch"]["dyn"]
    assert not dyn["stopped"] and dyn["samples_aborted"] == 0
    assert len(res["outputs"]) == 20 == dyn["prompts_dispatched"]


def test_hold_waits_for_judging_and_then_continues():
    res, st = run(target=6, hold_at=8, cap=40)
    dyn = res["dispatch"]["dyn"]
    assert dyn["holds"] > 0 and dyn["stopped"]
    assert len(res["outputs"]) == dyn["prompts_included"]


def test_plain_request_without_more_matches_stream():
    res, st = run(target=0, cap=8)                      # more() gives nothing beyond round 0 (cap = first)
    assert len(res["outputs"]) == 8 and not res["dispatch"]["dyn"]["stopped"]


def test_take_more_keeps_the_step_and_never_repeats():
    bank, rest = list(range(0, 50)), list(range(50, 60))
    o = M.FrontierMixOrder([1] * 60, [bank, rest], seed=3, schedule=[(1, 0.0, 0.0), (50, 0.0, 0.0), (51, 0.2, 0.0)],
                           max_age=25)
    first = o.take(8)
    t = o.t
    more = o.take_more(8) + o.take_more(8)
    assert o.t == t == 1                                 # the step does not advance
    assert len(set(first + more)) == 24                 # no index twice in one step
    assert set(o.last) == set(first + more)             # update() sees every round
    nxt = o.take(8)
    assert o.t == 2 and not set(nxt) & set(first + more)
