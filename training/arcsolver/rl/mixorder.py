"""Frontier-mix prompt order (--pool-order frontier_mix): every step's prompts are
  - fresh bank tasks (default 75%), continuing the run's seeded bank permutation;
  - revisits of past frontier tasks (default 20%): tasks whose latest group had lo..hi of G correct (default 1..6
    of 8), oldest first, at least min_gap steps after they were last drawn; a revisit that lands lo..hi again is
    re-queued, one outside the band leaves the queue (solved: graduated; 0/G: dropped);
  - fresh tasks of the other sources (default 5%, the ARC training set), continuing their permutation.
Revisit slots the queue can't fill go to fresh bank tasks. Both fresh strata are the two WeightedOrders of
pool.StratifiedOrder (same strata, same seeds), so a stratified checkpoint converts (`load_state_dict` accepts its
state) and the fresh draws continue where it stopped; `seed_history` then fills the queue from the run's rollout
records. train.py feeds `update` with each step's per-group outcomes (`frontier_results`). Stdlib only.
Pool extension (--pool-extend): `extend_from_records` continues a checkpoint on a grown pool; the never-drawn bank
tasks go to `pending` and are drawn before the bank order."""
from __future__ import annotations

import collections
import gzip
import re
import json
import os
import random

from arcsolver.rl import pool as rp

KIND = "frontier_mix"


class FrontierMixOrder:
    def __init__(self, counts: list, strata: list, seed: int, frontier_share: float = 0.20, rest_share: float = 0.05,
                 min_gap: int = 5, lo: int = 1, hi: int = 6, schedule=None, max_age: int = 0, new_per_step: int = 0):
        """schedule (opt-in): [(step, frontier_share, rest_share), ...] knots with increasing steps;
        the step being drawn gets the shares linearly interpolated between its two knots (the first / last knot's
        shares before / after them) and frontier_share / rest_share are ignored. max_age (opt-in): only queue
        entries drawn at most this many steps ago are revisited (0 = no cap). Without either, draws are unchanged."""
        if len(strata) != 2 or not strata[0] or not strata[1]:
            raise ValueError("need two non-empty strata: [bank, rest]")
        if frontier_share < 0 or rest_share < 0 or frontier_share + rest_share >= 1:
            raise ValueError(f"shares must be >= 0 and leave room for fresh bank: {frontier_share}, {rest_share}")
        self.schedule = [(int(s), float(f), float(r)) for s, f, r in schedule] if schedule else None
        for j, (s, f, r) in enumerate(self.schedule or []):
            if f < 0 or r < 0 or f + r >= 1 or (j and s <= self.schedule[j - 1][0]):
                raise ValueError(f"schedule knots need increasing steps and shares >= 0 summing below 1: {schedule}")
        if max_age < 0:
            raise ValueError(f"max_age must be >= 0: {max_age}")
        if new_per_step < 0:
            raise ValueError(f"new_per_step must be >= 0: {new_per_step}")
        self.new_per_step = int(new_per_step)
        self.max_age = int(max_age)
        if min_gap < 1 or not 1 <= lo <= hi:
            raise ValueError(f"need min_gap >= 1 and 1 <= lo <= hi: {min_gap}, {lo}, {hi}")
        self.strata = [list(s) for s in strata]
        # exactly pool.StratifiedOrder's sub-orders (seed + 7919 * j), so its checkpointed state converts
        self.orders = [rp.WeightedOrder([counts[i] for i in s], seed + 7919 * j) for j, s in enumerate(self.strata)]
        self.seed, self.frontier_share, self.rest_share = seed, frontier_share, rest_share
        self.min_gap, self.lo, self.hi = min_gap, lo, hi
        self.t = 0                    # steps drawn so far; the step being drawn is t + 1
        self.queue = {}               # pool index -> [last step drawn, correct, samples]
        self.last = {}                # pool index -> source of the latest take: "bank" / "rest" / "frontier"
        self.stats = {}
        self.pending = []             # never-drawn bank tasks, drawn before the bank order (--pool-extend)
        self.new_queue = []           # never-drawn bank tasks fed at most new_per_step per step (--pool-new-per-step)
        # --pool-focus-*: a task set drawn at least focus_per_step times per step: its
        # never-drawn tasks from new_queue first, then revisits, fewest draws first (then oldest last draw); fresh
        # draws never pick a focus task. Set by set_focus() after construction / restore; not checkpointed (the
        # draw counts are re-seeded from the rollout records at every start).
        self.focus, self.focus_set, self.focus_per_step, self.focus_draws = [], set(), 0, {}
        # focus_order "shuffle" (new and already-drawn focus tasks mixed, rather than new ones first): the focus set is
        # drawn in seeded shuffled cycles (cycle c = the focus list shuffled with seed c),
        # never-drawn and drawn tasks alike; focus_cursor = positions consumed so far (re-seeded from the records).
        self.focus_order, self.focus_cursor = "fewest", 0
        # --pool-focus-cap (bound how many tasks of a sub-list, e.g. never-solved ones, one step may take): with
        # focus_order "shuffle", a step takes at most focus_cap tasks of
        # focus_cap_set; the cycle positions of further ones are passed over (their turn comes in the next cycle) and
        # the next tasks of the cycle fill the step. From step focus_cap_from on. Set by set_focus_cap().
        self.focus_cap_set, self.focus_cap, self.focus_cap_from, self.focus_cap_sched = set(), 0, 1, []
        self._cap_step, self._cap_n, self._cap_skipped, self._cycles = 0, 0, 0, {}

    # --- drawing -------------------------------------------------------------------------------------------
    def _tie(self, i):
        return random.Random(self.seed * 1_000_003 + i).random()

    def shares(self, step: int) -> tuple:
        """(frontier_share, rest_share) for drawing `step`."""
        if not self.schedule:
            return self.frontier_share, self.rest_share
        knots = self.schedule
        if step <= knots[0][0]:
            return knots[0][1], knots[0][2]
        for (s0, f0, r0), (s1, f1, r1) in zip(knots, knots[1:]):
            if step == s1:
                return f1, r1
            if step < s1:
                w = (step - s0) / (s1 - s0)
                return f0 + w * (f1 - f0), r0 + w * (r1 - r0)
        return knots[-1][1], knots[-1][2]

    def quotas(self, k: int, step: int = None) -> tuple:
        fs, rs = self.shares(self.t + 1 if step is None else step)
        n_f = int(round(fs * k))
        n_r = int(round(rs * k))
        return k - n_f - n_r, n_f, n_r

    def _fresh(self, j: int, *skip) -> int:
        """The next fresh pool index of stratum j that is in none of `skip`: pending bank tasks first (a skipped one
        stays pending), then the stratum's order (a skipped draw is consumed, as before)."""
        if j == 0:
            for n, i in enumerate(self.pending):
                if not any(i in s for s in skip):
                    del self.pending[n]
                    return i
        while True:
            i = self.strata[j][self.orders[j].take(1)[0]]
            if not any(i in s for s in skip):
                return i

    def _take_new(self, limit: int, out: list, src: dict, *skip) -> int:
        """Up to `limit` tasks from the front of the new queue (a task already drawn this step stays queued)."""
        n, keep = 0, []
        while self.new_queue and n < limit:
            i = self.new_queue.pop(0)
            if i in src or any(i in s for s in skip):
                keep.append(i)
                continue
            out.append(i)
            src[i] = "bank"
            n += 1
        self.new_queue[:0] = keep
        return n

    def set_focus(self, focus: list, per_step: int, draws: dict, order: str = "fewest", cursor: int = 0,
                  max_frac: float = 0.5) -> None:
        """focus: pool indices; draws: pool index -> [draws so far, step of the latest draw] (focus_draws_from_records).
        order "fewest" (default): never-drawn focus tasks from new_queue first, then fewest draws / oldest; "shuffle":
        seeded shuffled cycles over the list AS GIVEN (pass it in a stable order, e.g. by task id), starting at
        position `cursor`; focus tasks waiting in new_queue are taken out of it (the cycle draws them). max_frac
        (--pool-focus-max-frac): the largest share of the bank the focus set may cover, since the other fresh draws
        skip it (default 0.5)."""
        if per_step < 0:
            raise ValueError(f"focus per_step must be >= 0: {per_step}")
        if order not in ("fewest", "shuffle") or cursor < 0:
            raise ValueError(f"focus order must be fewest or shuffle and cursor >= 0: {order}, {cursor}")
        if not 0.0 < max_frac < 1.0:
            raise ValueError(f"focus max_frac must be in (0, 1): {max_frac}")
        if per_step and len(set(self.strata[0]) - set(int(i) for i in focus)) < int(len(self.strata[0]) * (1 - max_frac)):
            raise ValueError(f"focus set covers more than {max_frac:g} of the bank: the fresh draws, which skip it, "
                             f"could run out (--pool-focus-max-frac)")
        if order == "shuffle":
            seen = set()
            self.focus = [int(i) for i in focus if not (int(i) in seen or seen.add(int(i)))]
        else:
            self.focus = sorted(set(int(i) for i in focus))
        self.focus_set = set(self.focus)
        self.focus_per_step = int(per_step)
        self.focus_order, self.focus_cursor = order, int(cursor)
        if order == "shuffle":
            self.new_queue = [i for i in self.new_queue if i not in self.focus_set]
        self.focus_draws = {int(i): [int(v[0]), int(v[1])] for i, v in draws.items() if int(i) in self.focus_set}

    def focus_cycle(self, c: int) -> list:
        key = (c, len(self.focus))
        if key not in self._cycles:                 # cached: _take_focus and the cursor replay read it per position
            order = list(self.focus)
            random.Random(self.seed * 1_000_003 + 52_361 * (c + 1) + 20_261_016).shuffle(order)
            self._cycles = {key: order}
        return self._cycles[key]

    def set_focus_cap(self, capped: list, cap: int, from_step: int = 1, schedule: list = None) -> None:
        """--pool-focus-cap: at most `cap` tasks of `capped` (pool indices) per step from step `from_step` on (see
        __init__); needs focus_order "shuffle" (call after set_focus). schedule: [(first step, cap), ...] with
        increasing steps replaces (from_step, cap); each cap holds until the next knot, so the cursor replay of a resume
        applies every step's own cap (append knots, never edit earlier ones)."""
        sched = [(int(a), int(b)) for a, b in schedule] if schedule else [(int(from_step), int(cap))]
        if self.focus_order != "shuffle" or any(a < 1 or b < 0 for a, b in sched) \
                or any(sched[j][0] <= sched[j - 1][0] for j in range(1, len(sched))):
            raise ValueError(f"focus cap needs focus order shuffle and knots (step >= 1, cap >= 0) with increasing steps: "
                             f"{self.focus_order}, {sched}")
        self.focus_cap_set = set(int(i) for i in capped) & self.focus_set
        self.focus_cap_sched = sched
        self.focus_cap_from, self.focus_cap = sched[0]

    def cap_at(self, step: int):
        """The cap of the capped set at this step (None before the first knot = no cap)."""
        c = None
        for a, b in self.focus_cap_sched:
            if step >= a:
                c = b
        return c

    def _cap_blocks(self, i: int, step: int, n_capped: int) -> bool:
        c = self.cap_at(step) if self.focus_cap_set else None
        return c is not None and i in self.focus_cap_set and n_capped >= c

    def _cap_counts(self, i: int, step: int) -> bool:
        return bool(self.focus_cap_set) and i in self.focus_cap_set and self.cap_at(step) is not None

    def replay_focus_cursor(self, per_step: dict) -> int:
        """The shuffle cursor after the recorded steps (per_step = {step: focus tasks drawn that step}, from the rollout
        records): the cycles walked again with the cap rule, as _take_focus walked them (its positions passed over for
        the cap count; a plain draw count would put a resumed run behind and re-draw recent tasks)."""
        pos, N = 0, len(self.focus)
        for step in sorted(per_step):
            n, n_cap, need = 0, 0, int(per_step[step])
            while n < need and N:
                c, off = divmod(pos, N)
                i = self.focus_cycle(c)[off]
                pos += 1
                if self._cap_blocks(i, step, n_cap):
                    continue
                n_cap += self._cap_counts(i, step)
                n += 1
        return pos

    def _take_focus(self, limit: int, out: list, src: dict, *skip) -> int:
        """Up to `limit` focus tasks not drawn yet this step and not waiting in new_queue: fewest draws first, then the
        oldest latest draw, then a seeded tie; with focus_order "shuffle" the next positions of the shuffled cycles."""
        if limit <= 0:
            return 0
        if self.focus_order == "shuffle":
            n, N, tried = 0, len(self.focus), 0
            while n < limit and N and tried < 2 * N:
                c, off = divmod(self.focus_cursor, N)
                i = self.focus_cycle(c)[off]
                self.focus_cursor += 1
                tried += 1
                if i in src or any(i in s for s in skip):
                    continue                                   # already in this step: its turn is spent
                if self._cap_blocks(i, self._cap_step, self._cap_n):
                    self._cap_skipped += 1
                    continue                                   # over the --pool-focus-cap: passed over this cycle
                self._cap_n += self._cap_counts(i, self._cap_step)
                out.append(i)
                src[i] = "bank"
                n += 1
            return n
        waiting = set(self.new_queue)
        cands = sorted((self.focus_draws.get(i, [0, 0])[0], self.focus_draws.get(i, [0, 0])[1], self._tie(i), i)
                       for i in self.focus if i not in src and i not in waiting and not any(i in s for s in skip))
        n = 0
        for _, _, _, i in cands[:limit]:
            out.append(i)
            src[i] = "bank"
            n += 1
        return n

    def _count_focus(self, taken, step: int) -> int:
        n = 0
        for i in taken:
            if i in self.focus_set:
                c = self.focus_draws.setdefault(i, [0, 0])
                c[0] += 1
                c[1] = step
                n += 1
        return n

    def take(self, k: int) -> list:
        step = self.t + 1
        n_b, n_f, n_r = self.quotas(k, step)
        ready = sorted((v[0], self._tie(i), i) for i, v in self.queue.items()
                       if v[0] <= step - self.min_gap and (not self.max_age or v[0] >= step - self.max_age))
        front = [i for _, _, i in ready[:n_f]]
        n_b += n_f - len(front)                                    # unfilled revisit slots: fresh bank
        out, src = list(front), {i: "frontier" for i in front}
        quota_new = (max(self.new_per_step, self.focus_per_step) if self.focus_per_step and self.focus_order != "shuffle"
                     else self.new_per_step)
        n_new = self._take_new(min(quota_new, n_b), out, src)
        n_foc = 0
        self._cap_step, self._cap_n, self._cap_skipped = step, 0, 0
        if self.focus_per_step:                                    # top the focus set up to its quota with revisits
            have = sum(i in self.focus_set for i in out)
            n_foc = self._take_focus(min(max(0, self.focus_per_step - have), n_b - n_new), out, src)
        waiting = set(self.new_queue)                              # fed by the quota only, never by fresh draws
        if self.focus_per_step:
            waiting |= self.focus_set                              # focus tasks come only through their quota
        for j, n in ((0, n_b - n_new - n_foc), (1, n_r)):
            got = 0
            while got < n:                                         # a fresh draw that is already in this step: skip
                i = self._fresh(j, src, waiting)
                out.append(i)
                src[i] = "bank" if j == 0 else "rest"
                got += 1
        n_focus = self._count_focus(out, step) if self.focus_per_step else 0
        self.t, self.last = step, src
        self.stats = {"frontier_mix/revisits": len(front), "frontier_mix/fresh_bank": n_b,
                      "frontier_mix/fresh_rest": n_r, "frontier_mix/queue_ready": len(ready)}
        if self.pending:
            self.stats["frontier_mix/pending_bank"] = len(self.pending)
        if self.new_per_step:
            self.stats.update({"frontier_mix/fresh_new": n_new, "frontier_mix/new_queue": len(self.new_queue)})
        if self.focus_per_step:
            dr = [self.focus_draws.get(i, [0, 0])[0] for i in self.focus]
            self.stats.update({"frontier_mix/focus": n_focus, "frontier_mix/focus_revisits": n_foc,
                               "frontier_mix/focus_min_draws": min(dr) if dr else 0,
                               "frontier_mix/focus_max_draws": max(dr) if dr else 0,
                               **({"frontier_mix/focus_cursor": self.focus_cursor} if self.focus_order == "shuffle" else {}),
                               **({"frontier_mix/focus_capped_taken": self._cap_n,
                                   "frontier_mix/focus_capped_passed": self._cap_skipped} if self.focus_cap_set else {})})
        if self.schedule:
            self.stats.update(zip(("frontier_mix/frontier_share", "frontier_mix/rest_share"), self.shares(step)))
        return out

    def take_more(self, k: int) -> list:
        """k more prompts for the step drawn by the latest take (dynamic sampling): the same step
        (t is not advanced, so the schedule, min_gap and max_age see the real step), the same quotas, never a pool
        index already drawn this step, and merged into `last` so update() credits every round's sources."""
        step = self.t
        n_b, n_f, n_r = self.quotas(k, step)
        drawn = set(self.last)
        ready = sorted((v[0], self._tie(i), i) for i, v in self.queue.items()
                       if i not in drawn and v[0] <= step - self.min_gap
                       and (not self.max_age or v[0] >= step - self.max_age))
        front = [i for _, _, i in ready[:n_f]]
        n_b += n_f - len(front)
        out, src = list(front), {i: "frontier" for i in front}
        n_new = self._take_new(min(max(0, self.new_per_step - self.stats.get("frontier_mix/fresh_new", 0)), n_b),
                               out, src, drawn)
        waiting = set(self.new_queue) | (self.focus_set if self.focus_per_step else set())
        for j, n in ((0, n_b - n_new), (1, n_r)):
            got = 0
            while got < n:
                i = self._fresh(j, src, drawn, waiting)
                out.append(i)
                src[i] = "bank" if j == 0 else "rest"
                got += 1
        if self.focus_per_step:
            self._count_focus(out, step)
        self.last.update(src)
        for key, n in (("frontier_mix/revisits", len(front)), ("frontier_mix/fresh_bank", n_b),
                       ("frontier_mix/fresh_rest", n_r), *((("frontier_mix/fresh_new", n_new),) if n_new else ())):
            self.stats[key] = self.stats.get(key, 0) + n
        return out

    # --- outcomes ----------------------------------------------------------------------------------------------
    def update(self, results) -> None:
        """results: (pool index, correct, samples, truncated) per group of the latest take."""
        prev, now, graduated, dropped = [], [], 0, 0
        for i, c, n, _trunc in results:
            if self.last.get(i) == "frontier" and i in self.queue:
                q = self.queue[i]
                prev.append(q[1] / max(1, q[2]))
                now.append(c / max(1, n))
            if self.lo <= c <= self.hi and c < n:
                self.queue[i] = [self.t, c, n]
            elif i in self.queue:
                del self.queue[i]
                graduated += c > self.hi
                dropped += c < self.lo
        self.stats.update({"frontier_mix/queue": len(self.queue), "frontier_mix/graduated": graduated,
                           "frontier_mix/dropped": dropped})
        if prev:                                                   # same tasks, earlier policy vs now (paired)
            self.stats.update({"frontier_mix/revisit_prev_success": sum(prev) / len(prev),
                               "frontier_mix/revisit_now_success": sum(now) / len(now)})

    def metrics(self) -> dict:
        return dict(self.stats)

    def seed_history(self, history) -> int:
        """history: [(step, [(pool index, correct, samples, truncated), ...]), ...] in step order, from the run's
        records before this order took over; sets t to the last step. Returns the queue size."""
        for step, results in history:
            for i, c, n, _trunc in results:
                if self.lo <= c <= self.hi and c < n:
                    self.queue[i] = [step, c, n]
                else:
                    self.queue.pop(i, None)
            self.t = max(self.t, step)
        return len(self.queue)

    # --- state -------------------------------------------------------------------------------------------------
    @property
    def epoch(self):
        return self.orders[0].epoch

    def state_dict(self) -> dict:
        d = {"kind": KIND, "sizes": [len(s) for s in self.strata], "orders": [o.state_dict() for o in self.orders],
                "t": self.t, "queue": {str(i): v for i, v in sorted(self.queue.items())},
                "params": {"frontier_share": self.frontier_share, "rest_share": self.rest_share,
                           "min_gap": self.min_gap, "lo": self.lo, "hi": self.hi,
                           **({"schedule": [list(x) for x in self.schedule]} if self.schedule else {}),
                           **({"max_age": self.max_age} if self.max_age else {})}}
        if self.pending:                                           # only while a pool extension is being drawn
            d["pending"] = list(self.pending)
        if self.new_queue:
            d["new_queue"] = list(self.new_queue)
        if self.new_per_step:
            d["params"]["new_per_step"] = self.new_per_step
        return d

    def load_state_dict(self, s: dict) -> None:
        """A frontier_mix state, or a pool.StratifiedOrder state (conversion: the fresh strata continue, the queue
        starts empty; follow with seed_history)."""
        if s.get("kind") not in (KIND, "stratified") or s.get("sizes") != [len(x) for x in self.strata]:
            raise ValueError(f"frontier-mix order mismatch: checkpoint {s.get('kind', 'weighted')} {s.get('sizes')}, "
                             f"now {KIND} {[len(x) for x in self.strata]}")
        for o, so in zip(self.orders, s["orders"]):
            o.load_state_dict(so)
        if s["kind"] == KIND:
            self.t = int(s["t"])
            self.queue = {int(i): list(v) for i, v in s["queue"].items()}
            self.pending = [int(i) for i in s.get("pending", [])]
            self.new_queue = [int(i) for i in s.get("new_queue", [])]
            if self.new_queue and not self.new_per_step:           # resumed without the quota: drain them first
                self.pending, self.new_queue = self.new_queue + self.pending, []

    def extend_from_records(self, s: dict, out_dir: str, step: int, index_of: dict, seed: int,
                            tiers: bool = False) -> dict:
        """--pool-extend (the bank grew mid-run): continue the frontier_mix checkpoint state
        `s` on a GROWN pool, whose indices no longer match the checkpoint's. From the run's rollout records of steps
        1..step (by task id, `index_of` = task id -> pool index of this pool): the revisit queue is re-seeded
        (seed_history: the same band rule as update), and every bank task never drawn goes to `pending` in a seeded
        shuffle, drawn before the bank order; once pending is empty the bank order starts its first epoch over the
        grown bank. Other strata whose size is unchanged (ARC train) continue from `s`. Every step 1..step needs its
        record (else the drawn set and the queue would be wrong): raises otherwise. Returns counts for the log.
        tiers (--pool-extend-tiers, the bank grew again during a later pass): pending also
        holds the bank tasks drawn fewer times than the most-drawn ones, by draw count (never-drawn first, exactly
        as without tiers, then drawn once, ...; each tier in its own seeded shuffle), so the pass in progress
        finishes before any task is drawn again; the bank order's first epoch over the grown bank starts after.
        With new_per_step > 0 (--pool-new-per-step: spread new tasks over the steps) the never-drawn
        tasks go to new_queue instead of pending: each step takes at most new_per_step of them, the rest of its bank
        quota comes from pending / the bank order (which skips tasks still waiting in new_queue)."""
        if s.get("kind") != KIND:
            raise ValueError(f"--pool-extend needs a {KIND} checkpoint order, not {s.get('kind', 'weighted')}")
        hist = history_from_records(out_dir, step, index_of)
        missing = sorted(set(range(1, step + 1)) - {st for st, _ in hist})
        if missing:
            raise ValueError(f"--pool-extend: no rollout records for steps {missing[:10]} in {out_dir}/rollouts")
        for j, (o, so, n_old) in enumerate(zip(self.orders, s["orders"], s["sizes"])):
            if j and n_old == len(self.strata[j]):
                o.load_state_dict(so)
        self.queue, self.last, self.t = {}, {}, 0
        n_queue = self.seed_history(hist)
        self.t = int(step)
        drawn = {g[0] for _, res in hist for g in res}
        self.pending = [i for i in self.strata[0] if i not in drawn]
        random.Random(seed * 1_000_003 + 20261013).shuffle(self.pending)
        self.new_queue = []
        if self.new_per_step:                                      # never-drawn tasks: fed per step, not first
            self.new_queue, self.pending = self.pending, []
        extra = {}
        if tiers:
            n_draws = collections.Counter(g[0] for _, res in hist for g in res)
            top = max((n_draws[i] for i in self.strata[0]), default=0)
            sizes = {0: len(self.pending) + len(self.new_queue)}
            for t in range(1, top):
                tier = [i for i in self.strata[0] if n_draws[i] == t]
                random.Random(seed * 1_000_003 + 20261013 + 7919 * t).shuffle(tier)
                self.pending += tier
                sizes[t] = len(tier)
            extra = {"tiers": sizes, "max_draws": top,
                     "at_max_draws": sum(n_draws[i] == top for i in self.strata[0]) if top else 0}
        if self.new_per_step:
            extra.update({"new_queue": len(self.new_queue), "new_per_step": self.new_per_step})
        return {"bank": len(self.strata[0]), "bank_before": s["sizes"][0],
                "bank_drawn": len(drawn & set(self.strata[0])), "pending": len(self.pending),
                "queue": n_queue, "queue_before": len(s.get("queue", {})), "steps": len(hist), "t": self.t, **extra}


def focus_draw_count(out_dir: str, from_step: int, upto_step: int, index_of: dict) -> int:
    """--pool-focus-order shuffle: the number of focus draws (fresh groups of tasks in index_of) in steps
    from_step..upto_step = the shuffled cycles' positions consumed since the shuffle began."""
    return sum(focus_draws_from_records(out_dir, from_step, upto_step, index_of, per_step_total=True).values())


def focus_draws_from_records(out_dir: str, from_step: int, upto_step: int, index_of: dict, per_step_total: bool = False) -> dict:
    """--pool-focus-*: pool index -> [fresh draws, step of the latest one] for the tasks in index_of (task id -> pool
    index), from rollouts/step_N.jsonl.gz of steps from_step..upto_step. A draw = a group of on-policy rollouts that is
    not a prefix continuation (as history_from_records). Reads only each record's leading task id / group and the
    prefix / off_policy markers (off_policy: records of other trainers; the long texts are JSON-escaped, so these key
    strings cannot occur inside them)."""
    head = re.compile(r'^\{"task_id": "([^"]+)".*?"group": (\d+)')
    draws = {}
    for step in range(max(1, from_step), upto_step + 1):
        path = os.path.join(out_dir, "rollouts", f"step_{step:06d}.jsonl.gz")
        if not os.path.exists(path):
            continue
        seen = set()
        with gzip.open(path, "rt") as f:
            for line in f:
                m = head.match(line[:2000])
                if not m or m.group(1) not in index_of:
                    continue
                if '"prefix": {' in line or '"off_policy": true' in line:
                    continue
                seen.add((index_of[m.group(1)], int(m.group(2))))
        if per_step_total:
            draws[step] = len(seen)
            continue
        for i, _g in seen:
            c = draws.setdefault(i, [0, 0])
            c[0] += 1
            c[1] = step
    return draws


def history_from_records(out_dir: str, upto_step: int, index_of: dict) -> list:
    """The run's per-group outcomes for steps 1..upto_step from rollouts/step_N.jsonl.gz (same definition as
    train.frontier_results: correct = outcome > 0, on-policy rollouts only). index_of maps task id -> pool
    index; tasks no longer in the pool are skipped. Missing step files are skipped."""
    hist = []
    for step in range(1, upto_step + 1):
        path = os.path.join(out_dir, "rollouts", f"step_{step:06d}.jsonl.gz")
        if not os.path.exists(path):
            continue
        groups = {}
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("off_policy", False) or r.get("prefix") or r["task_id"] not in index_of:
                    continue                           # prefix groups (--prefix-groups) are not draws
                g = groups.setdefault(r["group"], [index_of[r["task_id"]], 0, 0, 0])
                g[1] += r["seq"]["outcome"] > 0
                g[2] += 1
                g[3] += r["finish"] == "length"
        hist.append((step, [tuple(groups[k]) for k in sorted(groups)]))
    return hist
