"""Frontier task selection for RL (--pool-order frontier): spend rollouts where groups are mixed.

On hard pools a large share of groups can be 0/G and dropped (zero advantage), holding most truncations and a large
share of all sampled tokens; all-correct groups are dropped too. This order keeps a per-task history of group outcomes and
draws a step's tasks without replacement with probability proportional to a weight from that history:

  unseen     never sampled                                     w_unseen   (default 1.0)
  frontier   last result mixed (1..G-1 of G correct)            w_frontier (2.0)
  solved     last result G/G                                    w_solved   (0.15)
  dead       0/G and at least trunc_hi of its samples truncated w_dead     (0.1)
  failed     0/G, mostly finished but wrong                     w_failed   (0.5)

A task's state is its most recent group; `revisit_after` steps after a solved/dead/failed result its weight decays
back towards w_unseen (linear over that many steps), so nothing is excluded for good and the policy's progress is
picked up. Draws are seeded per step (seed, step), so a resumed run repeats exactly; state_dict holds the history.
Kept separate from pool.py's WeightedOrder/StratifiedOrder, whose checkpoint formats stay unchanged.
"""
from __future__ import annotations

import random

KIND = "frontier_v1"
DEFAULTS = dict(w_unseen=1.0, w_frontier=2.0, w_solved=0.15, w_dead=0.1, w_failed=0.5, trunc_hi=0.5,
                revisit_after=200)
STATES = ("unseen", "frontier", "solved", "dead", "failed")


class FrontierOrder:
    def __init__(self, n_items: int, seed: int, group_size: int, **params):
        if n_items <= 0:
            raise ValueError("empty pool")
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown frontier parameters {sorted(unknown)}")
        self.p = {**DEFAULTS, **params}
        if not 0 < self.p["trunc_hi"] <= 1 or self.p["revisit_after"] < 1:
            raise ValueError("trunc_hi must be in (0, 1] and revisit_after >= 1")
        self.n, self.seed, self.group_size = n_items, seed, group_size
        self.steps = 0
        self.visits = [0] * n_items
        self.state = ["unseen"] * n_items
        self.last_step = [-1] * n_items          # step of the most recent result
        self.successes = [0] * n_items           # cumulative correct samples
        self.samples = [0] * n_items             # cumulative samples

    # --- weights and draws ---------------------------------------------------------------------------------
    def weight(self, i: int) -> float:
        s = self.state[i]
        w = self.p["w_" + s]
        if s in ("solved", "dead", "failed"):
            age = self.steps - self.last_step[i]
            frac = min(1.0, max(0.0, age / self.p["revisit_after"]))
            w = w + frac * (self.p["w_unseen"] - w)
        return w

    def take(self, k: int) -> list:
        """k distinct items (all items if k >= n), drawn without replacement by weight; seeded per step."""
        rng = random.Random(self.seed * 1_000_003 + self.steps)
        k = min(k, self.n)
        # Efraimidis-Spirakis weighted sampling without replacement: key = u^(1/w), take the k largest
        keys = []
        for i in range(self.n):
            w = self.weight(i)
            u = rng.random()
            keys.append((u ** (1.0 / w) if w > 0 else 0.0, i))
        keys.sort(reverse=True)
        out = [i for _, i in keys[:k]]
        self.steps += 1
        for i in out:
            self.visits[i] += 1
        return out

    # --- history updates -------------------------------------------------------------------------------------
    def update(self, results) -> None:
        """results: iterable of (item, n_correct, n_samples, n_truncated) for the groups of the step just taken."""
        step = self.steps - 1
        for i, c, n, t in results:
            if n <= 0 or not 0 <= c <= n or not 0 <= t <= n:
                raise ValueError(f"bad group result for item {i}: {c}/{n} correct, {t} truncated")
            self.successes[i] += c
            self.samples[i] += n
            self.last_step[i] = step
            if 0 < c < n:
                self.state[i] = "frontier"
            elif c == n:
                self.state[i] = "solved"
            elif t >= self.p["trunc_hi"] * n:
                self.state[i] = "dead"
            else:
                self.state[i] = "failed"

    # --- metrics / checkpointing -----------------------------------------------------------------------------
    @property
    def epoch(self) -> float:
        return sum(self.visits) / self.n

    def counts(self) -> dict:
        out = {s: 0 for s in STATES}
        for s in self.state:
            out[s] += 1
        return out

    def state_dict(self) -> dict:
        return {"kind": KIND, "n": self.n, "seed": self.seed, "group_size": self.group_size, "params": dict(self.p),
                "steps": self.steps, "visits": list(self.visits), "state": list(self.state),
                "last_step": list(self.last_step), "successes": list(self.successes), "samples": list(self.samples)}

    def load_state_dict(self, s: dict) -> None:
        if s.get("kind") != KIND:
            raise ValueError(f"checkpoint holds a {s.get('kind', 'weighted')} order; this run uses {KIND}")
        for key in ("n", "seed", "group_size", "params"):
            mine = getattr(self, key) if key != "params" else self.p
            if s[key] != mine:
                raise ValueError(f"frontier order mismatch on {key}: checkpoint {s[key]}, now {mine}")
        self.steps = s["steps"]
        for key in ("visits", "state", "last_step", "successes", "samples"):
            if len(s[key]) != self.n:
                raise ValueError(f"frontier order {key} has {len(s[key])} entries, pool has {self.n}")
            setattr(self, key, list(s[key]))
