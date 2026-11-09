"""RL prompt pools, prompt orders and the pool-measurement scoring (pure python; used by config.py and train.py).

Pools (--pool): "arc_train" = the official ARC-AGI-2 training tasks, "bank" = generated tasks (--bank-dir), "mixed" =
both. Built with core.build_pool (so ARC-AGI-2 evaluation is excluded by directory, task id and prompt fingerprint,
and the periodic-eval bank holdout tasks are held out), with the unused source replaced by an empty directory.

Pool file (--pool-file, optional: a pool file with measured success rates): {"meta", "summary", "tasks": {task_id:
{"budgets": {"<B>": {"success", ...}}}}}. Plain {task_id: rate} and {task_id: {"success": rate}} files are accepted
too. Filtering keeps the tasks whose measured success is strictly between 0 and 1 (a group of G samples can then
disagree, so GRPO gets an outcome signal); --pool-oversample W repeats those W times per epoch instead of (or on
top of) filtering.

Budget scoring (sample_budget_stats, for producing such a file): every task is sampled once at the largest budget; a
sample counts at a smaller budget B only if it finished (EOS) within B generated tokens. The first B tokens of
generation do not depend on the cap, so this is exactly the success rate of a B-capped run (per-request seeds
additionally make each B-capped sample identical to the prefix, but are not needed for the rates).
"""
from __future__ import annotations

import json
import tempfile

from arcsolver.rl import core as oc
from arcsolver.rl import rewards as rr

POOLS = ("arc_train", "bank", "mixed")


# --- pool ---------------------------------------------------------------------------------------------------------

def build_rl_pool(pool: str, bank_dir: str, arc_train_dir: str, arc_eval_dir: str, trained_ids: set,
                  holdout_ids=(), load=None) -> tuple:
    """(items, report) for the chosen source(s); raises if an evaluation task could be in it."""
    if pool not in POOLS:
        raise ValueError(f"--pool must be one of {POOLS}")
    with tempfile.TemporaryDirectory(prefix="rl_no_source_") as empty:
        bank = bank_dir if pool in ("bank", "mixed") else empty
        arc = arc_train_dir if pool in ("arc_train", "mixed") else empty
        items, report = oc.build_pool(bank, arc, arc_eval_dir, trained_ids, holdout_ids=holdout_ids, load=load)
    oc.assert_no_eval(items, arc_eval_dir, load=load)
    report["pool"] = pool
    return items, report


def load_pool_rates(path: str, budget: int | None = None, key: str = "success") -> dict:
    """task_id -> measured rate. Measurement files hold one entry per budget; `budget` picks it (required when
    the file has several)."""
    with open(path) as f:
        d = json.load(f)
    tasks = d["tasks"] if isinstance(d, dict) and isinstance(d.get("tasks"), dict) else d
    out = {}
    for tid, v in tasks.items():
        if isinstance(v, (int, float)):
            out[tid] = float(v)
            continue
        if "budgets" in v:
            b = v["budgets"]
            if budget is None:
                if len(b) != 1:
                    raise ValueError(f"{path} has budgets {sorted(b, key=int)}; pass --pool-budget")
                (entry,) = b.values()
            else:
                if str(budget) not in b:
                    raise ValueError(f"{path}: task {tid} has no budget {budget} (has {sorted(b, key=int)})")
                entry = b[str(budget)]
            out[tid] = float(entry[key])
        else:
            out[tid] = float(v[key])
    return out


def filter_pool(items: list, rates: dict, mode: str = "learnable", unmeasured: str = "drop",
                lo: float = 0.0, hi: float = 1.0) -> tuple:
    """mode "learnable": keep lo < rate < hi; "positive": keep rate > lo (solved at least sometimes, including
    always-solved tasks: augmentation can break what the unaugmented task allowed); "none": keep all measured.
    unmeasured: "keep" | "drop"."""
    if mode not in ("learnable", "positive", "none") or unmeasured not in ("keep", "drop"):
        raise ValueError((mode, unmeasured))
    kept, rep = [], {"unmeasured": 0, "outside": 0, "kept": 0}
    for it in items:
        r = rates.get(it.task_id)
        if r is None:
            rep["unmeasured"] += 1
            if unmeasured == "keep":
                kept.append(it)
            continue
        if (mode == "learnable" and not (lo < r < hi)) or (mode == "positive" and not r > lo):
            rep["outside"] += 1
            continue
        kept.append(it)
    rep["kept"] = len(kept)
    return kept, rep


def oversample_counts(items: list, rates: dict, factor: int, lo: float = 0.0, hi: float = 1.0) -> list:
    """Per item: `factor` copies per epoch if its measured rate is strictly inside (lo, hi), else 1."""
    if factor < 1:
        raise ValueError("oversample factor must be >= 1")
    return [factor if (rates.get(it.task_id) is not None and lo < rates[it.task_id] < hi) else 1 for it in items]


class WeightedOrder:
    """oc.PromptOrder over an expanded index list (item i appears counts[i] times per epoch): seeded, epoch-wise
    shuffled, resumable. The prompts of one step are distinct items whenever the pool allows it (a repeat of an
    item already in the step is pushed to the next step)."""

    def __init__(self, counts: list, seed: int):
        self.expanded = [i for i, c in enumerate(counts) for _ in range(c)]
        self.n_items = len(counts)
        self.order = oc.PromptOrder(len(self.expanded), seed)
        self.carry = []
        self.solved = set()            # --pool-drop-solved: items never drawn again (a group went all-correct)

    def mark_solved(self, items) -> int:
        """--pool-drop-solved (drop a task once one of its groups is all correct): these items are skipped from now on.
        Returns how many were new."""
        new = {int(i) for i in items} - self.solved
        self.solved |= new
        return len(new)

    def take(self, k: int) -> list:
        live = self.n_items - len(self.solved)
        if live <= 0:
            raise RuntimeError("every pool item is solved (--pool-drop-solved): nothing left to draw")
        out, carry = [], []
        for i in self.carry:
            if i in self.solved:
                continue
            if len(out) < k and i not in out:
                out.append(i)
            else:
                carry.append(i)
        guard = skipped = 0
        while len(out) < k:
            i = self.expanded[self.order.take(1)[0]]
            if i in self.solved:
                skipped += 1
                if skipped > 100 * len(self.expanded):
                    raise RuntimeError("--pool-drop-solved: no unsolved item found")
                continue
            if i in out and len(set(out)) < live and guard < 10 * len(self.expanded):
                carry.append(i)
                guard += 1
                continue
            out.append(i)
        self.carry = carry
        return out

    @property
    def epoch(self):
        return self.order.epoch

    def state_dict(self) -> dict:
        return {"order": self.order.state_dict(), "carry": list(self.carry), "n_expanded": len(self.expanded),
                "solved": sorted(self.solved)}

    def load_state_dict(self, s: dict) -> None:
        if s.get("kind") == "stratified":
            raise ValueError("checkpoint holds a stratified order (--bank-share); this run uses one weighted order")
        if s["n_expanded"] != len(self.expanded):
            raise ValueError(f"weighted order mismatch: checkpoint {s['n_expanded']} slots, now {len(self.expanded)}")
        self.order.load_state_dict(s["order"])
        self.carry = list(s["carry"])
        self.solved = {int(i) for i in s.get("solved", [])}


class StratifiedOrder:
    """Per-step quotas over strata of the pool (e.g. a fixed share of every step's prompts from bank tasks): stratum j
    supplies round(shares[j] * k) of a step's k prompts (the last stratum takes the remainder),
    each stratum drawn by its own WeightedOrder over its items, so per-item counts (oversampling) still apply.
    Seeded and resumable like WeightedOrder; `epoch` is the first stratum's."""

    def __init__(self, counts: list, strata: list, shares: list, seed: int):
        if len(strata) != len(shares) or len(strata) < 2:
            raise ValueError("need one share per stratum and at least two strata")
        if any(not s for s in strata):
            raise ValueError("empty stratum")
        if any(x < 0 for x in shares) or abs(sum(shares) - 1.0) > 1e-6:
            raise ValueError(f"shares must be >= 0 and sum to 1: {shares}")
        self.strata = [list(s) for s in strata]
        self.shares = list(shares)
        self.orders = [WeightedOrder([counts[i] for i in s], seed + 7919 * j) for j, s in enumerate(self.strata)]

    def quotas(self, k: int) -> list:
        q = [int(round(x * k)) for x in self.shares[:-1]]
        return q + [k - sum(q)]

    def take(self, k: int) -> list:
        out = []
        for s, order, n in zip(self.strata, self.orders, self.quotas(k)):
            if n > 0:
                out += [s[j] for j in order.take(n)]
        return out

    @property
    def epoch(self):
        return self.orders[0].epoch

    def state_dict(self) -> dict:
        return {"kind": "stratified", "sizes": [len(s) for s in self.strata], "shares": list(self.shares),
                "orders": [o.state_dict() for o in self.orders]}

    def load_state_dict(self, s: dict) -> None:
        if s.get("kind") != "stratified" or s.get("sizes") != [len(x) for x in self.strata]:
            raise ValueError(f"stratified order mismatch: checkpoint {s.get('kind', 'weighted')} {s.get('sizes')}, "
                             f"now {[len(x) for x in self.strata]}")
        for o, so in zip(self.orders, s["orders"]):
            o.load_state_dict(so)


# --- measurement scoring -----------------------------------------------------------------------------------------

def sample_budget_stats(task: dict, tt: "rr.TokenText", finish: str, budgets: list) -> dict:
    """Per budget B (str): finished (EOS within B tokens), closed ("</think>" within B tokens), em_frac / em_all
    (only if finished within B: attempt-1 exact match, mean over test outputs / all outputs), loop (the digit-masked
    detector on the first B tokens), loop_verbatim, len (min(n, B))."""
    n = len(tt)
    text = tt.text
    close_tok = None
    think = rr.reasoning_text(text)
    if len(think) < len(text):
        close_tok = tt.token_at(len(think))
    full_ans = rr.answer_eval(task, text, finish)
    out = {}
    for B in budgets:
        fin = finish == "stop" and n <= B
        txt = text if n <= B else tt.prefix_text(B)
        lp = rr.loop_info(txt, "templated")
        ok = fin and full_ans["closed"]
        out[str(B)] = {"finished": fin, "closed": close_tok is not None and close_tok < B,
                       "em_frac": full_ans["em_frac"] if ok else 0.0, "em_all": bool(ok and full_ans["em_all"]),
                       "loop": lp["onset"] is not None, "loop_verbatim": lp["kind"] == "verbatim", "len": min(n, B)}
    return out


def task_budget_stats(samples: list, budgets: list) -> dict:
    """samples: per-sample outputs of sample_budget_stats -> per budget: success (mean em_frac = the RL outcome
    reward's mean), all_correct, pass@k-style solved (>= 1 correct), finished/closed/loop rates, mean length."""
    out = {}
    n = len(samples)
    for B in map(str, budgets):
        xs = [s[B] for s in samples]
        out[B] = {"n": n, "success": sum(x["em_frac"] for x in xs) / n,
                  "all_correct": sum(x["em_all"] for x in xs) / n,
                  "n_correct": sum(x["em_frac"] > 0 for x in xs),
                  "finished": sum(x["finished"] for x in xs) / n, "closed": sum(x["closed"] for x in xs) / n,
                  "loop": sum(x["loop"] for x in xs) / n, "loop_verbatim": sum(x["loop_verbatim"] for x in xs) / n,
                  "len_mean": sum(x["len"] for x in xs) / n}
    return out


def summarise_pool(tasks: dict, budgets: list, max_prompt_tokens: int | None = None) -> dict:
    """tasks: task_id -> {"budgets": task_budget_stats, "n_prompt_tokens"} -> per budget: task counts by success
    (0, strictly between, 1), mean success / finished / loop rates; `learnable_fit` counts the learnable tasks
    whose prompt fits max_prompt_tokens (what the RL trainer can use)."""
    out = {}
    for B in map(str, budgets):
        rows = [(tid, t["budgets"][B], t.get("n_prompt_tokens", 0)) for tid, t in tasks.items() if B in t["budgets"]]
        if not rows:
            continue
        n = len(rows)
        s = [r[1]["success"] for r in rows]
        out[B] = {"n_tasks": n, "mean_success": sum(s) / n,
                  "n_zero": sum(x == 0 for x in s), "n_learnable": sum(0 < x < 1 for x in s), "n_one": sum(x == 1 for x in s),
                  "n_any_correct": sum(x > 0 for x in s),
                  "mean_finished": sum(r[1]["finished"] for r in rows) / n,
                  "mean_loop": sum(r[1]["loop"] for r in rows) / n,
                  "mean_len": sum(r[1]["len_mean"] for r in rows) / n}
        if max_prompt_tokens:
            out[B]["n_learnable_fit"] = sum(0 < r[1]["success"] < 1 and r[2] <= max_prompt_tokens for r in rows)
    return out
