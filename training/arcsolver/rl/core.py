"""Pure-python pieces of the RL trainer: no torch, no vLLM.

Deterministic bookkeeping that is easy to get subtly wrong and cheap to unit-test:
* the prompt pool (generated "bank" tasks + ARC-AGI-2 training, never ARC-AGI-2 evaluation, seen/unseen tags),
* prompt rendering through the tokenizer's chat template,
* the paragraph-level loop detector, the epoch-shuffled prompt order (resumable), per-request seeds and the eval
  metric aggregation.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field

from arcsolver.prompt import arc_render as ar

ARM = "r2k_zb_max"
THINK_CLOSE = "</think>"
GEN_PROMPT_SUFFIX = "<|im_start|>assistant\n<think>\n"


# --- prompt pool ---------------------------------------------------------------------------------

@dataclass
class PromptItem:
    task_id: str
    source: str          # "bank" | "arc_train" (pool); "bank200" | "arc_eval" (eval set only)
    path: str
    seen: bool           # the task's traces were in the SFT data (--stage2-cache)
    n_prompt_tokens: int = 0
    student_ids: list = field(default_factory=list, repr=False)

    def tag(self) -> str:
        return "seen" if self.seen else "unseen"

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("student_ids")
        return d


def list_task_files(directory: str) -> dict:
    """task_id -> path for every *.json in a directory (task_id = file stem)."""
    return {os.path.basename(p)[:-5]: p for p in sorted(glob.glob(os.path.join(directory, "*.json")))}


def task_fingerprint(task: dict) -> str:
    """Content hash of what the prompt shows (train pairs + test inputs). Two files with the same
    fingerprint render the same prompt, whatever their names."""
    shown = {"train": [[p["input"], p["output"]] for p in task["train"]],
             "test": [p["input"] for p in task["test"]]}
    return hashlib.sha1(json.dumps(shown, separators=(",", ":")).encode()).hexdigest()


def trained_task_ids(stage2_cache_path: str) -> set:
    """Task ids whose reasoning traces were in the SFT data cache (a JSON {"examples": [{"prov": ...}]}).
    examples[].prov = "trace|cohort|task_id|arm|slot" (aux rows start with "aux|" and name no task)."""
    with open(stage2_cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    out = set()
    for ex in cache["examples"]:
        prov = ex.get("prov") or ""
        parts = prov.split("|")
        if parts[0] == "trace" and len(parts) >= 3:
            out.add(parts[2])
    return out


def _same_dir(a: str, b: str) -> bool:
    return os.path.realpath(a).rstrip("/") == os.path.realpath(b).rstrip("/")


def build_pool(bank_dir: str, arc_train_dir: str, arc_eval_dir: str, trained_ids: set,
               holdout_ids=(), load=None) -> tuple[list, dict]:
    """The prompt pool: generated bank tasks + ARC-AGI-2 training tasks.

    ARC-AGI-2 evaluation is excluded three ways: a source directory may not be the evaluation
    directory, no pool task id may be an evaluation task id, and no pool task may render the same
    prompt as an evaluation task (content fingerprint). `holdout_ids` (the periodic-eval bank tasks)
    are left out too, so the eval measures tasks RL never trained on.
    Returns (items sorted by (source, task_id), report dict)."""
    load = load or ar.load_task
    for d in (bank_dir, arc_train_dir):
        if _same_dir(d, arc_eval_dir):
            raise ValueError(f"prompt source {d} is the ARC-AGI-2 evaluation directory")
    eval_files = list_task_files(arc_eval_dir)
    if not eval_files:
        raise ValueError(f"no evaluation tasks found in {arc_eval_dir}; refusing to build a pool "
                         f"that cannot be checked against them")
    eval_fps = {task_fingerprint(load(p)): tid for tid, p in eval_files.items()}
    holdout = set(holdout_ids)
    items, report = [], {"dropped_eval_id": [], "dropped_eval_content": [], "held_out": [], "duplicates": [],
                         "dropped_holdout_content": []}
    seen_fp = {}
    # held-out tasks are excluded by content too, so a renamed copy cannot enter the pool
    holdout_fps = {}
    for directory in (bank_dir, arc_train_dir):
        for tid, path in list_task_files(directory).items():
            if tid in holdout:
                holdout_fps.setdefault(task_fingerprint(load(path)), tid)
    for source, directory in (("bank", bank_dir), ("arc_train", arc_train_dir)):
        for tid, path in list_task_files(directory).items():
            if tid in eval_files:
                report["dropped_eval_id"].append(tid)
                continue
            if tid in holdout:
                report["held_out"].append(tid)
                continue
            task = load(path)
            fp = task_fingerprint(task)
            if fp in eval_fps:
                report["dropped_eval_content"].append((tid, eval_fps[fp]))
                continue
            if fp in holdout_fps:
                report["dropped_holdout_content"].append((tid, holdout_fps[fp]))
                continue
            if fp in seen_fp:                    # same prompt under two names: keep the first
                report["duplicates"].append((tid, seen_fp[fp]))
                continue
            seen_fp[fp] = tid
            items.append(PromptItem(task_id=tid, source=source, path=path, seen=tid in trained_ids))
    report["n_items"] = len(items)
    report["n_by_source"] = {s: sum(1 for x in items if x.source == s) for s in ("bank", "arc_train")}
    report["n_seen"] = sum(1 for x in items if x.seen)
    return items, report


def assert_no_eval(items, arc_eval_dir: str, load=None) -> None:
    """Hard check before training starts (also used by the tests)."""
    load = load or ar.load_task
    eval_files = list_task_files(arc_eval_dir)
    eval_fps = {task_fingerprint(load(p)) for p in eval_files.values()}
    for it in items:
        if it.task_id in eval_files or _same_dir(os.path.dirname(it.path), arc_eval_dir):
            raise AssertionError(f"evaluation task {it.task_id} in the prompt pool")
        if task_fingerprint(load(it.path)) in eval_fps:
            raise AssertionError(f"pool task {it.task_id} renders an evaluation task's prompt")


# Optional text prefilled after the generation prompt, e.g. "<alpha>\n" for a tagged render style: set once per process
# with set_think_prefix(tokenizer, text); render_ids then appends its token ids for that tokenizer (other tokenizers are
# unaffected). Empty = off (the default; nothing changes).
_THINK_PREFIX = {"text": "", "ids": [], "tok": None}


def set_think_prefix(tokenizer, text: str) -> list:
    ids = tokenizer.encode(text, add_special_tokens=False) if text else []
    _THINK_PREFIX.update(text=text or "", ids=[int(x) for x in ids], tok=getattr(tokenizer, "name_or_path", None))
    return list(_THINK_PREFIX["ids"])


def think_prefix_ids(tokenizer=None) -> list:
    if not _THINK_PREFIX["ids"]:
        return []
    if tokenizer is not None and getattr(tokenizer, "name_or_path", None) != _THINK_PREFIX["tok"]:
        return []
    return list(_THINK_PREFIX["ids"])


# Optional chat-template override and reasoning effort: set once per process with set_chat_template(path, effort).
# render_ids passes both to apply_chat_template for every prompt, so the trainer, the sampler and the evals render
# identically.
_TEMPLATE = {"chat_template": None, "path": None, "reasoning_effort": None}


def set_chat_template(path: str | None = None, reasoning_effort: str | None = None) -> dict:
    if reasoning_effort is not None and reasoning_effort not in ("xhigh", "medium", "low"):
        raise ValueError(f"unknown reasoning effort {reasoning_effort!r}")
    text = None
    if path:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    _TEMPLATE.update(chat_template=text, path=path, reasoning_effort=reasoning_effort)
    return {"chat_template": path, "reasoning_effort": reasoning_effort}


def template_kwargs_for(**template_kwargs) -> dict:
    """The apply_chat_template kwargs render_ids uses (template override and effort, if set)."""
    if _TEMPLATE["chat_template"] is not None:
        template_kwargs.setdefault("chat_template", _TEMPLATE["chat_template"])
    if _TEMPLATE["reasoning_effort"] is not None:
        template_kwargs.setdefault("reasoning_effort", _TEMPLATE["reasoning_effort"])
    return template_kwargs


def render_ids(tokenizer, messages, **template_kwargs) -> list:
    """Chat template -> token ids with the generation prompt (as the eval does): thinking on, or off for the
    render_output arm (arc_render default channel "output"), plus the think prefix if one is set; the chat-template
    override and reasoning effort from set_chat_template, if any."""
    template_kwargs = template_kwargs_for(**template_kwargs)
    template_kwargs.setdefault("enable_thinking", ar.default_channel() == "think")
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **template_kwargs)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    return [int(x) for x in ids] + think_prefix_ids(tokenizer)


def student_messages(task, arm: str = ARM) -> list:
    return ar.build_messages(task, arm)


GEN_PROMPT_SUFFIX_OUTPUT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def check_generation_prompt(tokenizer, ids, who: str) -> None:
    want = GEN_PROMPT_SUFFIX if ar.default_channel() == "think" else GEN_PROMPT_SUFFIX_OUTPUT
    pre = think_prefix_ids(tokenizer)
    if pre:
        if list(ids[-len(pre):]) != pre:
            raise ValueError(f"{who} prompt must end with the think prefix {_THINK_PREFIX['text']!r}")
        ids = ids[:-len(pre)]
    tail = tokenizer.decode(ids[-12:])
    if not tail.endswith(want):
        raise ValueError(f"{who} prompt must end with {want!r}; ends with {tail!r}")


# --- loops -------------------------------------------------------------------------------------------

def _loop_onset(paras, key):
    seen, rep = set(), []
    for p in paras:
        k = key(p)
        rep.append(k in seen)
        seen.add(k)
    n = len(paras)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + rep[i]
    for i in range(n):
        if rep[i] and suffix[i] / (n - i) >= 0.8 and n - i >= 3:
            return i
    return None


def loop_kind(text: str) -> str:
    """Paragraph-level loop detector.
    The reasoning loops from paragraph i on if paragraph i repeats an earlier one and >= 80% of the
    paragraphs from i on (at least 3) are repeats. "verbatim" = exact repeats, "templated" =
    repeats after masking digits, "none" = no loop."""
    think = ar.reasoning_part(text)
    paras = [p.strip() for p in think.split("\n\n") if p.strip()]
    if _loop_onset(paras, lambda p: p) is not None:
        return "verbatim"
    if _loop_onset(paras, lambda p: re.sub(r"\d+", "#", p)) is not None:
        return "templated"
    return "none"


# --- prompt order --------------------------------------------------------------------------------

class PromptOrder:
    """Epoch-wise seeded shuffle over pool indices; resumable (epoch, cursor)."""

    def __init__(self, n: int, seed: int):
        if n <= 0:
            raise ValueError("empty prompt pool")
        self.n, self.seed, self.epoch, self.cursor = n, seed, 0, 0
        self._perm = self._make(0)

    def _make(self, epoch):
        perm = list(range(self.n))
        random.Random(self.seed * 1_000_003 + epoch).shuffle(perm)
        return perm

    def take(self, k: int) -> list:
        out = []
        while len(out) < k:
            if self.cursor >= self.n:
                self.epoch, self.cursor = self.epoch + 1, 0
                self._perm = self._make(self.epoch)
            out.append(self._perm[self.cursor])
            self.cursor += 1
        return out

    def state_dict(self) -> dict:
        return {"n": self.n, "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, s: dict) -> None:
        if s["n"] != self.n or s["seed"] != self.seed:
            raise ValueError(f"prompt order mismatch: checkpoint n={s['n']} seed={s['seed']}, "
                             f"now n={self.n} seed={self.seed} (did the pool change?)")
        self.epoch, self.cursor = s["epoch"], s["cursor"]
        self._perm = self._make(self.epoch)


def request_seed(seed: int, step: int, index: int, n: int) -> int:
    """Per-request vLLM seed; vLLM gives the n children seed + 0..n-1, so requests are spaced by n."""
    base = ((seed * 1_000_003 + step) * 10_007) % (2 ** 30)
    return base + index * n


# --- metrics -------------------------------------------------------------------------------------

def eval_group_metrics(scores: list, loops: list, lens: list, n: int, ks) -> dict:
    """scores = arc_render.score_task results of one eval group; loops/lens per sample."""
    out = {"n_tasks": len(scores)}
    if not scores:
        return out
    for k in ks:
        out[f"pass@{k}"] = sum(s[f"pass@{k}"] for s in scores) / len(scores)
    total = sum(s["n_samples"] for s in scores)
    out["closed_rate"] = sum(s["closed"] for s in scores) / total
    out["well_formed_rate"] = sum(s["well_formed"] for s in scores) / total
    out["loop_rate"] = sum(k != "none" for k in loops) / max(1, len(loops))
    out["len_mean"] = sum(lens) / max(1, len(lens))
    out["tasks_any_correct"] = sum(1 for s in scores if s.get(f"pass@{max(ks)}", 0) > 0)
    return out
