"""Run-directory helpers of the RL trainer: data paths, task cache, the periodic eval set, atomic JSON writers and the
checkpoint directory listing."""
from __future__ import annotations

import gzip
import json
import os
import random

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import core as oc

# Path flags. Every one is a CLI flag (--bank-dir, ...); the default comes from the environment variable in brackets,
# else it is empty (a missing required path fails at startup).
DEFAULTS = dict(
    student=os.environ.get("ARC_RL_STUDENT", ""),                  # HF model dir of the policy's start checkpoint
    bank_dir=os.environ.get("ARC_RL_BANK_DIR", ""),                # generated tasks (*.json), --pool bank / mixed
    arc_train_dir=os.environ.get("ARC_RL_ARC_TRAIN_DIR", ""),      # ARC-AGI-2 training tasks, --pool arc_train / mixed
    arc_eval_dir=os.environ.get("ARC_RL_ARC_EVAL_DIR", ""),        # ARC-AGI-2 evaluation tasks: never trained on
    bank200_dir=os.environ.get("ARC_RL_BANK_EVAL_DIR", ""),        # optional bank holdout: tasks/ + seen_split.json
    stage2_cache=os.environ.get("ARC_RL_SFT_CACHE", ""),           # optional SFT data cache: marks pool tasks "seen"
    vllm_python=os.environ.get("ARC_VLLM_PYTHON", "python"),       # interpreter of the vLLM environment
    vllm_env_script=os.environ.get("ARC_VLLM_ENV_SCRIPT", ""),     # optional shell file sourced before each engine
)


def write_json(path, obj):
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def write_jsonl_gz(path, rows):
    tmp = path + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class TaskCache:
    def __init__(self):
        self.d = {}

    def __call__(self, path):
        if path not in self.d:
            self.d[path] = ar.load_task(path)
        return self.d[path]


def build_eval_set(args, tok, tasks, log):
    """Fixed periodic-eval set, chosen once per run (stored in eval_set.json and reused on resume):
    eval_bank/2 SFT-seen + eval_bank/2 unseen tasks of the bank holdout (--bank200-dir), eval_arc ARC-AGI-2 evaluation
    tasks, all with prompts that leave --eval-max-tokens of room in the sampler's context. A source whose count is 0 or
    whose directory is not given contributes no tasks."""
    path = os.path.join(args.out, "eval_set.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        items = [oc.PromptItem(**{k: v for k, v in d.items() if k in ("task_id", "source", "path", "seen", "n_prompt_tokens")})
                 for d in saved["items"]]
        for it in items:
            it.student_ids = oc.render_ids(tok, oc.student_messages(tasks(it.path), args.arm))
        return items
    room = args.sampler_max_model_len - args.eval_max_tokens
    rng = random.Random(20260927)
    all_bank = getattr(args, "eval_bank_all", False)
    use_bank = bool(args.bank200_dir) and (args.eval_bank > 0 or all_bank)
    use_arc = bool(args.arc_eval_dir) and args.eval_arc > 0
    split = {"seen": [], "unseen": []}
    bank_files = {}
    if use_bank:
        with open(os.path.join(args.bank200_dir, "seen_split.json")) as f:
            split = json.load(f)
        bank_files = oc.list_task_files(os.path.join(args.bank200_dir, "tasks"))
    arc_files = oc.list_task_files(args.arc_eval_dir) if use_arc else {}
    items = []

    def pick(ids, k, source, seen_fn, files):
        ids = sorted(ids)
        rng.shuffle(ids)
        got = []
        for tid in ids:
            if len(got) >= k:
                break
            t = tasks(files[tid])
            sids = oc.render_ids(tok, oc.student_messages(t, args.arm))
            if len(sids) > room:
                continue
            it = oc.PromptItem(task_id=tid, source=source, path=files[tid], seen=seen_fn(tid), n_prompt_tokens=len(sids))
            it.student_ids = sids
            got.append(it)
        return got

    half = args.eval_bank // 2
    if use_bank:
        items += pick(split["seen"], len(split["seen"]) if all_bank else half, "bank200", lambda _: True, bank_files)
        items += pick(split["unseen"], len(split["unseen"]) if all_bank else args.eval_bank - half,
                      "bank200", lambda _: False, bank_files)
    if use_arc:
        items += pick(list(arc_files), args.eval_arc, "arc_eval", lambda _: False, arc_files)
    write_json(path, {"room_tokens": room, "items": [it.meta() for it in items]})
    log(f"eval set: {sum(i.source == 'bank200' and i.seen for i in items)} bank seen, "
        f"{sum(i.source == 'bank200' and not i.seen for i in items)} bank unseen, "
        f"{sum(i.source == 'arc_eval' for i in items)} ARC-AGI-2 eval (prompt <= {room} tokens)")
    return items


def ckpt_dirs(out):
    """The run's complete checkpoint directories (checkpoints/step_N with a COMPLETE marker), oldest first."""
    root = os.path.join(out, "checkpoints")
    if not os.path.isdir(root):
        return []
    ds = [os.path.join(root, d) for d in os.listdir(root) if d.startswith("step_") and d[5:].isdigit()]
    return sorted((d for d in ds if os.path.exists(os.path.join(d, "COMPLETE"))),
                  key=lambda d: int(os.path.basename(d)[5:]))
