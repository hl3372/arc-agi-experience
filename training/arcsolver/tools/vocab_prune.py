#!/usr/bin/env python3
"""Shrink a Qwen3.5-family vocabulary (embed_tokens + lm_head) to the tokens our task needs.

Qwen3.5 / Qwen3.8 checkpoints have 248,320 vocab rows: 248,044 byte-level BPE tokens, 33 added/special tokens and
243 padding rows with no token string. Our task (ARC grids, English reasoning, digits) uses a tiny fraction of them.
This tool keeps
  * all added/special tokens (<|im_start|>, <|im_end|>, <think>, </think>, vision/audio/tool tokens, ...),
  * all 256 single-byte tokens, so ANY text stays encodable (byte-level "fallback"),
  * every ASCII-only token,
  * every token that occurs when encoding a corpus (our prompts + the model's own reasoning traces),
  * the merge ancestors of every kept token (closure; makes the pruned BPE behave exactly like the old one),
renumbers the kept tokens 0..K-1 in old-id order, and rewrites tokenizer + checkpoint consistently.

Why tokenization is unchanged (exactly) for any text whose old encoding uses only kept tokens: every token is
produced by exactly one merge (checked), the keep-set is closed under "parts of the merge that makes a kept
token", and the new merge list is the old one restricted to merges whose parts and result are kept, in the
same rank order. Every merge the old BPE applies on such a text has kept parts and a kept result, so at each
step the old choice (lowest rank, leftmost) is still available and still the lowest among the new, smaller
candidate set. Texts that need a dropped token are encoded with smaller kept pieces (ultimately single
bytes) and decode back to the same string.

Library:
  build_keep_set(tokenizer_json, corpus_texts, extra_ids=())  -> sorted old ids
  iter_corpus_texts(...)                                       -> traces / ARC prompts / extra files
  prune_tokenizer(src_dir, dst_dir, keep)                      -> tokenizer.json, tokenizer_config.json, ...
  prune_checkpoint(src_dir, dst_dir, keep)                     -> streamed safetensors + configs + id_map.json
  check_tokenizer_equivalence(old, new, texts)                 -> counts + examples
  make_tiny_checkpoint(...), check_model_equivalence(...)      -> exact-logit check on CPU (or GPU)

CLI (python -m arcsolver.tools.vocab_prune <cmd> -h for options):
  keep-set           --tokenizer DIR --out keep.json [--extra-file F ...] [--keep-chars symbols] [--align N]
  prune              --src MODEL_DIR --dst OUT_DIR --keep keep.json    (tokenizer + checkpoint + other files)
  prune-tokenizer    --src DIR --dst DIR --keep keep.json
  prune-checkpoint   --src DIR --dst DIR --keep keep.json
  validate-tokenizer --src OLD_DIR --dst NEW_DIR [--max-traces N] [--json out.json]
  validate-model     --src OLD_CKPT --dst NEW_CKPT [--device cuda]     (sequential loads; exact logits)
  selftest-tiny      --tokenizer DIR --work DIR                        (tiny random model, real vocab size)

Only stdlib + numpy are needed for keep-set/pruning; `tokenizers` for encoding; torch + transformers only for
the model checks. The checkpoint rewrite is a raw byte-level safetensors stream (any dtype, FP8 included),
shard by shard, so memory stays at one copy buffer.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator

import numpy as np

# Corpus defaults: no trace bank and no task dirs (pass --trace-bank / --task-dir / --extra-file); the release's
# prompt arm; every prompt file shipped next to arcsolver/prompt/arc_render.py, added verbatim.
DEFAULT_TRACE_BANK = ""
DEFAULT_TRACE_MODELS = ()  # empty = every model in the trace bank's index
DEFAULT_TASK_DIRS = ()
DEFAULT_ARMS = ("r2k_zb_max",)  # arcsolver.prompt.arc_render arms to render


def _default_prompt_dir() -> str:
    try:
        from arcsolver.prompt import arc_render
    except ImportError:
        return ""
    return os.path.join(os.path.dirname(os.path.abspath(arc_render.__file__)), "prompts")


DEFAULT_PROMPT_DIR = _default_prompt_dir()

# Optional extension (--keep-chars symbols): every token whose text is ASCII plus Latin-1 symbols (° ± ² × ÷ ...),
# General Punctuation, super/subscripts, arrows and math operators (e.g. " ≥", "—", "²", "→", "°C").
SYMBOL_RANGES = ((0x00A0, 0x00BF), (0x00D7, 0x00D7), (0x00F7, 0x00F7), (0x2000, 0x206F), (0x2070, 0x209F),
                 (0x2190, 0x21FF), (0x2200, 0x22FF))

ID_MAP_FORMAT = "qwen38-vocab-prune/1"
KEEP_FORMAT = "qwen38-vocab-keep/1"

# --------------------------------------------------------------------------------------------------------------
# GPT-2 byte-level alphabet


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte -> printable-char table used by byte-level BPE (Qwen's tokenizer.json strings)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}


def token_bytes(token: str) -> bytes:
    """Raw bytes of a byte-level BPE token string."""
    return bytes(BYTE_DECODER[c] for c in token)


def token_display(token: str) -> str:
    """Readable form of a token (partial UTF-8 shown as \\x escapes)."""
    try:
        return token_bytes(token).decode("utf-8", errors="backslashreplace")
    except KeyError:  # an added token: not byte-level encoded
        return token


def _is_ascii_bytes(b: bytes) -> bool:
    return all(x < 128 for x in b)


def parse_char_ranges(spec: str) -> tuple[tuple[int, int], ...]:
    """'symbols' or 'symbols,0370-03FF,00B0' -> ((lo, hi), ...) of code points (hex)."""
    out = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        if part == "symbols":
            out += SYMBOL_RANGES
        else:
            lo, _, hi = part.partition("-")
            out.append((int(lo, 16), int(hi or lo, 16)))
    return tuple(out)


# --------------------------------------------------------------------------------------------------------------
# tokenizer.json


def _tokenizer_json(path: str) -> str:
    return os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TokInfo:
    """The parts of a byte-level BPE tokenizer.json the pruner needs."""

    def __init__(self, path: str):
        self.path = _tokenizer_json(path)
        with open(self.path, encoding="utf-8") as f:
            self.data = json.load(f)
        model = self.data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"{self.path}: expected a BPE model, got {model.get('type')}")
        self.vocab: dict[str, int] = model["vocab"]
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        if len(self.id_to_token) != len(self.vocab):
            raise ValueError("duplicate ids in model.vocab")
        self.merges_are_strings = bool(model["merges"]) and isinstance(model["merges"][0], str)
        self.merges: list[tuple[str, str]] = []
        for m in model["merges"]:
            pair = tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
            if len(pair) != 2:
                raise ValueError(f"unparseable merge {m!r}")
            self.merges.append(pair)
        self.added = {t["id"]: t["content"] for t in self.data.get("added_tokens", [])}
        self.n_ids = max(max(self.id_to_token), max(self.added, default=-1)) + 1
        # result id -> (left id, right id): the merge that creates each token
        self.parents: dict[int, tuple[int, int]] = {}
        for a, b in self.merges:
            r = self.vocab.get(a + b)
            ia, ib = self.vocab.get(a), self.vocab.get(b)
            if r is None or ia is None or ib is None:
                raise ValueError(f"merge {a!r} {b!r} references a token missing from the vocab")
            if r in self.parents:  # would break the closure argument; does not happen for Qwen
                raise ValueError(f"token {a + b!r} is produced by more than one merge")
            self.parents[r] = (ia, ib)

    def tokenizer(self):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(self.path)

    def single_byte_ids(self) -> set[int]:
        return {self.vocab[c] for c in BYTE_ENCODER.values()}

    def policy_sets(self) -> dict[str, set[int]]:
        """The corpus-independent part of the keep policy."""
        ascii_ids = {i for t, i in self.vocab.items() if _is_ascii_bytes(token_bytes(t))}
        return {"added": set(self.added), "single_byte": self.single_byte_ids(), "ascii_only": ascii_ids}

    def ids_within_chars(self, ranges) -> set[int]:
        """Tokens whose bytes are complete UTF-8 made only of ASCII and code points inside `ranges`."""
        out = set()
        for t, i in self.vocab.items():
            try:
                text = token_bytes(t).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not text.isascii() and all(ord(c) < 128 or any(lo <= ord(c) <= hi for lo, hi in ranges) for c in text):
                out.add(i)
        return out

    def close(self, keep: set[int]) -> int:
        """Add the merge ancestors of every kept token. Returns how many ids were added."""
        before = len(keep)
        stack = [i for i in keep if i in self.parents]
        while stack:
            for p in self.parents[stack.pop()]:
                if p not in keep:
                    keep.add(p)
                    if p in self.parents:
                        stack.append(p)
        return len(keep) - before

    def align(self, keep: set[int], multiple: int) -> int:
        """Grow `keep` to a multiple of `multiple` with real tokens (lowest unkept BPE ids whose merge parts are
        already kept, i.e. the most frequent ones; closure is preserved). Returns how many were added."""
        need = (-len(keep)) % multiple
        added = 0
        for i in sorted(self.id_to_token):
            if added == need:
                break
            if i in keep:
                continue
            par = self.parents.get(i)
            if par is None or par[0] not in keep or par[1] not in keep:
                continue  # unreachable (no merge) or would need more ancestors
            keep.add(i)
            added += 1
        if added != need:
            raise RuntimeError(f"could not align the keep-set to a multiple of {multiple}")
        return added

    def check_keep(self, keep: Iterable[int]) -> list[int]:
        """Validate a keep list against this tokenizer; returns it sorted. Raises ValueError on violations."""
        keep = sorted({int(i) for i in keep})
        ks = set(keep)
        if keep and (keep[0] < 0 or keep[-1] >= self.n_ids):
            raise ValueError(f"keep ids must be real tokens in [0, {self.n_ids}); got {keep[0]}..{keep[-1]}")
        holes = [i for i in keep if i not in self.id_to_token and i not in self.added]
        if holes:
            raise ValueError(f"{len(holes)} keep ids have no token string, e.g. {holes[:5]}")
        missing = sorted(set(self.added) - ks)
        if missing:
            raise ValueError(f"added/special tokens must be kept; missing {[self.added[i] for i in missing[:8]]}")
        missing = sorted(self.single_byte_ids() - ks)
        if missing:
            raise ValueError(f"all 256 single-byte tokens must be kept (byte fallback); {len(missing)} missing")
        open_ = [i for i in keep if i in self.parents and not set(self.parents[i]) <= ks]
        if open_:
            ex = [token_display(self.id_to_token[i]) for i in open_[:5]]
            raise ValueError(f"keep-set is not closed under merges ({len(open_)} kept tokens have dropped parts, "
                             f"e.g. {ex}); build it with build_keep_set")
        return keep


def make_maps(keep: list[int], n_old: int) -> tuple[np.ndarray, np.ndarray]:
    """old_to_new (int64[n_old], -1 = dropped) and new_to_old (int64[K])."""
    new_to_old = np.asarray(keep, dtype=np.int64)
    old_to_new = np.full(n_old, -1, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(len(keep), dtype=np.int64)
    return old_to_new, new_to_old


def write_id_map(dst_dir: str, keep: list[int], n_old: int, **extra) -> str:
    old_to_new, _ = make_maps(keep, n_old)
    path = os.path.join(dst_dir, "id_map.json")
    doc = {"format": ID_MAP_FORMAT, "old_vocab_size": int(n_old), "new_vocab_size": len(keep),
           "new_to_old": [int(i) for i in keep], "old_to_new": old_to_new.tolist(), **extra}
    _write_json(path, doc, indent=None)
    return path


def load_keep(path: str) -> list[int]:
    """keep.json ({"keep": [...]}), id_map.json ({"new_to_old": [...]}) or a bare JSON list."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        return [int(i) for i in doc]
    for key in ("keep", "new_to_old"):
        if key in doc:
            return [int(i) for i in doc[key]]
    raise ValueError(f"{path}: no 'keep' or 'new_to_old' list")


def _write_json(path: str, obj, indent=2) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------------------------
# corpus


def _strings_in(obj) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings_in(v)


def read_text_file(path: str) -> Iterator[str]:
    """Texts from a file or directory (recursively). .jsonl: every string in every line's JSON value;
    .json: every string in the document; anything else: the whole file as one text (UTF-8)."""
    if os.path.isdir(path):
        for root, _, files in sorted(os.walk(path)):
            for name in sorted(files):
                yield from read_text_file(os.path.join(root, name))
        return
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield from _strings_in(json.loads(line))
    elif path.endswith(".json"):
        yield from _strings_in(_read_json(path))
    else:
        with open(path, encoding="utf-8") as f:
            yield f.read()


def _request_texts(path: str) -> Iterator[str]:
    req = _read_json(path)
    for msg in req.get("messages", []):
        yield from _strings_in(msg.get("content"))


def iter_trace_texts(trace_bank: str = DEFAULT_TRACE_BANK, models: Iterable[str] = DEFAULT_TRACE_MODELS, *,
                     include_requests: bool = True, max_traces: int | None = None, seed: int = 0) -> Iterator[str]:
    """reasoning.txt + answer.txt (+ the request's prompt messages) of the given models' traces (all models when
    `models` is empty). Trace bank layout: <trace_bank>/index.jsonl with one {"trace_id", "model", "path"} per trace,
    and <trace_bank>/<path>/{reasoning.txt, answer.txt, request.json ({"messages": [...]})}.
    max_traces: a seeded random subset (None = all)."""
    models = set(models)
    with open(os.path.join(trace_bank, "index.jsonl"), encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if not models or r["model"] in models]
    rows.sort(key=lambda r: r["trace_id"])
    if max_traces is not None and max_traces < len(rows):
        rows = random.Random(seed).sample(rows, max_traces)
    for r in rows:
        d = os.path.join(trace_bank, r["path"])
        for name in ("reasoning.txt", "answer.txt"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as fh:
                    yield fh.read()
        if include_requests and os.path.exists(os.path.join(d, "request.json")):
            yield from _request_texts(os.path.join(d, "request.json"))


def _arc_render():
    from arcsolver.prompt import arc_render  # the exact prompt renderer (pure python)
    return arc_render


def iter_prompt_texts(task_dirs: Iterable[str] = DEFAULT_TASK_DIRS, arms: Iterable[str] = DEFAULT_ARMS, *,
                      prompt_dir: str | None = DEFAULT_PROMPT_DIR, max_tasks: int | None = None,
                      seed: int = 0) -> Iterator[str]:
    """ARC prompts as our pipeline renders them (system prompt once per arm, then each task's user message),
    plus every file in prompt_dir. max_tasks: a seeded random subset per directory."""
    if prompt_dir and os.path.isdir(prompt_dir):
        for p in sorted(glob.glob(os.path.join(prompt_dir, "*"))):
            if os.path.isfile(p):
                yield from read_text_file(p)
    arms = list(arms)
    if not arms:
        return
    ar = _arc_render()
    for arm in arms:
        yield ar.system_prompt(arm)
    for d in task_dirs:
        paths = sorted(glob.glob(os.path.join(d, "*.json")))
        if max_tasks is not None and max_tasks < len(paths):
            paths = random.Random(seed).sample(paths, max_tasks)
        for p in paths:
            task = ar.load_task(p)
            for arm in arms:
                for msg in ar.build_messages(task, arm)[1:]:  # system prompt already yielded
                    yield msg["content"]


def iter_corpus_texts(trace_bank: str | None = DEFAULT_TRACE_BANK, models: Iterable[str] = DEFAULT_TRACE_MODELS,
                      task_dirs: Iterable[str] = DEFAULT_TASK_DIRS, arms: Iterable[str] = DEFAULT_ARMS,
                      extra_files: Iterable[str] = (), *, include_requests: bool = True,
                      prompt_dir: str | None = DEFAULT_PROMPT_DIR, max_traces: int | None = None,
                      max_tasks: int | None = None, seed: int = 0) -> Iterator[str]:
    """Our prompts + the model's own reasoning traces + any extra text files (e.g. the pipeline's exact prompts)."""
    if trace_bank:
        yield from iter_trace_texts(trace_bank, models, include_requests=include_requests,
                                    max_traces=max_traces, seed=seed)
    yield from iter_prompt_texts(task_dirs, arms, prompt_dir=prompt_dir, max_tasks=max_tasks, seed=seed)
    for path in extra_files:
        yield from read_text_file(path)


def _batches(it: Iterable, n: int) -> Iterator[list]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_batch(tok, texts: list[str]) -> list[list[int]]:
    fn = getattr(tok, "encode_batch_fast", None) or tok.encode_batch
    return [e.ids for e in fn(texts, add_special_tokens=False)]


def corpus_token_counts(tok, texts: Iterable[str], n_ids: int, *, skip_ascii: bool = True,
                        batch_size: int = 64, stats: dict | None = None) -> np.ndarray:
    """Occurrence count of every token id when encoding `texts`.

    skip_ascii: pure-ASCII texts are not encoded (exact shortcut: a BPE token's bytes are a substring of the
    text, so such texts only produce ASCII-only tokens, which the policy keeps anyway). Their tokens are then
    not counted."""
    counts = np.zeros(n_ids, dtype=np.int64)
    st = {"texts": 0, "ascii_texts_skipped": 0, "texts_encoded": 0, "chars_encoded": 0, "tokens_encoded": 0}

    def todo():
        for t in texts:
            st["texts"] += 1
            if skip_ascii and t.isascii():
                st["ascii_texts_skipped"] += 1
                continue
            yield t

    for batch in _batches(todo(), batch_size):
        for ids in _encode_batch(tok, batch):
            arr = np.asarray(ids, dtype=np.int64)
            counts += np.bincount(arr, minlength=n_ids)[:n_ids]
            st["tokens_encoded"] += len(arr)
        st["texts_encoded"] += len(batch)
        st["chars_encoded"] += sum(map(len, batch))
    if stats is not None:
        stats.update(st)
    return counts


# --------------------------------------------------------------------------------------------------------------
# keep-set


def build_keep_set(tokenizer_json_path: str, corpus_texts: Iterable[str], extra_ids: Iterable[int] = (), *,
                   char_ranges=(), align: int = 0, stats: dict | None = None, batch_size: int = 64) -> list[int]:
    """Sorted old ids to keep: all added/special tokens, all 256 single-byte tokens, every ASCII-only token,
    every token produced by encoding `corpus_texts`, `extra_ids`, and the merge ancestors of all of these.
    char_ranges: also keep every token made of ASCII + these code-point ranges (e.g. SYMBOL_RANGES).
    align > 1: grow K to a multiple of `align` with the most frequent remaining real tokens (no padding rows).
    stats: filled with counts and the corpus' non-ASCII tokens."""
    info = TokInfo(tokenizer_json_path)
    sets = info.policy_sets()
    keep: set[int] = set().union(*sets.values())
    base = set(keep)
    cstats: dict = {}
    counts = corpus_token_counts(info.tokenizer(), corpus_texts, info.n_ids, batch_size=batch_size, stats=cstats)
    corpus = set(np.flatnonzero(counts).tolist())
    keep |= corpus
    extra = {int(i) for i in extra_ids}
    bad = sorted(i for i in extra if i not in info.id_to_token and i not in info.added)
    if bad:
        raise ValueError(f"extra_ids without a token string (padding rows or out of range): {bad[:10]}")
    keep |= extra
    by_chars = info.ids_within_chars(char_ranges) if char_ranges else set()
    n_chars = len(by_chars - keep)
    keep |= by_chars
    n_closure = info.close(keep)
    n_align = info.align(keep, align) if align and align > 1 else 0
    keep_sorted = info.check_keep(keep)
    if stats is not None:
        beyond = sorted(corpus - base, key=lambda i: -counts[i])
        stats.update({
            "tokenizer": info.path, "tokenizer_sha256": _sha256(info.path),
            "tokenizer_ids": info.n_ids, "bpe_tokens": len(info.vocab), "added_tokens": len(sets["added"]),
            "single_byte_tokens": len(sets["single_byte"]), "ascii_only_tokens": len(sets["ascii_only"]),
            "policy_base": len(base), "corpus": cstats, "corpus_distinct_ids": len(corpus),
            "corpus_ids_beyond_base": len(beyond), "extra_ids_beyond_base": len(extra - base - corpus),
            "char_ranges": [f"{lo:04X}-{hi:04X}" for lo, hi in char_ranges], "char_range_tokens_added": n_chars,
            "closure_added": n_closure, "align": align, "align_added": n_align, "K": len(keep_sorted),
            "corpus_nonascii_token_occurrences": int(counts[beyond].sum()) if beyond else 0,
            "corpus_tokens_beyond_base": [
                {"id": i, "token": token_display(info.id_to_token.get(i, info.added.get(i, ""))),
                 "count": int(counts[i])} for i in beyond],
        })
    return keep_sorted


# --------------------------------------------------------------------------------------------------------------
# tokenizer pruning

TOKENIZER_COPY_UNCHANGED = ("chat_template.jinja", "chat_template.json", "special_tokens_map.json")


def _remap_post_processor(node, old2new: dict[int, int]):
    """Remap ids inside a tokenizers post_processor (TemplateProcessing / Bert / Roberta / Sequence)."""
    if not isinstance(node, dict):
        return node
    kind = node.get("type")
    if kind == "TemplateProcessing":
        for spec in node.get("special_tokens", {}).values():
            spec["ids"] = [old2new[i] for i in spec["ids"]]
    elif kind in ("BertProcessing", "RobertaProcessing"):
        for key in ("sep", "cls"):
            tok, i = node[key]
            node[key] = [tok, old2new[i]]
    elif kind == "Sequence":
        for p in node.get("processors", []):
            _remap_post_processor(p, old2new)
    return node


ENGLISH_SAMPLE = [
    "The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX! the quick brown fox?",
    "Solve this ARC puzzle. Each training example shows an input grid and the output grid that one hidden "
    "transformation produces from it. Work out the transformation from all of the training examples, then "
    "apply it to each test input.",
    "Training example 1\nInput (3 rows, 5 columns):\n00120\n33333\n90807\n\nOutput (2 rows, 2 columns):\n12\n34",
    "r0 | 9999999999999999999\nr1 | 9825555455555553589\nr1: 9@c0 8@c1 2@c2 5@c3-c6 4@c7 5@c8-c14",
    "Wait, let's re-check: row 3, col 7 is 5 (not 4). So the rule holds for (3,7) -> (7,3). Hmm... 12x12 => 144 cells.",
    "def f(grid):\n    return [[c if c != 0 else 7 for c in row] for row in grid]  # it's 100% ok\n\tx += 1\r\n",
    "We'll, they're, I'm, you've, he'd, she'll, it's, ISN'T, Won't, y'all",
    "  leading spaces,trailing spaces   \n\n\n  \t\t mixed\twhitespace \r\n\r\n end",
    "".join(chr(c) for c in range(32, 127)),
    "0123456789 " * 8 + "\n" + "\n".join("".join(str((r * 7 + c * 3) % 10) for c in range(30)) for r in range(30)),
    "Final answer:\n<think>\nabc\n</think>\n\n0101\n1010",
]


def random_ascii_texts(n: int, seed: int = 0) -> list[str]:
    """Random ASCII strings of four kinds: raw printable noise, English-like words, digit grids, code/math."""
    rng = random.Random(seed)
    printable = [chr(c) for c in range(32, 127)] + ["\n", "\t", "\r", "  ", "\n\n"]
    words = ("the a an grid row column cell color pattern rule input output example test training shape object "
             "left right top bottom each every same different count width height blue red zero one two three "
             "rotate flip mirror copy fill border background foreground let's we're check so wait hmm actually "
             "therefore because if then else cannot isn't").split()
    seps = [" ", " ", " ", "  ", "\n", ", ", ". ", ": ", "; ", "\t", " - ", "\n\n", "? ", "! "]
    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            s = "".join(rng.choice(printable) for _ in range(rng.randint(1, 300)))
        elif kind == 1:
            parts = []
            for _ in range(rng.randint(1, 80)):
                w = rng.choice(words)
                w = w.upper() if rng.random() < 0.05 else (w.capitalize() if rng.random() < 0.15 else w)
                parts.append(w + rng.choice(seps))
            s = "".join(parts)
        elif kind == 2:
            h, w = rng.randint(1, 30), rng.randint(1, 30)
            label = rng.random() < 0.3
            rows = []
            for r in range(h):
                row = "".join(str(rng.randint(0, 9)) for _ in range(w))
                rows.append(f"r{r} | {row}" if label else (" ".join(row) if rng.random() < 0.2 else row))
            s = "\n".join(rows)
        else:
            a, b = rng.randint(0, 10 ** rng.randint(1, 6)), rng.randint(0, 999)
            s = rng.choice([f"x = {a} * (y + {b})", f"grid[{b % 30}][{a % 30}] == {a % 10}", f"({a},{b})->({b},{a})",
                            f"{a}/{b} = {a / max(b, 1):.4f}", f"if cell=={b % 10}: count+={a}", f"{a:,} and {b:08d}"])
            s = s * rng.randint(1, 5)
        out.append(s)
    return out


NONASCII_SAMPLES = [
    "naïve café — résumé, façade, Ångström, Zürich, São Paulo, Kraków",
    "3 × 4 = 12; 7 → 8; a ≤ b ≥ c ≠ d ≈ e; ±1; x² + y³; ½ ¼ ¾; 90°; π ≈ 3.14159; √2; ∑ ∏ ∫; ∀x ∃y",
    "“double” ‘single’ «guillemets» „low“ … – — • · ′ ″ ‰ € £ ¥ © ® ™ §",
    "← ↑ → ↓ ↔ ⇒ ⇐ ⇔ ↺ ↻ ⟶",
    "■ □ ▪ ▫ ▲ ▼ ◆ ◇ ○ ● █ ▓ ▒ ░ ┌─┐ │ └─┘ ├ ┤ ┬ ┴ ┼ ✓ ✗ ★ ☆",
    "😀 🎉 👍🏽 👨‍👩‍👧 🟥🟧🟨🟩🟦🟪⬛⬜ 🔴🔵",
    "中文测试：你好，世界！网格的颜色。",
    "日本語のテキスト、カタカナとひらがな。",
    "한국어 텍스트입니다.",
    "Русский текст, Ελληνικά γράμματα, العربية, עברית, हिन्दी, ไทย",
    "é vs é; Å vs Å (NFC); ​ zero width; ﻿ BOM;   nbsp;   thin space",
    "ﬁ ligature, Ⅻ, ①②③, ｆｕｌｌｗｉｄｔｈ, 𝔸𝕓𝕔, 𝟘𝟙𝟚",
    "Row 3: 0 0 → 1 1 (× 2); the cell at (2,3) becomes 5 ⇒ done ✓",
    "\x00\x01\x7f control chars, \u2028 line sep, \u0085 NEL, \U0010FFFD plane-16 private use",
]


def random_unicode_texts(n: int, seed: int = 0) -> list[str]:
    """Random strings mixing ASCII words with code points from many scripts/symbol blocks."""
    rng = random.Random(seed)
    ranges = [(0x80, 0xFF), (0x100, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F),
              (0x2000, 0x206F), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2500, 0x25FF), (0x2600, 0x27BF),
              (0x3040, 0x30FF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0xFF00, 0xFFEF), (0x1F300, 0x1FAFF),
              (0x1D400, 0x1D7FF), (0x10000, 0x1007F), (0xE000, 0xE0FF)]
    words = "the grid cell row color rule output input 0 1 2 3 4 5 6 7 8 9".split()
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 40)):
            if rng.random() < 0.5:
                parts.append(rng.choice(words) + rng.choice([" ", "", "\n", ", "]))
            else:
                lo, hi = rng.choice(ranges)
                parts.append("".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 6))))
        out.append("".join(parts))
    return out


def _load_tokenizers_tokenizer(path_or_tok):
    if isinstance(path_or_tok, str):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(_tokenizer_json(path_or_tok))
    return path_or_tok


def _string_id_map(old_tok, new_tok) -> np.ndarray:
    """new id -> old id by matching token strings (independent of id_map.json)."""
    k = new_tok.get_vocab_size(with_added_tokens=True)
    out = np.empty(k, dtype=np.int64)
    for n in range(k):
        s = new_tok.id_to_token(n)
        o = old_tok.token_to_id(s) if s is not None else None
        if o is None:
            raise ValueError(f"new id {n} ({s!r}) has no counterpart in the old tokenizer")
        out[n] = o
    return out


def check_tokenizer_equivalence(old, new, texts: Iterable[str], *, max_examples: int = 5,
                                batch_size: int = 64) -> dict:
    """Encode every text with the old and the new tokenizer.
    * all old ids kept     -> new ids must equal the mapped old ids (counted as equal / mismatch);
    * some old id dropped  -> decode(new) must equal decode(old) (round trip; counted as roundtrip_ok / _fail),
                              and also the text itself up to NFC normalization (the normalizer both share).
    Also checks every new id's token string and special flag against the old tokenizer."""
    old_tok, new_tok = _load_tokenizers_tokenizer(old), _load_tokenizers_tokenizer(new)
    new_to_old = _string_id_map(old_tok, new_tok)
    n_old = max(old_tok.get_vocab_size(with_added_tokens=True), int(new_to_old.max()) + 1)
    old_to_new = np.full(n_old, -1, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(len(new_to_old))
    r = {"texts": 0, "chars": 0, "old_tokens": 0, "new_tokens": 0,
         "all_kept": {"texts": 0, "equal": 0, "mismatch": 0, "examples": []},
         "with_dropped": {"texts": 0, "roundtrip_ok": 0, "roundtrip_fail": 0, "not_nfc_identity": 0,
                          "old_tokens": 0, "new_tokens": 0, "dropped_occurrences": 0, "examples": [],
                          "top_dropped": []}}
    dropped = Counter()
    for batch in _batches(texts, batch_size):
        olds, news = _encode_batch(old_tok, batch), _encode_batch(new_tok, batch)
        for t, o, n in zip(batch, olds, news):
            o, n = np.asarray(o, dtype=np.int64), np.asarray(n, dtype=np.int64)
            r["texts"] += 1
            r["chars"] += len(t)
            r["old_tokens"] += len(o)
            r["new_tokens"] += len(n)
            mapped = old_to_new[o] if len(o) else o
            if (mapped >= 0).all():
                a = r["all_kept"]
                a["texts"] += 1
                if np.array_equal(mapped, n):
                    a["equal"] += 1
                else:
                    a["mismatch"] += 1
                    if len(a["examples"]) < max_examples:
                        m = min(len(mapped), len(n))
                        pos = int(np.flatnonzero(mapped[:m] != n[:m])[0]) if (mapped[:m] != n[:m]).any() else m
                        a["examples"].append({"text": t[:300], "first_diff": pos,
                                              "old": [old_tok.id_to_token(int(i)) for i in o[pos:pos + 8]],
                                              "new": [new_tok.id_to_token(int(i)) for i in n[pos:pos + 8]]})
                continue
            w = r["with_dropped"]
            w["texts"] += 1
            w["old_tokens"] += len(o)
            w["new_tokens"] += len(n)
            drop = o[mapped < 0]
            w["dropped_occurrences"] += len(drop)
            dropped.update(drop.tolist())
            dn = new_tok.decode(n.tolist(), skip_special_tokens=False)
            do = old_tok.decode(o.tolist(), skip_special_tokens=False)
            if dn == do:
                w["roundtrip_ok"] += 1
            else:
                w["roundtrip_fail"] += 1
                if len(w["examples"]) < max_examples:
                    w["examples"].append({"text": t[:300], "old_decoded": do[:300], "new_decoded": dn[:300]})
            if dn != unicodedata.normalize("NFC", t):
                w["not_nfc_identity"] += 1
    r["with_dropped"]["top_dropped"] = [{"id": i, "token": token_display(old_tok.id_to_token(i) or ""), "count": c}
                                       for i, c in dropped.most_common(20)]
    # vocabulary-level checks
    bad_str, bad_special = 0, 0
    old_added = {t.content: t.special for t in old_tok.get_added_tokens_decoder().values()}
    for n_id, at in new_tok.get_added_tokens_decoder().items():
        if old_added.get(at.content) != at.special:
            bad_special += 1
    for n_id, o_id in enumerate(new_to_old.tolist()):
        if new_tok.id_to_token(n_id) != old_tok.id_to_token(o_id):
            bad_str += 1
    r["vocab"] = {"K": int(len(new_to_old)), "token_string_mismatch": bad_str, "special_flag_mismatch": bad_special,
                  "added_tokens_old": len(old_added), "added_tokens_new": len(new_tok.get_added_tokens_decoder())}
    return r


def ascii_selfcheck(old, new, n_random: int = 400, seed: int = 0) -> dict:
    """ASCII/English tokenization must be unchanged: English/grid/code samples, random ASCII strings, and the
    text of every ASCII-only kept token. Returns the equivalence report (all texts land in 'all_kept')."""
    old_tok, new_tok = _load_tokenizers_tokenizer(old), _load_tokenizers_tokenizer(new)
    tok_texts = []
    for n in range(new_tok.get_vocab_size(with_added_tokens=True)):
        s = new_tok.id_to_token(n)
        if s is None:
            continue
        try:
            b = token_bytes(s)
        except KeyError:
            continue  # added token
        if _is_ascii_bytes(b):
            tok_texts.append(b.decode("ascii"))
    texts = ENGLISH_SAMPLE + random_ascii_texts(n_random, seed) + tok_texts
    return check_tokenizer_equivalence(old_tok, new_tok, texts)


def prune_tokenizer(src_dir: str, dst_dir: str, keep: Iterable[int], *, check: bool = True) -> dict:
    """Write the pruned tokenizer files into dst_dir:
    tokenizer.json (vocab renumbered 0..K-1 in old-id order, merges restricted to kept parts+result, added tokens
    and post-processor ids remapped), tokenizer_config.json (added_tokens_decoder keys remapped), vocab.json /
    merges.txt / added_tokens.json if present, chat template and special_tokens_map copied unchanged.
    check: verify ASCII/English tokenization is unchanged (raises RuntimeError otherwise)."""
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        raise ValueError("dst_dir must differ from src_dir")
    os.makedirs(dst_dir, exist_ok=True)
    info = TokInfo(src_dir)
    keep = info.check_keep(keep)
    ks = set(keep)
    old2new = {o: n for n, o in enumerate(keep)}

    data = copy.deepcopy(info.data)
    model = data["model"]
    model["vocab"] = {info.id_to_token[o]: n for n, o in enumerate(keep) if o in info.id_to_token}
    kept_merges = [(a, b) for a, b in info.merges
                   if info.vocab[a] in ks and info.vocab[b] in ks and info.vocab[a + b] in ks]
    model["merges"] = [f"{a} {b}" for a, b in kept_merges] if info.merges_are_strings else [[a, b] for a, b in kept_merges]
    for t in data.get("added_tokens", []):
        t["id"] = old2new[t["id"]]
    data["post_processor"] = _remap_post_processor(data.get("post_processor"), old2new)
    if isinstance(data.get("padding"), dict) and "pad_id" in data["padding"]:
        data["padding"]["pad_id"] = old2new[data["padding"]["pad_id"]]
    dst_json = os.path.join(dst_dir, "tokenizer.json")
    _write_json(dst_json, data)

    written = ["tokenizer.json"]
    cfg_path = os.path.join(src_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        cfg = _read_json(cfg_path)
        if "added_tokens_decoder" in cfg:
            cfg["added_tokens_decoder"] = {str(old2new[int(k)]): v for k, v in cfg["added_tokens_decoder"].items()}
        _write_json(os.path.join(dst_dir, "tokenizer_config.json"), cfg)
        written.append("tokenizer_config.json")
    if os.path.exists(os.path.join(src_dir, "vocab.json")):
        _write_json(os.path.join(dst_dir, "vocab.json"), model["vocab"], indent=None)
        written.append("vocab.json")
    if os.path.exists(os.path.join(src_dir, "merges.txt")):
        with open(os.path.join(src_dir, "merges.txt"), encoding="utf-8") as f:
            first = f.readline()
        with open(os.path.join(dst_dir, "merges.txt"), "w", encoding="utf-8") as f:
            if first.startswith("#version"):
                f.write(first if first.endswith("\n") else first + "\n")
            for a, b in kept_merges:
                f.write(f"{a} {b}\n")
        written.append("merges.txt")
    if os.path.exists(os.path.join(src_dir, "added_tokens.json")):
        added = _read_json(os.path.join(src_dir, "added_tokens.json"))
        _write_json(os.path.join(dst_dir, "added_tokens.json"), {k: old2new[v] for k, v in added.items()})
        written.append("added_tokens.json")
    for name in TOKENIZER_COPY_UNCHANGED:
        if os.path.exists(os.path.join(src_dir, name)):
            shutil.copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))
            written.append(name)
    id_map = os.path.join(dst_dir, "id_map.json")  # prune_checkpoint rewrites it with the checkpoint's row count
    if not os.path.exists(id_map) or _read_json(id_map).get("new_to_old") != keep:
        write_id_map(dst_dir, keep, info.n_ids)
        written.append("id_map.json")

    summary = {"K": len(keep), "old_tokens": info.n_ids, "bpe_kept": len(model["vocab"]),
               "merges_old": len(info.merges), "merges_kept": len(kept_merges), "written": written}
    if check:
        rep = ascii_selfcheck(info.path, dst_json)
        a = rep["all_kept"]
        if a["mismatch"] or rep["with_dropped"]["texts"] or rep["vocab"]["token_string_mismatch"] \
                or rep["vocab"]["special_flag_mismatch"]:
            raise RuntimeError(f"ASCII tokenization changed after pruning: {json.dumps(rep)[:2000]}")
        summary["ascii_selfcheck"] = {"texts": rep["texts"], "equal": a["equal"]}
    return summary


# --------------------------------------------------------------------------------------------------------------
# checkpoint pruning (raw safetensors streaming)

_TOKEN_ID_KEY = re.compile(r"(?:^|_)token_(?:id|ids|index)$")
_TOKEN_LIST_KEYS = {"suppress_tokens", "begin_suppress_tokens", "bad_words_ids"}
_VOCAB_SIZE_PATHS = {"vocab_size", "text_config.vocab_size"}
_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".input_scale", ".scales", ".qweight", ".weight_packed")


def _read_st_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def _copy_range(fi, fo, offset: int, nbytes: int, chunk: int) -> None:
    fi.seek(offset)
    while nbytes > 0:
        buf = fi.read(min(chunk, nbytes))
        if not buf:
            raise IOError("unexpected end of safetensors file")
        fo.write(buf)
        nbytes -= len(buf)


def _runs(sorted_ids: list[int]) -> list[tuple[int, int]]:
    """[(start, length)] of consecutive ids."""
    out = []
    for i in sorted_ids:
        if out and out[-1][0] + out[-1][1] == i:
            out[-1][1] += 1
        else:
            out.append([i, 1])
    return [(s, n) for s, n in out]


def rewrite_safetensors(src: str, dst: str, row_slices: dict[str, list[int]], *, chunk: int = 64 << 20) -> dict:
    """Stream src -> dst: tensors named in row_slices keep only the listed rows (dim 0, sorted), every other
    tensor is copied byte for byte. Works for any dtype (bf16, fp8, ...). Returns {name: (old_shape, new_shape)}."""
    header, data_start = _read_st_header(src)
    meta = header.pop("__metadata__", None)
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    new_header: dict = {} if meta is None else {"__metadata__": meta}
    plan, offset, changed = [], 0, {}
    for name, ti in items:
        s, e = ti["data_offsets"]
        nbytes = e - s
        shape = list(ti["shape"])
        if name in row_slices:
            rows = shape[0]
            if rows == 0 or nbytes % rows:
                raise ValueError(f"{name}: cannot split {nbytes} bytes into {rows} rows")
            row_bytes = nbytes // rows
            ids = row_slices[name]
            if ids and (ids[0] < 0 or ids[-1] >= rows):
                raise ValueError(f"{name}: kept rows out of range [0, {rows})")
            new_shape = [len(ids)] + shape[1:]
            plan.append((name, s, row_bytes, _runs(ids)))
            new_nbytes = len(ids) * row_bytes
            changed[name] = (shape, new_shape)
        else:
            new_shape = shape
            plan.append((name, s, nbytes, None))
            new_nbytes = nbytes
        new_header[name] = {"dtype": ti["dtype"], "shape": new_shape, "data_offsets": [offset, offset + new_nbytes]}
        offset += new_nbytes
    missing = set(row_slices) - set(changed)
    if missing:
        raise KeyError(f"{src}: tensors not found: {sorted(missing)}")
    hb = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    hb += b" " * (-len(hb) % 8)
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        for name, s, size, runs in plan:
            if runs is None:
                _copy_range(fi, fo, data_start + s, size, chunk)
            else:
                for start, n in runs:
                    _copy_range(fi, fo, data_start + s + start * size, n * size, chunk)
        written = fo.tell()
    if written != 8 + len(hb) + offset:
        raise IOError(f"{dst}: wrote {written} bytes, expected {8 + len(hb) + offset}")
    os.replace(tmp, dst)
    return changed


def remap_config_ids(node, old2new: dict[int, int], new_vocab_size: int, path: str = "") -> list:
    """In place: every *_token_id / *_token_ids / *_token_index (and suppress/bad-words lists) is remapped,
    every vocab_size set to new_vocab_size; recurses into sub-configs (text_config, vision_config, ...).
    Returns [(key path, old, new)]. Raises ValueError if a referenced id is not kept. vocab_size is only
    rewritten at the top level and in text_config."""
    changes = []

    def remap(v, where):
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, int):
            if v not in old2new:
                raise ValueError(f"{where} = {v} refers to a token that is not kept")
            return old2new[v]
        if isinstance(v, list):
            return [remap(x, where) for x in v]
        return v

    if isinstance(node, dict):
        for k in list(node):
            v = node[k]
            where = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)) and not (_TOKEN_ID_KEY.search(k) or k in _TOKEN_LIST_KEYS):
                changes += remap_config_ids(v, old2new, new_vocab_size, where)
            elif k == "vocab_size" and isinstance(v, int):
                if where in _VOCAB_SIZE_PATHS:  # the LM vocab only (not e.g. an audio codebook size)
                    node[k] = new_vocab_size
                    changes.append((where, v, new_vocab_size))
            elif _TOKEN_ID_KEY.search(k) or k in _TOKEN_LIST_KEYS:
                nv = remap(v, where)
                if nv != v:
                    changes.append((where, v, nv))
                node[k] = nv
    elif isinstance(node, list):
        for i, v in enumerate(node):
            changes += remap_config_ids(v, old2new, new_vocab_size, f"{path}[{i}]")
    return changes


def _config_vocab_size(cfg: dict) -> int:
    if isinstance(cfg.get("text_config"), dict) and "vocab_size" in cfg["text_config"]:
        return int(cfg["text_config"]["vocab_size"])
    if "vocab_size" in cfg:
        return int(cfg["vocab_size"])
    raise ValueError("config.json has no vocab_size")


def _list_shards(src_dir: str) -> tuple[dict | None, list[str]]:
    idx_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        index = _read_json(idx_path)
        return index, sorted(set(index["weight_map"].values()))
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(src_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors in {src_dir}")
    return None, files


def is_vocab_tensor_name(name: str) -> bool:
    return name == "lm_head.weight" or name.endswith(".lm_head.weight") or name.endswith("embed_tokens.weight")


def prune_checkpoint(src_dir: str, dst_dir: str, keep: Iterable[int], *, vocab_tensors: Iterable[str] | None = None,
                     chunk: int = 64 << 20, verbose: bool = False) -> dict:
    """Slice the vocab tensors (embed_tokens, lm_head) of a safetensors checkpoint to the kept rows, shard by
    shard, copying every other tensor unchanged; rewrite model.safetensors.index.json, config.json (vocab_size = K,
    all special ids remapped incl. text/vision sub-configs), generation_config.json, and write id_map.json.
    vocab_tensors: explicit names (default: lm_head.weight / *embed_tokens.weight with dim0 == vocab_size).
    Refuses tensors with dim0 == vocab_size it does not know and quantized vocab tensors (scales present)."""
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        raise ValueError("dst_dir must differ from src_dir")
    os.makedirs(dst_dir, exist_ok=True)
    cfg = _read_json(os.path.join(src_dir, "config.json"))
    old_vocab = _config_vocab_size(cfg)
    keep = sorted({int(i) for i in keep})
    if not keep or keep[0] < 0 or keep[-1] >= old_vocab:
        raise ValueError(f"keep ids must lie in [0, {old_vocab})")
    K = len(keep)
    old2new = {o: n for n, o in enumerate(keep)}

    index, shards = _list_shards(src_dir)
    headers = {s: _read_st_header(os.path.join(src_dir, s))[0] for s in shards}
    all_names = {n: s for s, h in headers.items() for n in h if n != "__metadata__"}
    if vocab_tensors is None:
        vocab_names = sorted(n for n, s in all_names.items()
                             if is_vocab_tensor_name(n) and headers[s][n]["shape"][:1] == [old_vocab])
    else:
        vocab_names = sorted(vocab_tensors)
        for n in vocab_names:
            if n not in all_names:
                raise KeyError(f"vocab tensor {n} not in checkpoint")
            if headers[all_names[n]][n]["shape"][0] != old_vocab:
                raise ValueError(f"{n}: dim0 {headers[all_names[n]][n]['shape'][0]} != vocab_size {old_vocab}")
    if not vocab_names:
        raise ValueError(f"no embed_tokens / lm_head tensor with {old_vocab} rows found")
    unknown = sorted(n for n, s in all_names.items()
                     if n not in vocab_names and headers[s][n]["shape"][:1] == [old_vocab])
    if unknown:
        raise ValueError(f"tensors with dim0 == vocab_size that are not being pruned: {unknown}")
    for vn in vocab_names:
        base = vn[: -len(".weight")]
        quant = [n for n in all_names if n.startswith(base + ".") and n != vn and n.endswith(_SCALE_SUFFIXES)]
        if quant:
            raise NotImplementedError(f"{vn} is quantized ({quant}); only unquantized vocab tensors can be sliced")

    sliced, total = {}, 0
    for s in shards:
        rows = {n: keep for n in vocab_names if all_names[n] == s}
        changed = rewrite_safetensors(os.path.join(src_dir, s), os.path.join(dst_dir, s), rows, chunk=chunk)
        sliced.update(changed)
        h, _ = _read_st_header(os.path.join(dst_dir, s))
        total += sum(ti["data_offsets"][1] - ti["data_offsets"][0] for n, ti in h.items() if n != "__metadata__")
        if verbose:
            print(f"  {s}: {'sliced ' + ', '.join(changed) if changed else 'copied'}", flush=True)
    if index is not None:
        index = copy.deepcopy(index)
        index.setdefault("metadata", {})["total_size"] = total
        _write_json(os.path.join(dst_dir, "model.safetensors.index.json"), index)

    changes = {"config.json": remap_config_ids(cfg, old2new, K)}
    _write_json(os.path.join(dst_dir, "config.json"), cfg)
    gpath = os.path.join(src_dir, "generation_config.json")
    if os.path.exists(gpath):
        gcfg = _read_json(gpath)
        changes["generation_config.json"] = remap_config_ids(gcfg, old2new, K)
        _write_json(os.path.join(dst_dir, "generation_config.json"), gcfg)
    write_id_map(dst_dir, keep, old_vocab)
    return {"K": K, "old_vocab_size": old_vocab, "sliced": sliced, "total_size": total, "shards": len(shards),
            "config_changes": changes}


SKIP_COPY = {"crc32.txt"}  # checksums of the old tokenizer files would be stale


def prune_model_dir(src_dir: str, dst_dir: str, keep: Iterable[int], *, verbose: bool = False) -> dict:
    """prune_tokenizer + prune_checkpoint, then copy every other top-level file (processor configs, README, ...)."""
    keep = list(keep)
    tok = prune_tokenizer(src_dir, dst_dir, keep)
    ck = prune_checkpoint(src_dir, dst_dir, keep, verbose=verbose)
    copied = []
    for name in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, name)
        if not os.path.isfile(p) or os.path.exists(os.path.join(dst_dir, name)) or name in SKIP_COPY:
            continue
        if name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            raise ValueError(f"unexpected weight file {name}")
        shutil.copyfile(p, os.path.join(dst_dir, name))
        copied.append(name)
    return {"tokenizer": tok, "checkpoint": ck, "copied": copied}


# --------------------------------------------------------------------------------------------------------------
# model equivalence (tiny random model with the real vocab size)

TINY_TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                 layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                 num_attention_heads=2, num_key_value_heads=1, head_dim=64, linear_key_head_dim=16,
                 linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                 linear_conv_kernel_dim=4, full_attention_interval=4, mtp_num_hidden_layers=1)
TINY_VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                   num_position_embeddings=64)
QWEN38_DEFAULT_CONFIG = {  # fallback when no real config.json is at hand (ids from Qwen/Qwen3.8-27B)
    "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
    "image_token_id": 248056, "video_token_id": 248057, "vision_start_token_id": 248053,
    "vision_end_token_id": 248054, "tie_word_embeddings": False,
    "text_config": {"model_type": "qwen3_5_text", "vocab_size": 248320, "bos_token_id": 248044,
                    "eos_token_id": 248044, "pad_token_id": None, "tie_word_embeddings": False,
                    "partial_rotary_factor": 0.25, "attn_output_gate": True, "rms_norm_eps": 1e-6,
                    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                        "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                        "rope_type": "default"}},
    "vision_config": {"model_type": "qwen3_5", "patch_size": 16, "spatial_merge_size": 2,
                      "temporal_patch_size": 2, "in_channels": 3, "deepstack_visual_indexes": []},
}
QWEN38_DEFAULT_GENERATION = {"bos_token_id": 248044, "eos_token_id": [248046, 248044], "pad_token_id": 248044,
                             "do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}


def make_tiny_checkpoint(dst_dir: str, *, template_dir: str | None = None, seed: int = 0, dtype: str = "bfloat16",
                         shard_size: str = "20MB", add_fake_mtp: bool = True) -> dict:
    """A random Qwen3_5ForConditionalGeneration with the REAL vocab size (248,320 rows), tiny elsewhere: 4 text
    layers (3 linear_attention + 1 full_attention), a 1-block vision tower. Same tensor names as the real
    checkpoint (model.language_model.embed_tokens.weight, lm_head.weight), sharded so the two vocab tensors sit
    in different shards. add_fake_mtp: an extra shard with mtp.* tensors (one FP8 + scale) that must be copied
    byte for byte (transformers ignores mtp.* on load, like the real checkpoint).
    template_dir: config.json / generation_config.json to take special ids, rope and vocab size from."""
    import torch
    import transformers
    from transformers import Qwen3_5Config

    tpl = os.path.join(template_dir, "config.json") if template_dir else None
    cfg = _read_json(tpl) if tpl and os.path.exists(tpl) else copy.deepcopy(QWEN38_DEFAULT_CONFIG)
    cfg["text_config"].update(TINY_TEXT)
    rot = int(TINY_TEXT["head_dim"] * cfg["text_config"].get("partial_rotary_factor", 0.25)) // 2
    cfg["text_config"].setdefault("rope_parameters", {})["mrope_section"] = [rot - 2 * (rot // 3), rot // 3, rot // 3]
    cfg["vision_config"].update(TINY_VISION, out_hidden_size=TINY_TEXT["hidden_size"])
    arch = cfg.get("architectures", ["Qwen3_5ForConditionalGeneration"])[0]
    config = Qwen3_5Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "transformers_version",
                                                                         "model_type", "dtype", "torch_dtype")})
    torch.manual_seed(seed)
    model = getattr(transformers, arch)(config).to(getattr(torch, dtype)).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for emb in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            emb.copy_((torch.randn(emb.shape, generator=g) * 0.5).to(emb.dtype))
    model.save_pretrained(dst_dir, max_shard_size=shard_size)
    gtpl = os.path.join(template_dir, "generation_config.json") if template_dir else None
    gen = _read_json(gtpl) if gtpl and os.path.exists(gtpl) else dict(QWEN38_DEFAULT_GENERATION)
    _write_json(os.path.join(dst_dir, "generation_config.json"), gen)
    if add_fake_mtp:
        from safetensors.torch import save_file
        h = TINY_TEXT["hidden_size"]
        fake = {"mtp.fc.weight": torch.randn(h, 2 * h, generator=g).to(torch.bfloat16),
                "mtp.norm.weight": torch.randn(h, generator=g).to(torch.bfloat16),
                "mtp.layers.0.mlp.down_proj.weight": torch.randn(h, 2 * h, generator=g).to(torch.float8_e4m3fn),
                "mtp.layers.0.mlp.down_proj.weight_scale_inv": torch.rand(1, 1, generator=g)}
        name = "model-mtp-extra.safetensors"
        save_file(fake, os.path.join(dst_dir, name), metadata={"format": "pt"})
        ipath = os.path.join(dst_dir, "model.safetensors.index.json")
        index = _read_json(ipath)
        for k in fake:
            index["weight_map"][k] = name
        _write_json(ipath, index)
    return {"dir": dst_dir, "config": cfg}


def _model_forward(path: str, inputs, device: str, dtype):
    import torch
    import transformers
    arch = _read_json(os.path.join(path, "config.json")).get("architectures", ["AutoModelForCausalLM"])[0]
    cls = getattr(transformers, arch)
    model = cls.from_pretrained(path, dtype=dtype).to(device).eval()
    with torch.no_grad():
        out = model(input_ids=inputs.to(device), use_cache=False, output_hidden_states=True)
    logits = out.logits.detach().cpu()
    hidden = [h.detach().cpu() for h in out.hidden_states]
    del model, out
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return logits, hidden


def check_model_equivalence(src_dir: str, dst_dir: str, keep: Iterable[int] | None = None, *, n_seqs: int = 2,
                            seq_len: int = 32, device: str = "cpu", dtype: str = "bfloat16", seed: int = 0) -> dict:
    """Random sequences of kept ids through the original and the pruned checkpoint (loaded one after the other).
    The pruned logits must equal the original logits gathered at the kept ids, and all hidden states must be
    identical (exact, same dtype). Vision placeholder/special ids are excluded from the inputs."""
    import torch
    if keep is None:
        keep = load_keep(os.path.join(dst_dir, "id_map.json"))
    keep = sorted(keep)
    cfg = _read_json(os.path.join(src_dir, "config.json"))
    old_vocab = _config_vocab_size(cfg)
    old_to_new, _ = make_maps(keep, old_vocab)
    special = [v for _, v in _iter_token_ids(cfg)]
    lo = min(special) if special else old_vocab  # added/special tokens start here (248044 for Qwen3.8)
    gen = os.path.join(src_dir, "generation_config.json")
    eos = [v for _, v in _iter_token_ids(_read_json(gen))] if os.path.exists(gen) else []
    cands = torch.tensor([i for i in keep if i < lo] + sorted(set(eos)), dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    old_ids = cands[torch.randint(len(cands), (n_seqs, seq_len), generator=g)]
    new_ids = torch.from_numpy(old_to_new[old_ids.numpy()])
    tdtype = getattr(torch, dtype)
    lo_old, h_old = _model_forward(src_dir, old_ids, device, tdtype)
    ref = lo_old[..., torch.tensor(keep)]
    del lo_old
    lo_new, h_new = _model_forward(dst_dir, new_ids, device, tdtype)
    diff = (lo_new.float() - ref.float()).abs()
    return {"shape_new": list(lo_new.shape), "shape_ref": list(ref.shape), "dtype": str(lo_new.dtype),
            "logits_exact": bool(lo_new.shape == ref.shape and torch.equal(lo_new, ref)),
            "max_abs_diff": float(diff.max()), "argmax_equal": bool(torch.equal(lo_new.argmax(-1), ref.argmax(-1))),
            "hidden_exact": bool(len(h_old) == len(h_new) and all(torch.equal(a, b) for a, b in zip(h_old, h_new))),
            "n_logits": int(ref.numel())}


def _iter_token_ids(node, path=""):
    """(key path, id) for every int under a *_token_id(s) key."""
    if isinstance(node, dict):
        for k, v in node.items():
            where = f"{path}.{k}" if path else k
            if _TOKEN_ID_KEY.search(k):
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, int) and not isinstance(x, bool):
                        yield where, x
            elif isinstance(v, (dict, list)):
                yield from _iter_token_ids(v, where)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_token_ids(v, path)


# --------------------------------------------------------------------------------------------------------------
# CLI


def _corpus_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trace-bank", default=DEFAULT_TRACE_BANK,
                   help="trace bank dir (index.jsonl + per-trace dirs; see iter_trace_texts); '' (default) = none")
    p.add_argument("--models", nargs="*", default=list(DEFAULT_TRACE_MODELS), help="trace-bank models (default: all)")
    p.add_argument("--no-requests", action="store_true", help="skip the traces' request.json prompts")
    p.add_argument("--task-dir", nargs="*", default=list(DEFAULT_TASK_DIRS), dest="task_dirs")
    p.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS), help="arcsolver.prompt.arc_render arms to render")
    p.add_argument("--prompt-dir", default=DEFAULT_PROMPT_DIR, help="every file here is added verbatim")
    p.add_argument("--extra-file", nargs="*", default=[], dest="extra_files",
                   help="text/.json/.jsonl files or directories to add to the corpus")
    p.add_argument("--max-traces", type=int, default=None, help="seeded subset; -1 = all")
    p.add_argument("--max-tasks", type=int, default=None, help="per task directory; -1 = all")
    p.add_argument("--seed", type=int, default=0)


def _corpus_from_args(a) -> Iterator[str]:
    def lim(x):
        return None if x is None or x < 0 else x
    return iter_corpus_texts(a.trace_bank or None, a.models, a.task_dirs, a.arms, a.extra_files,
                             include_requests=not a.no_requests, prompt_dir=a.prompt_dir or None,
                             max_traces=lim(a.max_traces), max_tasks=lim(a.max_tasks), seed=a.seed)


def _check_keep_source(keep_path: str, src_dir: str, force: bool) -> None:
    doc = _read_json(keep_path)
    want = doc.get("stats", {}).get("tokenizer_sha256") if isinstance(doc, dict) else None
    tj = _tokenizer_json(src_dir)
    if want and os.path.exists(tj) and _sha256(tj) != want and not force:
        raise SystemExit(f"{keep_path} was built from a different tokenizer.json than {tj} (use --force)")


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("keep-set", help="build keep.json from the tokenizer + corpus")
    p.add_argument("--tokenizer", required=True, help="dir with tokenizer.json, or the file")
    p.add_argument("--out", required=True)
    p.add_argument("--extra-ids", nargs="*", type=int, default=[])
    p.add_argument("--align", type=int, default=0, help="grow K to a multiple of N with real tokens (0 = off)")
    p.add_argument("--keep-chars", default="", help="also keep tokens made of ASCII + these code points: "
                   "'symbols' (punctuation, super/subscripts, arrows, math operators) and/or hex ranges '0370-03FF'")
    _corpus_args(p)

    for name in ("prune", "prune-tokenizer", "prune-checkpoint"):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True)
        p.add_argument("--dst", required=True)
        p.add_argument("--keep", required=True, help="keep.json / id_map.json")
        p.add_argument("--force", action="store_true", help="skip the tokenizer sha256 check of keep.json")

    p = sub.add_parser("validate-tokenizer", help="old vs pruned tokenization on corpus + random texts")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n-random", type=int, default=2000, help="random ASCII and random unicode texts each")
    p.add_argument("--json", default=None)
    p.add_argument("--no-chat-template", action="store_true")
    _corpus_args(p)
    p.set_defaults(max_traces=300, max_tasks=100)

    p = sub.add_parser("validate-model", help="exact logits: original (gathered at kept ids) vs pruned")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--n-seqs", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)

    p = sub.add_parser("selftest-tiny", help="tiny random model with the real vocab: prune and check logits")
    p.add_argument("--tokenizer", required=True, help="dir with tokenizer.json (+ config.json, generation_config.json)")
    p.add_argument("--work", required=True)
    p.add_argument("--keep", default=None, help="keep.json (default: policy + a small trace sample)")
    p.add_argument("--max-traces", type=int, default=50)

    a = ap.parse_args(argv)
    if a.cmd == "keep-set":
        stats: dict = {}
        keep = build_keep_set(a.tokenizer, _corpus_from_args(a), a.extra_ids, char_ranges=parse_char_ranges(a.keep_chars),
                              align=a.align, stats=stats)
        stats["corpus_args"] = {k: getattr(a, k) for k in ("trace_bank", "models", "no_requests", "task_dirs",
                                                            "arms", "prompt_dir", "extra_files", "max_traces",
                                                            "max_tasks", "seed")}
        _write_json(a.out, {"format": KEEP_FORMAT, "keep": keep, "stats": stats}, indent=1)
        show = {k: v for k, v in stats.items() if k != "corpus_tokens_beyond_base"}
        show["corpus_tokens_beyond_base (top 40)"] = stats["corpus_tokens_beyond_base"][:40]
        _print_json(show)
        print(f"K = {len(keep)} -> {a.out}")
    elif a.cmd in ("prune", "prune-tokenizer", "prune-checkpoint"):
        _check_keep_source(a.keep, a.src, a.force)
        keep = load_keep(a.keep)
        fn = {"prune": prune_model_dir, "prune-tokenizer": prune_tokenizer, "prune-checkpoint": prune_checkpoint}[a.cmd]
        kwargs = {"verbose": True} if a.cmd != "prune-tokenizer" else {}
        _print_json(fn(a.src, a.dst, keep, **kwargs))
    elif a.cmd == "validate-tokenizer":
        texts = list(_corpus_from_args(a)) + ENGLISH_SAMPLE + NONASCII_SAMPLES \
            + random_ascii_texts(a.n_random, a.seed) + random_unicode_texts(a.n_random, a.seed)
        rep = check_tokenizer_equivalence(a.src, a.dst, texts)
        if not a.no_chat_template:
            rep["chat_template"] = check_chat_template(a.src, a.dst)
        _print_json(rep)
        if a.json:
            _write_json(a.json, rep)
        ok = (rep["all_kept"]["mismatch"] == 0 and rep["with_dropped"]["roundtrip_fail"] == 0
              and rep["vocab"]["token_string_mismatch"] == 0
              and rep.get("chat_template", {}).get("mismatch", 0) == 0)
        return 0 if ok else 1
    elif a.cmd == "validate-model":
        rep = check_model_equivalence(a.src, a.dst, device=a.device, dtype=a.dtype, n_seqs=a.n_seqs,
                                      seq_len=a.seq_len)
        _print_json(rep)
        return 0 if rep["logits_exact"] and rep["hidden_exact"] else 1
    elif a.cmd == "selftest-tiny":
        src, dst = os.path.join(a.work, "tiny_src"), os.path.join(a.work, "tiny_pruned")
        if a.keep:
            keep = load_keep(a.keep)
        else:
            keep = build_keep_set(a.tokenizer, iter_corpus_texts(max_traces=a.max_traces, max_tasks=5))
        make_tiny_checkpoint(src, template_dir=a.tokenizer)
        for name in os.listdir(a.tokenizer):  # put the real tokenizer next to the tiny weights
            if name not in ("config.json", "generation_config.json", "model.safetensors.index.json") \
                    and os.path.isfile(os.path.join(a.tokenizer, name)) and not name.endswith(".safetensors"):
                shutil.copyfile(os.path.join(a.tokenizer, name), os.path.join(src, name))
        summary = prune_model_dir(src, dst, keep)
        rep = check_model_equivalence(src, dst)
        _print_json({"prune": summary["checkpoint"]["config_changes"], "K": len(keep), "model": rep})
        return 0 if rep["logits_exact"] and rep["hidden_exact"] else 1
    return 0


def check_chat_template(old_dir: str, new_dir: str) -> dict:
    """transformers AutoTokenizer on both dirs: chat-template renderings must tokenize to mapped-equal ids, and
    the special-token attributes must point at the same strings."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {"skipped": "transformers not installed"}
    if not os.path.isdir(old_dir) or not os.path.isdir(new_dir):
        return {"skipped": "needs tokenizer directories"}
    old, new = AutoTokenizer.from_pretrained(old_dir), AutoTokenizer.from_pretrained(new_dir)
    new_to_old = np.array([old.convert_tokens_to_ids(new.convert_ids_to_tokens(i)) for i in range(len(new))])
    convs = [
        [{"role": "user", "content": "Solve this ARC puzzle.\n0120\n3333"}],
        [{"role": "system", "content": "You are precise."}, {"role": "user", "content": "3 × 4 → ? naïve 中文 😀"}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<think>\nok\n</think>\n\nhello"},
         {"role": "user", "content": "grid:\n000\n111"}],
    ]
    old_to_new = np.full(len(old), -1, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(len(new))
    r = {"conversations": 0, "equal_ids": 0, "dropped_roundtrip_ok": 0, "mismatch": 0, "len_old": len(old),
         "len_new": len(new)}
    for conv in convs:
        for kw in ({"add_generation_prompt": True}, {"add_generation_prompt": True, "enable_thinking": False},
                   {"add_generation_prompt": False}):
            o = old.apply_chat_template(conv, tokenize=True, **kw)
            n = new.apply_chat_template(conv, tokenize=True, **kw)
            o = o["input_ids"] if isinstance(o, dict) or hasattr(o, "keys") else o
            n = n["input_ids"] if isinstance(n, dict) or hasattr(n, "keys") else n
            r["conversations"] += 1
            mapped = old_to_new[np.asarray(o, dtype=np.int64)]
            if (mapped >= 0).all():  # every old token kept: ids must match exactly
                ok = mapped.tolist() == list(n)
                r["equal_ids" if ok else "mismatch"] += 1
            else:  # dropped tokens (non-ASCII text): the rendering must decode to the same string
                ok = old.decode(o) == new.decode(n)
                r["dropped_roundtrip_ok" if ok else "mismatch"] += 1
    attrs = {}
    for attr in ("eos_token", "pad_token", "bos_token", "unk_token"):
        attrs[attr] = [getattr(old, attr, None), getattr(new, attr, None)]
    r["special_attrs"] = attrs
    r["special_attrs_equal"] = all(x == y for x, y in attrs.values())
    r["eos_id_maps"] = (old.eos_token_id is None and new.eos_token_id is None) or \
        int(new_to_old[new.eos_token_id]) == old.eos_token_id
    return r


if __name__ == "__main__":
    sys.exit(main())
