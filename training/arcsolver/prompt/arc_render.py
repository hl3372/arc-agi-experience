"""ARC task -> chat messages, exactly as the training traces were prompted.

The two main arms share the user message (user template "{task_block}", grid format
`indexed_runs_compact`) and differ only in the system prompt:
  r2k_zb_max          prompts/base_cruns_compact_zb.md          one answer
  r2k_zb_two_opt_max  prompts/base_cruns_compact_zb_two_opt.md  "Attempt 1:" (+ optional "Attempt 2:")

Pure python, no third-party imports: usable from the training environment and the vLLM eval
environment alike. Also holds the answer parser and the scoring helpers shared by the eval and tests.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    "r2k_zb_max": os.path.join(HERE, "prompts", "base_cruns_compact_zb.md"),
    "r2k_zb_two_opt_max": os.path.join(HERE, "prompts", "base_cruns_compact_zb_two_opt.md"),
    # Render-style traces number training pairs and tests from 0 ("pair 0"), so these arms label them from 0 too,
    # and their system prompts drop the "write working rows as digit runs" rule the renders don't follow.
    # render_think: reasoning in <think>, grids after it (like r2k_zb_max). render_output: no
    # reasoning channel; the reasoning is the visible reply and the grids follow a final ANSWER_MARKER line.
    "render_think": os.path.join(HERE, "prompts", "render_think.md"),
    "render_output": os.path.join(HERE, "prompts", "render_output.md"),
    # Plain-grid arm (for an off-the-shelf model): plain grids (digit rows plus each grid's size, no row labels or run
    # listings), so "answer in the same format as the grids" is unambiguous; examples numbered from 1. Answers are
    # parsed leniently (prose before the grids is ignored, the last grid blocks are the answer): see answer_grids.
    "qwen38": os.path.join(HERE, "prompts", "qwen38_solve.md"),
}
DEFAULT_ARM = "r2k_zb_max"
ZERO_BASED_ARMS = frozenset({"render_think", "render_output"})
OUTPUT_CHANNEL_ARMS = frozenset({"render_output"})
PLAIN_GRID_ARMS = frozenset({"qwen38"})
LENIENT_ANSWER_ARMS = frozenset({"qwen38"})
ANSWER_MARKER = "Final answer:"


_DEFAULT_CHANNEL = ["think"]


def set_default_channel(ch: str) -> None:
    """The channel used when a parser is called without one. The RL processes set it once from --arm (their
    reward code calls the parsers without a channel); everything else keeps "think"."""
    if ch not in ("think", "output"):
        raise ValueError(f"unknown channel {ch!r}")
    _DEFAULT_CHANNEL[0] = ch


def default_channel() -> str:
    return _DEFAULT_CHANNEL[0]


def channel(arm: str = DEFAULT_ARM) -> str:
    """'think': reasoning inside <think>...</think>, answer after it (generation with enable_thinking=True).
    'output': an empty think block, the reasoning in the reply, the answer after the last ANSWER_MARKER line
    (generation with enable_thinking=False)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return "output" if arm in OUTPUT_CHANNEL_ARMS else "think"

RUNS_TITLE = "Runs of the same grid (color@columns):"


def system_prompt(arm: str = DEFAULT_ARM) -> str:
    with open(ARMS[arm], encoding="utf-8") as f:
        text = f.read()
    # A file's terminal newline is a storage artifact, not prompt text.
    return text[:-1] if text.endswith("\n") else text


def _row_labels(count):
    width = len(f"r{count - 1}")
    return [f"r{index}".ljust(width) for index in range(count)]


def _indexed_rows(grid, separator=" "):
    return [f"{label} | " + separator.join(str(cell) for cell in row)
            for label, row in zip(_row_labels(len(grid)), grid)]


def _runs(values, axis):
    """`color@aSTART-aEND` per maximal run; a single cell is `color@aK`."""
    parts, start = [], 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            parts.append(f"{values[start]}@{axis}{start}" if index - start == 1
                         else f"{values[start]}@{axis}{start}-{axis}{index - 1}")
            start = index
    return " ".join(parts)


def _run_rows(grid):
    return [RUNS_TITLE] + [f"r{index}: {_runs(row, 'c')}" for index, row in enumerate(grid)]


def _indexed_block(label, grid, separator=""):
    lines = [f"{label} ({len(grid)} rows x {len(grid[0])} cols):"] + _indexed_rows(grid, separator)
    lines += _run_rows(grid)
    return "\n".join(lines)


def task_block(task, zero_based: bool = False) -> str:
    """`indexed_runs_compact`: training pairs in order, then test inputs in test order. Pairs and tests are
    numbered from 1 (the original arms) or from 0 (zero_based, the render arms)."""
    first = 0 if zero_based else 1
    parts = []
    for index, pair in enumerate(task["train"], first):
        for role in ("input", "output"):
            parts.append(_indexed_block(f"Training pair {index} {role}", pair[role]))
    for index, pair in enumerate(task["test"], first):
        parts.append(_indexed_block(f"Test input {index}", pair["input"]))
    return "\n\n".join(parts)


def _plain_grid(label, grid):
    return "\n".join([f"{label} ({len(grid)} rows, {len(grid[0])} columns):"]
                     + ["".join(str(c) for c in row) for row in grid])


def task_block_plain(task) -> str:
    """The qwen38 arm's user message: training examples numbered from 1 (input and output, each with its size),
    then the test input(s); grids are bare digit rows, the same format the answer must use."""
    parts = []
    for index, pair in enumerate(task["train"], 1):
        parts.append(f"Training example {index}\n" + _plain_grid("Input", pair["input"]) + "\n\n"
                     + _plain_grid("Output", pair["output"]))
    tests = task["test"]
    for index, pair in enumerate(tests, 1):
        parts.append(_plain_grid("Test input" if len(tests) == 1 else f"Test input {index}", pair["input"]))
    return "\n\n".join(parts)


def build_messages(task, arm: str = DEFAULT_ARM) -> list[dict]:
    """[system, user] exactly as the given arm sent them (user template "{task_block}")."""
    if arm in PLAIN_GRID_ARMS:
        return [{"role": "system", "content": system_prompt(arm)}, {"role": "user", "content": task_block_plain(task)}]
    return [{"role": "system", "content": system_prompt(arm)},
            {"role": "user", "content": task_block(task, zero_based=arm in ZERO_BASED_ARMS)}]


def load_task(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- answers -----------------------------------------------------------------------------------

THINK_END = "</think>"
_ROW = re.compile(r"^[0-9]+$")


_MARKER_LINE = re.compile(r"^[ \t]*" + re.escape(ANSWER_MARKER) + r"[ \t]*$", re.M)


def final_answer_text(completion: str, channel: str | None = None) -> tuple[str, bool]:
    """(text after the reasoning, reasoning_closed). Without a closing tag the sample never
    reached an answer (usually a truncation), and nothing is parsed from its reasoning.
    channel="output" (render_output arm): the reasoning is the reply itself and closes at the last line that is
    exactly ANSWER_MARKER. channel=None: default_channel()."""
    channel = channel or _DEFAULT_CHANNEL[0]
    if channel == "output":
        marks = list(_MARKER_LINE.finditer(completion))
        if not marks:
            return "", False
        return completion[marks[-1].end():], True
    if channel != "think":
        raise ValueError(f"unknown channel {channel!r}")
    if THINK_END not in completion:
        return "", False
    return completion.rsplit(THINK_END, 1)[1], True


def reasoning_part(completion: str, channel: str | None = None) -> str:
    """The reasoning: before the first "</think>" (think channel) or before the last ANSWER_MARKER line (output
    channel); the whole text if it never closed."""
    channel = channel or _DEFAULT_CHANNEL[0]
    if channel == "output":
        marks = list(_MARKER_LINE.finditer(completion))
        return completion[:marks[-1].start()] if marks else completion
    if channel != "think":
        raise ValueError(f"unknown channel {channel!r}")
    return completion.split(THINK_END)[0]


_ATTEMPT = re.compile(r"^\s*Attempt\s+([12])\s*:\s*$", re.M)


def split_attempts(answer: str) -> list[str]:
    """The two_opt arm writes "Attempt 1:" (and optionally "Attempt 2:") headers; the single
    answer arm writes grids only. Returns the attempt texts in order (one element if no header)."""
    marks = list(_ATTEMPT.finditer(answer))
    if not marks:
        return [answer]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(answer)
        out.append(answer[m.end():end])
    return out


def parse_grids(answer: str) -> list[list[list[int]]] | None:
    """Blank-line separated blocks of digit rows -> grids, per the system prompt's answer
    contract. Surrounding whitespace is tolerated; anything else in a block (labels, code
    fences, prose, spaced digits, ragged rows) makes that block unparseable (None)."""
    blocks = [b for b in re.split(r"\n\s*\n", answer.strip()) if b.strip()]
    if not blocks:
        return None
    grids = []
    for block in blocks:
        rows = [line.strip() for line in block.strip().split("\n")]
        if not rows or not all(_ROW.match(r) for r in rows) or len({len(r) for r in rows}) != 1:
            grids.append(None)
        else:
            grids.append([[int(c) for c in r] for r in rows])
    return grids


_SPACED_ROW = re.compile(r"^[0-9]( [0-9])+$")
_FENCE = re.compile(r"^\s*```")


def parse_grids_lenient(answer: str, n_test: int) -> list | None:
    """The answer grids of a lenient arm: every maximal run of consecutive digit-row lines is a grid candidate
    (a row of single digits separated by single spaces counts too; code-fence lines, prose and blank lines end a
    run); a ragged run is a malformed grid (None). The answer is the LAST n_test candidates, in order, so prose or
    working before the final grids is ignored. Fewer candidates than test outputs -> the missing ones are None.
    No candidate at all -> None."""
    runs, cur = [], []
    for line in answer.split("\n"):
        s = line.strip()
        if _ROW.match(s):
            cur.append(s)
            continue
        if _SPACED_ROW.match(s):
            cur.append(s.replace(" ", ""))
            continue
        if cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return None
    grids = [[[int(c) for c in r] for r in rows] if len({len(r) for r in rows}) == 1 else None for rows in runs]
    tail = grids[-n_test:] if n_test > 0 else []
    return [None] * (n_test - len(tail)) + tail


_ANSWER_MODE = ["strict"]


def answer_mode(arm: str = DEFAULT_ARM) -> str:
    """'lenient' for LENIENT_ANSWER_ARMS (parse_grids_lenient), else 'strict' (parse_grids on attempt 1)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return "lenient" if arm in LENIENT_ANSWER_ARMS else "strict"


def set_answer_mode(mode: str) -> None:
    if mode not in ("strict", "lenient"):
        raise ValueError(f"unknown answer mode {mode!r}")
    _ANSWER_MODE[0] = mode


def configure_arm(arm: str) -> None:
    """Process-wide parser settings for an arm: the reasoning channel and the answer mode."""
    set_default_channel(channel(arm))
    set_answer_mode(answer_mode(arm))


def answer_grids(answer: str, n_test: int, mode: str | None = None) -> tuple[list | None, list | None]:
    """(grids of the scored answer, grids of a second attempt or None) from the text after the reasoning."""
    mode = mode or _ANSWER_MODE[0]
    if mode == "lenient":
        return parse_grids_lenient(answer, n_test), None
    if mode != "strict":
        raise ValueError(f"unknown answer mode {mode!r}")
    attempts = split_attempts(answer)
    return (parse_grids(attempts[0]) if attempts else None), (parse_grids(attempts[1]) if len(attempts) > 1 else None)


def grid_key(grid) -> str | None:
    return None if grid is None else "\n".join("".join(str(c) for c in row) for row in grid)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021)."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def vote_top(keys: list, k: int) -> list:
    """The k most frequent parsed answers (ties broken by first appearance), unparsed excluded."""
    counts = Counter(x for x in keys if x is not None)
    first = {}
    for i, x in enumerate(keys):
        if x is not None and x not in first:
            first[x] = i
    return sorted(counts, key=lambda x: (-counts[x], first[x]))[:k]


def score_task(task, completions: list[str], ks=(1, 2, 8), channel: str | None = None,
               answer_mode: str | None = None) -> dict:
    """Official-style ARC scoring per test output, averaged over the task's test outputs.

    For each test output j: c_j = samples whose j-th grid matches exactly; pass@k is the
    unbiased estimator over the n samples; vote@2 = the correct grid is among the 2 most
    frequent answers (what a 2-attempt submission built from these samples would score).
    A sample that writes "Attempt 1:/Attempt 2:" is scored on attempt 1 only, so pass@k always
    counts one answer per sample; `second_attempt_rescues` reports how often attempt 2 alone
    would have been right.
    """
    expected = [grid_key(t["output"]) for t in task["test"]] if all("output" in t for t in task["test"]) else None
    n = len(completions)
    parsed, second = [], []
    closed = with_second = 0
    mode = answer_mode or _ANSWER_MODE[0]
    for comp in completions:
        text, ok = final_answer_text(comp, channel)
        closed += ok
        if mode == "lenient":
            parsed.append(parse_grids_lenient(text, len(task["test"])) if ok else None)
            second.append(None)
            continue
        attempts = split_attempts(text) if ok else []
        parsed.append(parse_grids(attempts[0]) if attempts else None)
        second.append(parse_grids(attempts[1]) if len(attempts) > 1 else None)
        with_second += len(attempts) > 1
    per_output = []
    for j in range(len(task["test"])):
        keys = [grid_key(g[j]) if (g is not None and j < len(g)) else None for g in parsed]
        rec = {"n_parsed": sum(k is not None for k in keys)}
        if expected is not None:
            c = sum(1 for k in keys if k == expected[j])
            rec["correct"] = c
            for k in ks:
                if k <= n:
                    rec[f"pass@{k}"] = pass_at_k(n, c, k)
            rec["vote@2"] = float(expected[j] in vote_top(keys, 2))
        per_output.append(rec)
    out = {"n_samples": n, "n_test": len(task["test"]), "closed": closed, "with_second_attempt": with_second,
           "well_formed": sum(1 for g in parsed if g is not None and len(g) == len(task["test"])
                              and all(x is not None for x in g)),
           "per_output": per_output}
    if expected is not None:
        for key in [f"pass@{k}" for k in ks if k <= n] + ["vote@2"]:
            out[key] = sum(p[key] for p in per_output) / len(per_output)
        # stricter task-level view: every test output right in the same sample
        all_right = sum(1 for g in parsed if g is not None and len(g) >= len(expected)
                        and all(grid_key(g[j]) == expected[j] for j in range(len(expected))))
        out["task_all_correct@1"] = all_right / n
        rescues = 0
        for g1, g2 in zip(parsed, second):
            if g2 is None:
                continue
            for j in range(len(expected)):
                a1 = grid_key(g1[j]) if (g1 is not None and j < len(g1)) else None
                a2 = grid_key(g2[j]) if j < len(g2) else None
                rescues += a1 != expected[j] and a2 == expected[j]
        out["second_attempt_rescues"] = rescues
    return out
