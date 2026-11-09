"""Small filesystem, hashing, rendering, and concurrency utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar
import hashlib
import json
import os
import re
import tempfile


T = TypeVar("T")
U = TypeVar("U")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    data = value if isinstance(value, bytes) else (
        value.encode("utf-8") if isinstance(value, str) else canonical_json(value).encode("utf-8")
    )
    return hashlib.sha256(data).hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


PROMPT_VERSION_RE = re.compile(r"prompt-version:\s*([^\s<]+)")
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def load_prompt(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8")
    match = PROMPT_VERSION_RE.search(text)
    if not match:
        raise ValueError(f"prompt lacks prompt-version marker: {path}")
    return text, match.group(1), content_hash(text)


def render_prompt(template: str, values: dict[str, Any]) -> str:
    required = set(PLACEHOLDER_RE.findall(template))
    missing = sorted(required - values.keys())
    extra = sorted(values.keys() - required)
    if missing:
        raise ValueError(f"missing prompt placeholders: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unused prompt values: {', '.join(extra)}")
    rendered = template
    for key in sorted(required):
        value = values[key]
        if not isinstance(value, str):
            # Render large grid/task payloads compactly. Fully rendered prompts
            # are still logged verbatim, while avoiding hundreds of kilobytes
            # of whitespace in later judging stages.
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        rendered = rendered.replace("{{" + key + "}}", value)
    leftovers = PLACEHOLDER_RE.findall(rendered)
    if leftovers:
        raise ValueError(f"unrendered placeholders: {', '.join(leftovers)}")
    return rendered


def parallel_map(
    function: Callable[[T], U], values: Iterable[T], workers: int
) -> list[U]:
    """Map concurrently while returning results in input order and surfacing errors."""

    items = list(values)
    if workers == 1:
        return [function(value) for value in items]
    results: list[U | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, value): index for index, value in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [value for value in results if value is not None]
