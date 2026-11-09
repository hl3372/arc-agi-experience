"""Isolated Claude Code structured-output invocation for ARCGEN experiments.

The production ARCGEN model wrapper is intentionally Codex-specific and is
part of several frozen baselines.  This module provides a separate provider
for controlled cross-model experiments without changing that wrapper.

Claude Code is run non-interactively in an empty temporary directory, with
customizations and tools disabled.  Every successful response must attest the
resolved model in Claude Code's JSON envelope; accepting the ``opus`` alias
without observing its resolution is deliberately not sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time

from .util import atomic_write_json, atomic_write_text, canonical_json, content_hash, read_json


CLAUDE_WRAPPER_VERSION = "arcgen-claude-code-wrapper-v1"
ALLOWED_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ClaudeCodeInvocationError(RuntimeError):
    """Raised when Claude Code cannot produce a safely attributable response."""


@dataclass(frozen=True)
class ClaudeCodeCapabilities:
    executable: str
    resolved_executable: str
    executable_sha256: str
    version: str
    supports_print: bool
    supports_model: bool
    supports_effort: bool
    effort_levels: tuple[str, ...]
    supports_json_schema: bool
    supports_json_output: bool
    supports_safe_mode: bool
    supports_no_session_persistence: bool
    supports_tool_disable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "resolved_executable": self.resolved_executable,
            "executable_sha256": self.executable_sha256,
            "version": self.version,
            "supports_print": self.supports_print,
            "supports_model": self.supports_model,
            "supports_effort": self.supports_effort,
            "effort_levels": list(self.effort_levels),
            "supports_json_schema": self.supports_json_schema,
            "supports_json_output": self.supports_json_output,
            "supports_safe_mode": self.supports_safe_mode,
            "supports_no_session_persistence": self.supports_no_session_persistence,
            "supports_tool_disable": self.supports_tool_disable,
        }


@dataclass(frozen=True)
class ClaudeCodeCallResult:
    stage: str
    call_id: str
    value: dict[str, Any]
    raw_response: str
    transport_envelope: dict[str, Any]
    resolved_models: tuple[str, ...]
    attempts: int
    cache_key: str
    cache_source: str | None
    log_dir: Path


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )
    return cleaned[:180] or "call"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect_claude_code(executable: str = "claude") -> ClaudeCodeCapabilities:
    """Inspect CLI capabilities without making a model call."""

    located = shutil.which(executable)
    if located is None:
        raise ClaudeCodeInvocationError(f"Claude Code executable not found: {executable!r}")
    resolved = Path(located).resolve()
    try:
        version_run = subprocess.run(
            [located, "--version"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=15, check=False,
        )
        help_run = subprocess.run(
            [located, "--help"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ClaudeCodeInvocationError(f"cannot inspect Claude Code: {exc}") from exc
    if version_run.returncode != 0 or help_run.returncode != 0:
        raise ClaudeCodeInvocationError(
            "Claude Code capability inspection failed: "
            + (version_run.stderr or help_run.stderr)[-1000:]
        )
    help_text = help_run.stdout
    effort_match = re.search(
        r"--effort\s+<level>[\s\S]{0,220}?\((low,\s*medium,\s*high,\s*xhigh,\s*max)\)",
        help_text,
    )
    levels = (
        tuple(item.strip() for item in effort_match.group(1).split(","))
        if effort_match
        else ()
    )
    capabilities = ClaudeCodeCapabilities(
        executable=executable,
        resolved_executable=str(resolved),
        executable_sha256=_file_sha256(resolved),
        version=version_run.stdout.strip(),
        supports_print="--print" in help_text,
        supports_model="--model <model>" in help_text,
        supports_effort="--effort <level>" in help_text,
        effort_levels=levels,
        supports_json_schema="--json-schema <schema>" in help_text,
        supports_json_output='"json"' in help_text and "--output-format" in help_text,
        supports_safe_mode="--safe-mode" in help_text,
        supports_no_session_persistence="--no-session-persistence" in help_text,
        supports_tool_disable='Use "" to disable all' in help_text,
    )
    required = {
        "non-interactive print": capabilities.supports_print,
        "model selection": capabilities.supports_model,
        "effort selection": capabilities.supports_effort,
        "JSON Schema": capabilities.supports_json_schema,
        "JSON output": capabilities.supports_json_output,
        "safe mode": capabilities.supports_safe_mode,
        "session isolation": capabilities.supports_no_session_persistence,
        "tool disabling": capabilities.supports_tool_disable,
        "all required efforts": set(("high", "xhigh", "max")) <= set(levels),
    }
    missing = [name for name, available in required.items() if not available]
    if missing:
        raise ClaudeCodeInvocationError(
            "Claude Code lacks required experiment capabilities: " + ", ".join(missing)
        )
    return capabilities


def sanitized_auth_status(executable: str = "claude") -> dict[str, Any]:
    """Return non-sensitive authentication readiness; this performs no model call."""

    located = shutil.which(executable)
    if located is None:
        raise ClaudeCodeInvocationError(f"Claude Code executable not found: {executable!r}")
    completed = subprocess.run(
        [located, "auth", "status"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=20, check=False,
    )
    if completed.returncode != 0:
        raise ClaudeCodeInvocationError(
            f"Claude Code auth status failed: {completed.stderr[-1000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeCodeInvocationError("Claude Code returned malformed auth status") from exc
    if not isinstance(value, Mapping):
        raise ClaudeCodeInvocationError("Claude Code returned malformed auth status")
    return {
        "logged_in": value.get("loggedIn") is True,
        "auth_method": value.get("authMethod"),
        "api_provider": value.get("apiProvider"),
        "subscription_type": value.get("subscriptionType"),
    }


def build_claude_code_command(
    *,
    executable: str,
    model_alias: str,
    effort: str,
    schema: Mapping[str, Any],
    system_prompt: str,
    max_budget_usd: float | None,
) -> list[str]:
    """Build the exact isolated non-interactive argv used for every call."""

    if effort not in ALLOWED_EFFORTS:
        raise ClaudeCodeInvocationError(f"unsupported Claude effort: {effort!r}")
    if not model_alias.strip():
        raise ClaudeCodeInvocationError("Claude model alias must be non-empty")
    command = [
        executable,
        "--print",
        "--model", model_alias,
        "--effort", effort,
        "--output-format", "json",
        "--json-schema", canonical_json(dict(schema)),
        "--system-prompt", system_prompt,
        "--safe-mode",
        "--disable-slash-commands",
        "--tools", "",
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        "--no-chrome",
        "--prompt-suggestions", "false",
    ]
    if max_budget_usd is not None:
        if max_budget_usd <= 0:
            raise ClaudeCodeInvocationError("max_budget_usd must be positive")
        command.extend(["--max-budget-usd", str(max_budget_usd)])
    return command


def resolved_models_from_envelope(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    """Identify the primary response model from transport-level accounting.

    Claude Code may use a small auxiliary model for structured-output plumbing
    even when the requested model generated the substantive response.  The
    top-level ``usage`` record describes the primary response.  Match that
    token accounting to one ``modelUsage`` entry and fail closed if multiple
    models are present but the primary cannot be identified.
    """

    direct: set[str] = set()
    for key in ("model", "model_name", "resolved_model"):
        value = envelope.get(key)
        if isinstance(value, str) and value.strip():
            direct.add(value.strip())
    if direct:
        return tuple(sorted(direct))

    usage = envelope.get("modelUsage")
    ordinary_usage = envelope.get("usage")
    if isinstance(ordinary_usage, Mapping):
        for key in ("model", "model_name", "resolved_model"):
            value = ordinary_usage.get(key)
            if isinstance(value, str) and value.strip():
                direct.add(value.strip())
        if direct:
            return tuple(sorted(direct))
    if not isinstance(usage, Mapping):
        return ()
    all_models = tuple(sorted(str(key) for key in usage if str(key).strip()))
    if len(all_models) <= 1:
        return all_models
    if not isinstance(ordinary_usage, Mapping):
        return ()

    pairs = (
        ("input_tokens", "inputTokens"),
        ("output_tokens", "outputTokens"),
        ("cache_creation_input_tokens", "cacheCreationInputTokens"),
        ("cache_read_input_tokens", "cacheReadInputTokens"),
    )
    available = [pair for pair in pairs if type(ordinary_usage.get(pair[0])) is int]
    matches = []
    for model, model_usage in usage.items():
        if not isinstance(model_usage, Mapping):
            continue
        if available and all(
            model_usage.get(camel) == ordinary_usage.get(snake)
            for snake, camel in available
        ):
            matches.append(str(model))
    return tuple(sorted(matches)) if len(matches) == 1 else ()


def all_models_from_envelope(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    """Return all primary and auxiliary model identifiers for audit logging."""

    models: set[str] = set()
    usage = envelope.get("modelUsage")
    if isinstance(usage, Mapping):
        models.update(str(key) for key in usage if str(key).strip())
    models.update(resolved_models_from_envelope(envelope))
    return tuple(sorted(models))


def parse_claude_code_envelope(
    stdout: str,
    *,
    expected_model_regex: str,
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    """Parse Claude Code's transport envelope and validate model attribution."""

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeCodeInvocationError("Claude Code emitted non-JSON transport output") from exc
    if not isinstance(envelope, dict):
        raise ClaudeCodeInvocationError("Claude Code transport output is not an object")
    if envelope.get("is_error") is True or envelope.get("subtype") in {
        "error", "error_max_turns", "error_during_execution",
    }:
        raise ClaudeCodeInvocationError(
            "Claude Code reported an error: " + str(envelope.get("result", "unknown error"))[:1000]
        )
    models = resolved_models_from_envelope(envelope)
    if not models:
        raise ClaudeCodeInvocationError(
            "Claude Code response did not unambiguously attest its primary model"
        )
    matcher = re.compile(expected_model_regex, flags=re.IGNORECASE)
    unexpected = [model for model in models if matcher.match(model) is None]
    if unexpected:
        raise ClaudeCodeInvocationError(
            "Claude Code resolved the configured alias to unexpected model(s): "
            + ", ".join(unexpected)
        )
    structured = envelope.get("structured_output")
    if structured is None:
        structured = envelope.get("result")
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError as exc:
            raise ClaudeCodeInvocationError(
                "Claude Code result did not contain structured JSON"
            ) from exc
    if not isinstance(structured, dict):
        raise ClaudeCodeInvocationError(
            "Claude Code response has no structured object"
        )
    raw = json.dumps(structured, ensure_ascii=False)
    return dict(structured), raw, models


class ClaudeCodeInvoker:
    """Structured, cached, model-attributed Claude Code invocation wrapper."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        run_dir: Path,
        *,
        cache_root: Path,
        output_schemas: Mapping[str, dict[str, Any]],
        output_parser: Callable[[str, str | dict[str, Any]], Any],
        system_prompt: str,
        capabilities: ClaudeCodeCapabilities | None = None,
        mock_responses: Mapping[str, Any] | None = None,
    ) -> None:
        self.settings = dict(settings)
        self.run_dir = run_dir
        self.calls_dir = run_dir / "claude_model_calls"
        self.schemas_dir = run_dir / "claude_schemas"
        self.cache_dir = cache_root / "claude_code_model_calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_schemas = dict(output_schemas)
        self.output_parser = output_parser
        self.system_prompt = system_prompt
        self.mock_responses = dict(mock_responses or {})
        self.mock = bool(self.settings.get("mock"))
        self._mock_lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(int(self.settings["concurrency"]))
        self.capabilities = capabilities or (
            self._mock_capabilities() if self.mock else inspect_claude_code(self.settings["executable"])
        )
        self._validate_settings()
        self._write_schemas()

    def _mock_capabilities(self) -> ClaudeCodeCapabilities:
        executable = str(self.settings.get("executable", "claude"))
        return ClaudeCodeCapabilities(
            executable=executable,
            resolved_executable=executable,
            executable_sha256="mock",
            version="Claude Code mock",
            supports_print=True,
            supports_model=True,
            supports_effort=True,
            effort_levels=ALLOWED_EFFORTS,
            supports_json_schema=True,
            supports_json_output=True,
            supports_safe_mode=True,
            supports_no_session_persistence=True,
            supports_tool_disable=True,
        )

    def _validate_settings(self) -> None:
        required = ("executable", "model_alias", "expected_model_regex", "timeout_seconds", "retries", "concurrency")
        missing = [key for key in required if key not in self.settings]
        if missing:
            raise ClaudeCodeInvocationError("missing Claude settings: " + ", ".join(missing))
        try:
            re.compile(str(self.settings["expected_model_regex"]))
        except re.error as exc:
            raise ClaudeCodeInvocationError("invalid expected_model_regex") from exc
        if type(self.settings["timeout_seconds"]) is not int or self.settings["timeout_seconds"] < 1:
            raise ClaudeCodeInvocationError("Claude timeout_seconds must be positive")
        if type(self.settings["retries"]) is not int or self.settings["retries"] < 0:
            raise ClaudeCodeInvocationError("Claude retries must be non-negative")
        if type(self.settings["concurrency"]) is not int or self.settings["concurrency"] < 1:
            raise ClaudeCodeInvocationError("Claude concurrency must be positive")

    def _write_schemas(self) -> None:
        for stage, schema in self.output_schemas.items():
            path = self.schemas_dir / f"{_safe_name(stage)}.schema.json"
            if path.exists() and read_json(path) != schema:
                raise ClaudeCodeInvocationError(f"schema changed during resume: {stage}")
            if not path.exists():
                atomic_write_json(path, schema)

    def _cache_key(self, stage: str, prompt: str, schema: Mapping[str, Any], effort: str) -> str:
        return content_hash(
            {
                "wrapper": CLAUDE_WRAPPER_VERSION,
                "stage": stage,
                "prompt": prompt,
                "schema": schema,
                "provider": "claude_code",
                "claude_code_version": self.capabilities.version,
                "claude_code_executable_sha256": self.capabilities.executable_sha256,
                "model_alias": self.settings["model_alias"],
                "expected_model_regex": self.settings["expected_model_regex"],
                "reasoning_effort": effort,
                "system_prompt": self.system_prompt,
            }
        )

    def _parse(self, stage: str, value: str | dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = self.output_parser(stage, value)
        except Exception as exc:
            raise ClaudeCodeInvocationError(f"malformed {stage} response: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ClaudeCodeInvocationError(f"{stage} parser did not return an object")
        return dict(parsed)

    def _mock_envelope(self, stage: str, call_id: str) -> str:
        keys = (f"{stage}/{call_id}", call_id, stage)
        with self._mock_lock:
            for key in keys:
                if key not in self.mock_responses:
                    continue
                value = self.mock_responses[key]
                if isinstance(value, list) and key == stage:
                    if not value:
                        raise ClaudeCodeInvocationError(f"mock response queue exhausted for {stage}")
                    value = value.pop(0)
                if isinstance(value, Mapping) and "structured_output" in value:
                    envelope = dict(value)
                else:
                    envelope = {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "structured_output": value,
                        "modelUsage": {"claude-opus-5-mock": {}},
                        "usage": {},
                    }
                return json.dumps(envelope, ensure_ascii=False)
        raise ClaudeCodeInvocationError(f"no mock response for {stage}/{call_id}")

    def _execute(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        *,
        effort: str,
        timeout_seconds: int,
    ) -> tuple[str, str]:
        if self.mock:
            raise AssertionError("mock execution is selected by invoke")
        command = build_claude_code_command(
            executable=self.capabilities.resolved_executable,
            model_alias=str(self.settings["model_alias"]),
            effort=effort,
            schema=schema,
            system_prompt=self.system_prompt,
            max_budget_usd=self.settings.get("max_budget_usd_per_call"),
        )
        with tempfile.TemporaryDirectory(prefix="arcgen-claude-") as temporary:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                cwd=temporary,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0:
            raise ClaudeCodeInvocationError(
                f"Claude Code exited with status {completed.returncode}: {completed.stderr[-2000:]}"
            )
        return completed.stdout, completed.stderr

    def invoke(
        self,
        *,
        stage: str,
        call_id: str,
        prompt: str,
        prompt_template_version: str,
        prompt_template_hash: str,
        parent_ids: Sequence[str] = (),
        response_validator: Callable[[dict[str, Any]], None] | None = None,
        reasoning_effort: str,
        timeout_seconds: int | None = None,
        retries: int | None = None,
        use_cache: bool = True,
    ) -> ClaudeCodeCallResult:
        if stage not in self.output_schemas:
            raise ClaudeCodeInvocationError(f"no output schema registered for {stage!r}")
        if reasoning_effort not in ALLOWED_EFFORTS:
            raise ClaudeCodeInvocationError(f"invalid Claude effort: {reasoning_effort}")
        if reasoning_effort not in self.capabilities.effort_levels:
            raise ClaudeCodeInvocationError(
                f"installed Claude Code does not advertise effort {reasoning_effort!r}"
            )
        schema = self.output_schemas[stage]
        timeout = int(timeout_seconds or self.settings["timeout_seconds"])
        retry_count = int(self.settings["retries"] if retries is None else retries)
        cache_key = self._cache_key(stage, prompt, schema, reasoning_effort)
        log_dir = self.calls_dir / _safe_name(stage) / _safe_name(call_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        command = build_claude_code_command(
            executable=self.capabilities.resolved_executable,
            model_alias=str(self.settings["model_alias"]),
            effort=reasoning_effort,
            schema=schema,
            system_prompt=self.system_prompt,
            max_budget_usd=self.settings.get("max_budget_usd_per_call"),
        )
        request = {
            "wrapper_version": CLAUDE_WRAPPER_VERSION,
            "stage": stage,
            "call_id": call_id,
            "parent_ids": list(parent_ids),
            "prompt_template_version": prompt_template_version,
            "prompt_template_hash": prompt_template_hash,
            "prompt_hash": content_hash(prompt),
            "schema_hash": content_hash(schema),
            "provider": "claude_code",
            "model_alias": self.settings["model_alias"],
            "expected_model_regex": self.settings["expected_model_regex"],
            "reasoning_effort": reasoning_effort,
            "claude_code_capabilities": self.capabilities.as_dict(),
            "timeout_seconds": timeout,
            "retries": retry_count,
            "cache_key": cache_key,
            "isolation": {
                "safe_mode": True,
                "tools_disabled": True,
                "slash_commands_disabled": True,
                "session_persistence": False,
                "empty_temporary_workdir": True,
            },
        }
        atomic_write_json(log_dir / "request.json", request)
        atomic_write_json(log_dir / "command.json", {"argv": command, "prompt_via_stdin": True})
        atomic_write_text(log_dir / "prompt.md", prompt)

        parsed_path = log_dir / "parsed_response.json"
        envelope_path = log_dir / "transport_envelope.json"
        result_path = log_dir / "result.json"
        raw_path = log_dir / "raw_response.txt"
        if parsed_path.exists() or result_path.exists() or envelope_path.exists():
            if not all(path.exists() for path in (parsed_path, result_path, envelope_path, raw_path)):
                raise ClaudeCodeInvocationError(f"partial Claude resume artifact in {log_dir}")
            result_record = read_json(result_path)
            if result_record.get("cache_key") != cache_key:
                raise ClaudeCodeInvocationError(f"stale Claude resume artifact in {log_dir}")
            envelope = read_json(envelope_path)
            _, raw, models = parse_claude_code_envelope(
                json.dumps(envelope, ensure_ascii=False),
                expected_model_regex=str(self.settings["expected_model_regex"]),
            )
            value = self._parse(stage, read_json(parsed_path))
            if value != self._parse(stage, raw):
                raise ClaudeCodeInvocationError("Claude run raw and parsed responses differ")
            if response_validator:
                response_validator(value)
            return ClaudeCodeCallResult(
                stage, call_id, value, raw, envelope, models,
                int(result_record.get("retry_count", 0)) + 1,
                cache_key, "run", log_dir,
            )

        global_cache = self.cache_dir / f"{cache_key}.json"
        if use_cache and global_cache.exists():
            record = read_json(global_cache)
            if record.get("cache_key") != cache_key:
                raise ClaudeCodeInvocationError("malformed Claude cache entry")
            envelope = record.get("transport_envelope")
            if not isinstance(envelope, Mapping):
                raise ClaudeCodeInvocationError("Claude cache lacks transport attribution")
            structured, raw, models = parse_claude_code_envelope(
                json.dumps(envelope, ensure_ascii=False),
                expected_model_regex=str(self.settings["expected_model_regex"]),
            )
            value = self._parse(stage, structured)
            if response_validator:
                response_validator(value)
            atomic_write_json(envelope_path, envelope)
            atomic_write_text(raw_path, raw)
            atomic_write_json(parsed_path, value)
            atomic_write_json(
                result_path,
                {
                    "cache_key": cache_key,
                    "cache_source": "global",
                    "retry_count": 0,
                    "resolved_models": list(models),
                    "all_transport_models": list(all_models_from_envelope(envelope)),
                    "usage": envelope.get("usage", {}),
                    "total_cost_usd": envelope.get("total_cost_usd"),
                },
            )
            return ClaudeCodeCallResult(
                stage, call_id, value, raw, dict(envelope), models,
                1, cache_key, "global", log_dir,
            )

        failures: list[str] = []
        for attempt in range(1, retry_count + 2):
            queued_at = time.time()
            acquired_at: float | None = None
            model_finished_at: float | None = None
            stdout = ""
            stderr = ""
            try:
                with self._semaphore:
                    acquired_at = time.time()
                    if self.mock:
                        stdout = self._mock_envelope(stage, call_id)
                    else:
                        stdout, stderr = self._execute(
                            prompt, schema, effort=reasoning_effort,
                            timeout_seconds=timeout,
                        )
                    model_finished_at = time.time()
                structured, raw, models = parse_claude_code_envelope(
                    stdout,
                    expected_model_regex=str(self.settings["expected_model_regex"]),
                )
                value = self._parse(stage, structured)
                if response_validator:
                    response_validator(value)
                completed_at = time.time()
                envelope = json.loads(stdout)
                timing = {
                    "queued_at_epoch_seconds": queued_at,
                    "slot_acquired_at_epoch_seconds": acquired_at,
                    "model_finished_at_epoch_seconds": model_finished_at,
                    "completed_at_epoch_seconds": completed_at,
                    "queue_wait_seconds": round(acquired_at - queued_at, 3),
                    "model_execution_seconds": round(model_finished_at - acquired_at, 3),
                    "parse_validation_seconds": round(completed_at - model_finished_at, 3),
                    "total_seconds": round(completed_at - queued_at, 3),
                }
                atomic_write_json(log_dir / f"attempt-{attempt}.timing.json", timing)
                atomic_write_text(log_dir / f"attempt-{attempt}.stdout.json", stdout)
                atomic_write_text(log_dir / f"attempt-{attempt}.stderr.txt", stderr)
                atomic_write_json(envelope_path, envelope)
                atomic_write_text(raw_path, raw)
                atomic_write_json(parsed_path, value)
                result_record = {
                    "cache_key": cache_key,
                    "cache_source": None,
                    "retry_count": attempt - 1,
                    "timing": timing,
                    "resolved_models": list(models),
                    "all_transport_models": list(all_models_from_envelope(envelope)),
                    "usage": envelope.get("usage", {}),
                    "model_usage": envelope.get("modelUsage", {}),
                    "total_cost_usd": envelope.get("total_cost_usd"),
                    "raw_response_hash": content_hash(raw),
                    "parsed_response_hash": content_hash(value),
                    "transport_envelope_hash": content_hash(envelope),
                }
                atomic_write_json(result_path, result_record)
                if use_cache:
                    atomic_write_json(
                        global_cache,
                        {
                            "cache_key": cache_key,
                            "stage": stage,
                            "provider": "claude_code",
                            "model_alias": self.settings["model_alias"],
                            "reasoning_effort": reasoning_effort,
                            "resolved_models": list(models),
                            "transport_envelope": envelope,
                            "parsed_response": value,
                        },
                    )
                return ClaudeCodeCallResult(
                    stage, call_id, value, raw, envelope, models,
                    attempt, cache_key, None, log_dir,
                )
            except subprocess.TimeoutExpired as exc:
                problem = f"attempt {attempt}: timed out after {timeout} seconds"
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            except (ClaudeCodeInvocationError, OSError, subprocess.SubprocessError) as exc:
                problem = f"attempt {attempt}: {exc}"
            failures.append(problem)
            completed_at = time.time()
            atomic_write_json(
                log_dir / f"attempt-{attempt}.timing.json",
                {
                    "queued_at_epoch_seconds": queued_at,
                    "slot_acquired_at_epoch_seconds": acquired_at,
                    "model_finished_at_epoch_seconds": model_finished_at,
                    "completed_at_epoch_seconds": completed_at,
                    "queue_wait_seconds": None if acquired_at is None else round(acquired_at - queued_at, 3),
                    "model_execution_seconds": None if acquired_at is None else round(completed_at - acquired_at, 3),
                    "total_seconds": round(completed_at - queued_at, 3),
                    "outcome": "error",
                },
            )
            atomic_write_text(
                log_dir / f"attempt-{attempt}.failure.log",
                problem + "\n\n--- stdout ---\n" + stdout + "\n\n--- stderr ---\n" + stderr,
            )
        atomic_write_json(
            log_dir / "parse_failures.json",
            {"stage": stage, "call_id": call_id, "failures": failures},
        )
        raise ClaudeCodeInvocationError(
            f"{stage}/{call_id} failed: " + "; ".join(failures)
        )


ATTESTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "attestation_token": {"type": "string"},
        "effort_acknowledgment": {"type": "string"},
    },
    "required": ["attestation_token", "effort_acknowledgment"],
    "additionalProperties": False,
}


def attest_claude_role_efforts(
    *,
    settings: Mapping[str, Any],
    run_dir: Path,
    cache_root: Path,
    system_prompt: str,
    efforts: Sequence[str],
    destination: Path,
) -> dict[str, Any]:
    """Make explicit tiny calls proving Opus resolution at every used effort."""

    capabilities = inspect_claude_code(str(settings["executable"]))

    def parser(stage: str, value: str | dict[str, Any]) -> dict[str, Any]:
        if stage != "claude_opus_attestation":
            raise ClaudeCodeInvocationError("unexpected attestation stage")
        data = json.loads(value) if isinstance(value, str) else dict(value)
        if set(data) != {"attestation_token", "effort_acknowledgment"}:
            raise ClaudeCodeInvocationError("malformed Claude attestation response")
        return data

    invoker = ClaudeCodeInvoker(
        settings,
        run_dir,
        cache_root=cache_root,
        output_schemas={"claude_opus_attestation": ATTESTATION_SCHEMA},
        output_parser=parser,
        system_prompt=system_prompt,
        capabilities=capabilities,
    )
    rows = []
    nonce = content_hash({"time": time.time_ns(), "version": capabilities.version})[:20]
    for effort in efforts:
        token = f"arcgen-opus-attestation-{effort}-{nonce}"

        def validate(value: dict[str, Any], expected: str = token) -> None:
            if value.get("attestation_token") != expected:
                raise ClaudeCodeInvocationError("Claude attestation token mismatch")

        result = invoker.invoke(
            stage="claude_opus_attestation",
            call_id=f"attest-{effort}-{nonce}",
            prompt=(
                "Return the requested structured object. Set attestation_token exactly to "
                f"{token!r}. Set effort_acknowledgment to a short statement that you received "
                f"an ARCGEN capability check configured at effort {effort!r}."
            ),
            prompt_template_version="arcgen-claude-opus-attestation-v1",
            prompt_template_hash=content_hash("arcgen-claude-opus-attestation-v1"),
            reasoning_effort=effort,
            response_validator=validate,
            retries=0,
            use_cache=False,
        )
        rows.append(
            {
                "effort": effort,
                "resolved_models": list(result.resolved_models),
                "call_log": str(result.log_dir),
                "cache_key": result.cache_key,
                "response_hash": content_hash(result.value),
            }
        )
    record = {
        "schema_version": 1,
        "kind": "claude_code_opus_role_effort_attestation",
        "created_at_epoch_seconds": round(time.time(), 3),
        "wrapper_version": CLAUDE_WRAPPER_VERSION,
        "model_alias": settings["model_alias"],
        "expected_model_regex": settings["expected_model_regex"],
        "capabilities": capabilities.as_dict(),
        "required_efforts": list(efforts),
        "attestations": rows,
    }
    record["attestation_hash"] = content_hash(record)
    atomic_write_json(destination, record)
    return record


def validate_claude_attestation(
    path: Path,
    *,
    settings: Mapping[str, Any],
    capabilities: ClaudeCodeCapabilities,
    required_efforts: Sequence[str],
    maximum_age_seconds: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise ClaudeCodeInvocationError(
            f"Claude Opus attestation is missing: {path}. Run the explicit attest-model command first."
        )
    record = read_json(path)
    supplied_hash = record.pop("attestation_hash", None)
    if supplied_hash != content_hash(record):
        raise ClaudeCodeInvocationError("Claude attestation hash is invalid")
    record["attestation_hash"] = supplied_hash
    if record.get("model_alias") != settings["model_alias"]:
        raise ClaudeCodeInvocationError("Claude attestation used a different model alias")
    if record.get("expected_model_regex") != settings["expected_model_regex"]:
        raise ClaudeCodeInvocationError("Claude attestation used a different model expectation")
    frozen = record.get("capabilities", {})
    for key, current in (
        ("version", capabilities.version),
        ("resolved_executable", capabilities.resolved_executable),
        ("executable_sha256", capabilities.executable_sha256),
    ):
        if frozen.get(key) != current:
            raise ClaudeCodeInvocationError(f"Claude installation changed since attestation: {key}")
    age = time.time() - float(record.get("created_at_epoch_seconds", 0))
    if age < 0 or age > maximum_age_seconds:
        raise ClaudeCodeInvocationError("Claude Opus attestation is stale")
    rows = record.get("attestations")
    if not isinstance(rows, list):
        raise ClaudeCodeInvocationError("Claude attestation rows are malformed")
    by_effort = {item.get("effort"): item for item in rows if isinstance(item, Mapping)}
    if set(required_efforts) - set(by_effort):
        raise ClaudeCodeInvocationError("Claude attestation does not cover every configured effort")
    matcher = re.compile(str(settings["expected_model_regex"]), flags=re.IGNORECASE)
    for effort in required_efforts:
        models = by_effort[effort].get("resolved_models", [])
        if not models or any(not isinstance(model, str) or not matcher.match(model) for model in models):
            raise ClaudeCodeInvocationError(
                f"Claude attestation for effort {effort} lacks a valid Opus 5 resolution"
            )
    return record


__all__ = [
    "ALLOWED_EFFORTS",
    "ATTESTATION_SCHEMA",
    "CLAUDE_WRAPPER_VERSION",
    "ClaudeCodeCallResult",
    "ClaudeCodeCapabilities",
    "ClaudeCodeInvocationError",
    "ClaudeCodeInvoker",
    "attest_claude_role_efforts",
    "all_models_from_envelope",
    "build_claude_code_command",
    "inspect_claude_code",
    "parse_claude_code_envelope",
    "resolved_models_from_envelope",
    "sanitized_auth_status",
    "validate_claude_attestation",
]
