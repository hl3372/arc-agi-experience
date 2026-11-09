"""Fresh-session model invocation, structured parsing, logging, and caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import json
import shutil
import subprocess
import tempfile
import threading
import time

from .schemas import OUTPUT_SCHEMAS, parse_stage_output
from .util import atomic_write_json, atomic_write_text, canonical_json, content_hash, read_json


WRAPPER_VERSION = "codex-cli-wrapper-v1"


class ModelInvocationError(RuntimeError):
    """Raised when a structured model call cannot be completed safely."""


@dataclass(frozen=True)
class CallResult:
    stage: str
    call_id: str
    value: dict[str, Any]
    raw_response: str
    attempts: int
    cache_key: str
    cache_source: str | None
    log_dir: Path


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_." else "_" for character in value)
    return cleaned[:180] or "call"


class ModelInvoker:
    """Invoke one isolated Codex session per role and cache only parsed successes."""

    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        *,
        output_schemas: Mapping[str, dict[str, Any]] | None = None,
        output_parser: Callable[[str, str | dict[str, Any]], Any] | None = None,
    ):
        self.config = config
        self.model = config["model"]
        self.run_dir = run_dir
        # The legacy pipeline continues to use the original registry and
        # parser by default.  Versioned pipelines can supply a disjoint schema
        # namespace without mutating or widening the frozen baseline path.
        self.output_schemas = dict(output_schemas or OUTPUT_SCHEMAS)
        self.output_parser = output_parser or parse_stage_output
        self.calls_dir = run_dir / "model_calls"
        self.schemas_dir = run_dir / "schemas"
        self.cache_dir = Path(config["paths"]["cache"]) / "model_calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mock = bool(config.get("run", {}).get("mock")) or self.model["provider"] == "mock"
        self.mock_responses: dict[str, Any] = {}
        self._semaphore = threading.BoundedSemaphore(int(self.model["concurrency"]))
        self._mock_lock = threading.Lock()
        mock_path = config.get("run", {}).get("mock_responses")
        if mock_path:
            source = Path(mock_path)
            if not source.is_absolute():
                source = Path(config["_config_path"]).parent / source
            value = read_json(source.resolve())
            if not isinstance(value, dict):
                raise ModelInvocationError("mock response fixture must be a JSON object")
            self.mock_responses = value
        self._write_schemas()
        if not self.mock and shutil.which(self.model["executable"]) is None:
            raise ModelInvocationError(
                f"model executable not found: {self.model['executable']!r}"
            )

    def _write_schemas(self) -> None:
        for stage, schema in self.output_schemas.items():
            path = self.schemas_dir / f"{_safe_name(stage)}.schema.json"
            if path.is_file():
                existing = read_json(path)
                if existing != schema:
                    raise ModelInvocationError(
                        f"structured-output schema changed within an existing run: {stage}"
                    )
            else:
                atomic_write_json(path, schema)

    def _schema(self, stage: str) -> tuple[dict[str, Any], Path]:
        try:
            schema = self.output_schemas[stage]
        except KeyError as exc:
            raise ModelInvocationError(f"no output schema registered for stage {stage!r}") from exc
        return schema, self.schemas_dir / f"{_safe_name(stage)}.schema.json"

    def _parse(self, stage: str, raw: str | dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = self.output_parser(stage, raw)
            plain = _plain(parsed)
        except Exception as exc:
            raise ModelInvocationError(f"malformed {stage} response: {exc}") from exc
        if not isinstance(plain, dict):
            raise ModelInvocationError(f"{stage} parser did not return an object")
        return plain

    def _verify_raw_parsed(
        self, stage: str, raw: str, parsed: dict[str, Any], *, source: str
    ) -> None:
        reparsed = self._parse(stage, raw)
        if reparsed != parsed:
            raise ModelInvocationError(
                f"{source} raw response does not match its parsed response"
            )

    @staticmethod
    def _validate_response_context(
        value: dict[str, Any], validator: Callable[[dict[str, Any]], None] | None
    ) -> None:
        if validator is None:
            return
        try:
            validator(value)
        except Exception as exc:
            raise ModelInvocationError(f"response failed call-context validation: {exc}") from exc

    def _cache_key(
        self,
        stage: str,
        prompt: str,
        schema: dict[str, Any],
        *,
        model_name: str,
        reasoning_effort: str,
    ) -> str:
        return content_hash(
            {
                "wrapper": WRAPPER_VERSION,
                "stage": stage,
                "prompt": prompt,
                "schema": schema,
                "provider": self.model["provider"],
                "model": model_name,
                "reasoning_effort": reasoning_effort,
            }
        )

    def _stage_execution_settings(
        self,
        stage: str,
        *,
        timeout_seconds: int | None = None,
        retries: int | None = None,
    ) -> dict[str, int]:
        override = self.model.get("stage_overrides", {}).get(stage, {})
        settings = {
            "timeout_seconds": int(
                override.get("timeout_seconds", self.model["timeout_seconds"])
            ),
            "retries": int(override.get("retries", self.model["retries"])),
        }
        if timeout_seconds is not None:
            if type(timeout_seconds) is not int or timeout_seconds < 1:
                raise ModelInvocationError("timeout_seconds override must be positive")
            settings["timeout_seconds"] = timeout_seconds
        if retries is not None:
            if type(retries) is not int or retries < 0:
                raise ModelInvocationError("retries override must be non-negative")
            settings["retries"] = retries
        return settings

    def invoke(
        self,
        *,
        stage: str,
        call_id: str,
        prompt: str,
        prompt_template_version: str,
        prompt_template_hash: str,
        parent_ids: list[str] | None = None,
        response_validator: Callable[[dict[str, Any]], None] | None = None,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: int | None = None,
        retries: int | None = None,
    ) -> CallResult:
        schema, schema_path = self._schema(stage)
        effective_model = model_name or self.model["name"]
        effective_effort = reasoning_effort or self.model["reasoning_effort"]
        if not isinstance(effective_model, str) or not effective_model.strip():
            raise ModelInvocationError("model_name override must be a non-empty string")
        if effective_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ModelInvocationError(
                "reasoning_effort override must be low, medium, high, xhigh, or max"
            )
        execution = self._stage_execution_settings(
            stage,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        cache_key = self._cache_key(
            stage,
            prompt,
            schema,
            model_name=effective_model,
            reasoning_effort=effective_effort,
        )
        log_dir = self.calls_dir / _safe_name(stage) / _safe_name(call_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "wrapper_version": WRAPPER_VERSION,
            "stage": stage,
            "call_id": call_id,
            "candidate_or_task_id": call_id,
            "parent_ids": list(parent_ids or []),
            "prompt_template_version": prompt_template_version,
            "prompt_template_hash": prompt_template_hash,
            "prompt_hash": content_hash(prompt),
            "schema_hash": content_hash(schema),
            "model": {
                "provider": self.model["provider"],
                "name": effective_model,
                "reasoning_effort": effective_effort,
                "timeout_seconds": execution["timeout_seconds"],
                "retries": execution["retries"],
            },
            "cache_key": cache_key,
        }
        request_path = log_dir / "request.json"
        prior_request_matches = False
        if request_path.is_file():
            prior_request = read_json(request_path)
            prior_request_matches = (
                isinstance(prior_request, dict)
                and prior_request.get("cache_key") == cache_key
            )
        atomic_write_json(request_path, request)
        atomic_write_text(log_dir / "prompt.md", prompt)

        parsed_path = log_dir / "parsed_response.json"
        result_path = log_dir / "result.json"
        raw_path = log_dir / "raw_response.txt"
        if parsed_path.is_file() or result_path.is_file():
            if (
                prior_request_matches
                and parsed_path.is_file()
                and raw_path.is_file()
                and not result_path.is_file()
            ):
                # The process may have been interrupted after atomic response
                # writes but before the tiny completion marker. Re-parse before
                # recovering; malformed partial data is never reused.
                value = self._parse(stage, read_json(parsed_path))
                raw_value = raw_path.read_text(encoding="utf-8")
                self._verify_raw_parsed(stage, raw_value, value, source="recovered run artifact")
                self._validate_response_context(value, response_validator)
                atomic_write_json(
                    result_path,
                    {
                        "cache_key": cache_key,
                        "cache_source": "recovered-run",
                        "retry_count": 0,
                        "raw_response_hash": content_hash(raw_value),
                        "parsed_response_hash": content_hash(value),
                    },
                )
                return CallResult(
                    stage, call_id, value, raw_value,
                    1, cache_key, "recovered-run", log_dir,
                )
            if not (parsed_path.is_file() and result_path.is_file() and raw_path.is_file()):
                raise ModelInvocationError(f"unrecoverable partial resume artifact in {log_dir}")
            result_record = read_json(result_path)
            if not isinstance(result_record, dict) or result_record.get("cache_key") != cache_key:
                raise ModelInvocationError(f"stale resume artifact in {log_dir}")
            value = self._parse(stage, read_json(parsed_path))
            raw_value = raw_path.read_text(encoding="utf-8")
            self._verify_raw_parsed(stage, raw_value, value, source="run artifact")
            self._validate_response_context(value, response_validator)
            if result_record.get("raw_response_hash") not in (None, content_hash(raw_value)):
                raise ModelInvocationError(f"run artifact raw-response hash mismatch in {log_dir}")
            if result_record.get("parsed_response_hash") not in (None, content_hash(value)):
                raise ModelInvocationError(f"run artifact parsed-response hash mismatch in {log_dir}")
            return CallResult(
                stage, call_id, value, raw_value,
                int(result_record.get("retry_count", 0)) + 1, cache_key, "run", log_dir,
            )

        global_cache = self.cache_dir / f"{cache_key}.json"
        if global_cache.is_file():
            record = read_json(global_cache)
            if not isinstance(record, dict) or record.get("cache_key") != cache_key:
                raise ModelInvocationError(f"malformed model cache entry: {global_cache}")
            if not isinstance(record.get("raw_response"), str) or "parsed_response" not in record:
                raise ModelInvocationError(f"incomplete model cache entry: {global_cache}")
            value = self._parse(stage, record["parsed_response"])
            raw_value = record["raw_response"]
            self._verify_raw_parsed(stage, raw_value, value, source="global cache entry")
            self._validate_response_context(value, response_validator)
            if record.get("raw_response_hash") not in (None, content_hash(raw_value)):
                raise ModelInvocationError(f"global cache raw-response hash mismatch: {global_cache}")
            if record.get("parsed_response_hash") not in (None, content_hash(value)):
                raise ModelInvocationError(f"global cache parsed-response hash mismatch: {global_cache}")
            if "raw_response_hash" not in record or "parsed_response_hash" not in record:
                record["raw_response_hash"] = content_hash(raw_value)
                record["parsed_response_hash"] = content_hash(value)
                atomic_write_json(global_cache, record)
            atomic_write_text(log_dir / "raw_response.txt", raw_value)
            atomic_write_json(log_dir / "parsed_response.json", value)
            atomic_write_json(
                log_dir / "result.json",
                {
                    "cache_key": cache_key,
                    "cache_source": "global",
                    "retry_count": 0,
                    "raw_response_hash": content_hash(raw_value),
                    "parsed_response_hash": content_hash(value),
                },
            )
            return CallResult(stage, call_id, value, raw_value, 1, cache_key, "global", log_dir)

        failures: list[str] = []
        for attempt in range(1, execution["retries"] + 2):
            queued_at = time.time()
            acquired_at: float | None = None
            model_finished_at: float | None = None
            parse_started_at: float | None = None
            stdout = ""
            stderr = ""
            raw = ""
            try:
                with self._semaphore:
                    acquired_at = time.time()
                    if self.mock:
                        raw = self._mock_response(stage, call_id)
                    else:
                        raw, stdout, stderr = self._codex_response(
                            prompt,
                            schema_path,
                            timeout_seconds=execution["timeout_seconds"],
                            model_name=effective_model,
                            reasoning_effort=effective_effort,
                        )
                    model_finished_at = time.time()
                parse_started_at = time.time()
                value = self._parse(stage, raw)
                self._validate_response_context(value, response_validator)
                completed_at = time.time()
                timing = {
                    "queued_at_epoch_seconds": queued_at,
                    "slot_acquired_at_epoch_seconds": acquired_at,
                    "model_finished_at_epoch_seconds": model_finished_at,
                    "parse_started_at_epoch_seconds": parse_started_at,
                    "completed_at_epoch_seconds": completed_at,
                    "queue_wait_seconds": round(acquired_at - queued_at, 3),
                    "model_execution_seconds": round(
                        model_finished_at - acquired_at, 3
                    ),
                    "parse_validation_seconds": round(
                        completed_at - parse_started_at, 3
                    ),
                    "total_seconds": round(completed_at - queued_at, 3),
                }
                atomic_write_json(
                    log_dir / f"attempt-{attempt}.timing.json", timing
                )
                atomic_write_text(log_dir / "raw_response.txt", raw)
                atomic_write_json(log_dir / "parsed_response.json", value)
                result = {
                    "cache_key": cache_key,
                    "cache_source": None,
                    "retry_count": attempt - 1,
                    # Preserve the original field for older report readers.
                    "elapsed_seconds": timing["total_seconds"],
                    "timing": timing,
                    "raw_response_hash": content_hash(raw),
                    "parsed_response_hash": content_hash(value),
                }
                atomic_write_json(log_dir / "result.json", result)
                atomic_write_json(
                    global_cache,
                    {
                        "cache_key": cache_key,
                        "stage": stage,
                        "model": request["model"],
                        "raw_response": raw,
                        "parsed_response": value,
                        "raw_response_hash": content_hash(raw),
                        "parsed_response_hash": content_hash(value),
                    },
                )
                return CallResult(stage, call_id, value, raw, attempt, cache_key, None, log_dir)
            except subprocess.TimeoutExpired as exc:
                completed_at = time.time()
                timing = {
                    "queued_at_epoch_seconds": queued_at,
                    "slot_acquired_at_epoch_seconds": acquired_at,
                    "model_finished_at_epoch_seconds": model_finished_at,
                    "parse_started_at_epoch_seconds": parse_started_at,
                    "completed_at_epoch_seconds": completed_at,
                    "queue_wait_seconds": (
                        None
                        if acquired_at is None
                        else round(acquired_at - queued_at, 3)
                    ),
                    "model_execution_seconds": (
                        None
                        if acquired_at is None
                        else round(completed_at - acquired_at, 3)
                    ),
                    "parse_validation_seconds": None,
                    "total_seconds": round(completed_at - queued_at, 3),
                    "outcome": "timeout",
                }
                atomic_write_json(
                    log_dir / f"attempt-{attempt}.timing.json", timing
                )
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                problem = (
                    f"attempt {attempt}: timed out after "
                    f"{execution['timeout_seconds']} seconds"
                )
                failures.append(problem)
                atomic_write_text(
                    log_dir / f"attempt-{attempt}.failure.log",
                    problem + "\n\n--- stdout ---\n" + stdout + "\n\n--- stderr ---\n" + stderr + "\n",
                )
            except (ModelInvocationError, OSError, subprocess.SubprocessError) as exc:
                completed_at = time.time()
                timing = {
                    "queued_at_epoch_seconds": queued_at,
                    "slot_acquired_at_epoch_seconds": acquired_at,
                    "model_finished_at_epoch_seconds": model_finished_at,
                    "parse_started_at_epoch_seconds": parse_started_at,
                    "completed_at_epoch_seconds": completed_at,
                    "queue_wait_seconds": (
                        None
                        if acquired_at is None
                        else round(acquired_at - queued_at, 3)
                    ),
                    "model_execution_seconds": (
                        None
                        if acquired_at is None
                        else round(
                            (model_finished_at or completed_at) - acquired_at,
                            3,
                        )
                    ),
                    "parse_validation_seconds": (
                        None
                        if parse_started_at is None
                        else round(completed_at - parse_started_at, 3)
                    ),
                    "total_seconds": round(completed_at - queued_at, 3),
                    "outcome": "error",
                }
                atomic_write_json(
                    log_dir / f"attempt-{attempt}.timing.json", timing
                )
                problem = f"attempt {attempt}: {exc}"
                failures.append(problem)
                if raw:
                    atomic_write_text(log_dir / f"attempt-{attempt}.raw_response.txt", raw)
                atomic_write_text(
                    log_dir / f"attempt-{attempt}.failure.log",
                    problem + "\n\n--- stdout ---\n" + stdout + "\n\n--- stderr ---\n" + stderr + "\n",
                )
        atomic_write_json(
            log_dir / "parse_failures.json",
            {"stage": stage, "call_id": call_id, "failures": failures},
        )
        raise ModelInvocationError(f"{stage}/{call_id} failed: {'; '.join(failures)}")

    def _mock_response(self, stage: str, call_id: str) -> str:
        keys = (f"{stage}/{call_id}", call_id, stage)
        with self._mock_lock:
            for key in keys:
                if key in self.mock_responses:
                    value = self.mock_responses[key]
                    if isinstance(value, list) and key == stage:
                        if not value:
                            raise ModelInvocationError(f"mock response queue exhausted for {stage}")
                        value = value.pop(0)
                    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        raise ModelInvocationError(f"no mock response for {stage}/{call_id}")

    def _codex_response(
        self,
        prompt: str,
        schema_path: Path,
        *,
        timeout_seconds: int,
        model_name: str,
        reasoning_effort: str,
    ) -> tuple[str, str, str]:
        with tempfile.TemporaryDirectory(prefix="arcgen-model-") as temporary:
            workdir = Path(temporary)
            response_path = workdir / "response.json"
            command = [
                self.model["executable"], "exec",
                "--model", model_name,
                "--config", f'model_reasoning_effort="{reasoning_effort}"',
                "--sandbox", "read-only",
                "--ephemeral", "--ignore-user-config", "--strict-config",
                "--skip-git-repo-check", "--cd", str(workdir),
                "--output-schema", str(schema_path.resolve()),
                "--output-last-message", str(response_path),
                "--color", "never", "-",
            ]
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise ModelInvocationError(
                    f"Codex exited with status {completed.returncode}: {completed.stderr[-2000:]}"
                )
            if not response_path.is_file():
                raise ModelInvocationError("Codex did not write its final response")
            raw = response_path.read_text(encoding="utf-8")
            return raw, completed.stdout, completed.stderr
