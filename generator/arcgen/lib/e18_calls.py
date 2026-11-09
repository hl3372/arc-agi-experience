"""Attempt-bounded invocation wrapper used by every E18 model runtime."""

from __future__ import annotations

import random
import re
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .model import CallResult, ModelInvocationError

# Stages whose retry also receives the preceding attempt's validation error (opt-in; empty for r23 behavior).
SEMANTIC_FEEDBACK_STAGES: set[str] = set()
from .util import atomic_write_json, read_json
from .util import content_hash
from copy import deepcopy
import json
from arcgen.astra_evidence_runtime_revision_20260911.structure import retry_feedback

ATTEMPT_POLICY = 'two-attempt-own-structure-feedback-v1'


# A physical attempt lost to the network (Codex could not reach its service: "Reconnecting... 5/5", "workspace routing
# discovery failed", a dropped stream, a suspended laptop waking up). The retry waits first, so both attempts do not fall
# inside the same outage (v1-b11 lost four tasks that way on 2026-10-07), and gets no "your response failed validation"
# feedback, since there was no response.
# Only outage signatures: routine Codex stderr also carries MCP "error sending request" and model-refresh "request
# timed out" lines, which do not mean the attempt was lost to the network.
# "Selected model is at capacity" is a provider outage of the same kind: the v1-b16 re-screen (2026-10-13) lost an idea
# call whose retry came 1 ms after the first capacity error and failed the same way; v1-b15 met it eight times.
# "429 Too Many Requests" (after Codex's own quick retries) is the same: in v1-b16-t and -g (2026-10-14) the provider
# refused calls in bursts of under a minute, on both accounts alike, when many calls started together; each retry came
# about 5 s later inside the same burst, and about 9% of calls failed both attempts. The wait is jittered by up to
# NETWORK_JITTER_SECONDS so that calls refused together do not all retry together.
NETWORK_FAILURE = re.compile(r'Reconnecting\.\.\. 5/5|routing discovery failed|stream disconnected before completion'
                             r'|network is unreachable|temporary failure in name resolution|selected model is at capacity'
                             r'|429 Too Many Requests', re.I)
NETWORK_BACKOFF_SECONDS = 180
NETWORK_JITTER_SECONDS = 120


def network_failure(error) -> bool:
    return bool(error) and bool(NETWORK_FAILURE.search(str(error)))


class E18AttemptError(ModelInvocationError):
    """Raised when the frozen two-attempt logical-call ceiling is exhausted."""


def _call_dir(inner: Any, stage: str, call_id: str) -> Path:
    visited: set[int] = set()
    target = inner
    while id(target) not in visited:
        visited.add(id(target))
        if hasattr(target, "calls_dir"):
            safe = lambda value: (
                "".join(
                    character if character.isalnum() or character in "-_." else "_"
                    for character in value
                )[:180]
                or "call"
            )
            return Path(target.calls_dir) / safe(stage) / safe(call_id)
        if not hasattr(target, "inner"):
            break
        target = target.inner
    raise E18AttemptError("cannot locate underlying model-call directory")


class E18AttemptInvoker:
    """Persist dispatch intent before every physical provider attempt.

    The historical wrapper performs both physical retries inside one method.
    E18 instead asks it for one attempt at a time and owns the two-attempt
    ceiling in a separate journal.  An interruption after dispatch intent but
    before an outcome consumes that attempt, so a reboot can never create an
    unrecorded third attempt.
    """

    def __init__(
        self, inner: Any, journal_path: str | Path, *, maximum_attempts: int = 2
    ) -> None:
        if maximum_attempts != 2:
            raise E18AttemptError("E18 freezes exactly two physical attempts")
        self.inner = inner
        self.journal_path = Path(journal_path)
        self.maximum_attempts = maximum_attempts
        self.lock = threading.RLock()
        self.value = (
            read_json(self.journal_path)
            if self.journal_path.is_file()
            else {"schema_version": 2, "policy": ATTEMPT_POLICY, "maximum_attempts": 2, "calls": {}}
        )
        self._validate()
        self._reconcile_interrupted_dispatches()
        self._validate()
        self._save()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _save(self) -> None:
        atomic_write_json(self.journal_path, self.value)

    def _validate(self) -> None:
        if (
            self.value.get("schema_version") != 2
            or self.value.get("policy") != ATTEMPT_POLICY
            or self.value.get("maximum_attempts") != 2
        ):
            raise E18AttemptError("E18 attempt journal schema changed")
        calls = self.value.get("calls")
        if not isinstance(calls, Mapping):
            raise E18AttemptError("E18 attempt journal calls must be a mapping")
        for key, row in calls.items():
            if not isinstance(row, Mapping) or not isinstance(
                row.get("attempts"), list
            ):
                raise E18AttemptError(f"malformed E18 attempt journal row {key}")
            attempts = row["attempts"]
            if [item.get("attempt") for item in attempts] != list(
                range(1, len(attempts) + 1)
            ):
                raise E18AttemptError(f"non-contiguous E18 attempts for {key}")
            if len(attempts) > 2:
                raise E18AttemptError(f"E18 attempt ceiling exceeded for {key}")

    def _reconcile_interrupted_dispatches(self) -> None:
        """Terminalize dispatch intents left open by a stopped coordinator.

        A persisted ``started`` row has already consumed its physical-attempt
        allowance.  If the provider result marker exists, the attempt finished
        and is reconciled as complete.  Otherwise it is explicitly marked
        interrupted so resume may use only the remaining allowance and the
        reporting ledger never mistakes a nonterminal attempt for a clean one.
        """

        now = round(time.time(), 3)
        for row in self.value["calls"].values():
            attempts = row["attempts"]
            if not attempts or attempts[-1].get("outcome") != "started":
                continue
            call_dir = _call_dir(self.inner, str(row["stage"]), str(row["call_id"]))
            current = attempts[-1]
            current["completed_at_epoch_seconds"] = now
            if (call_dir / "result.json").is_file():
                current["outcome"] = "complete_reconciled"
                row["status"] = "complete"
            else:
                current["outcome"] = "interrupted"
                row["status"] = (
                    "failed"
                    if len(attempts) >= self.maximum_attempts
                    else "retry_available"
                )

    @staticmethod
    def _archive_failed_attempt(call_dir: Path, attempt: int) -> None:
        destination = call_dir / "physical_attempts" / f"attempt-{attempt:02d}"
        if destination.exists():
            return
        files = []
        for pattern in (
            "attempt-*.timing.json",
            "attempt-*.failure.log",
            "attempt-*.raw_response.txt",
            "parse_failures.json",
            "request.json",
            "prompt.md",
        ):
            files.extend(call_dir.glob(pattern))
        if not files:
            return
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(set(files)):
            shutil.move(str(path), destination / path.name)

    @staticmethod
    def _attempt_kwargs(kwargs, attempt):
        value = dict(kwargs)
        feedback = attempt.get('structural_feedback')
        if feedback:
            value['prompt'] += '\n\nLocal validation feedback from your preceding response:\n' + json.dumps(feedback, sort_keys=True)
        fingerprint = content_hash({k: v for k, v in value.items() if k != 'response_validator'})
        if attempt.get('request_fingerprint') not in (None, fingerprint):
            raise E18AttemptError('persisted attempt request changed on resume')
        return value, fingerprint

    def invoke(self, *, stage: str, call_id: str, **kwargs: Any) -> CallResult:
        key = f"{stage}/{call_id}"
        call_dir = _call_dir(self.inner, stage, call_id)
        logical_hash = content_hash({k: v for k, v in kwargs.items() if k != 'response_validator'})
        with self.lock:
            row = self.value["calls"].setdefault(
                key,
                {
                    "stage": stage,
                    "call_id": call_id,
                    "attempts": [],
                    "status": "new",
                    "logical_request_fingerprint": logical_hash,
                },
            )
            if row["stage"] != stage or row["call_id"] != call_id:
                raise E18AttemptError("E18 attempt-journal call identity changed")
            if row.get('logical_request_fingerprint') != logical_hash:
                raise E18AttemptError('logical request changed on resume')
            # A result is the authoritative tiny completion marker.  Reconcile
            # an interruption that happened after it was written.
            if (call_dir / "result.json").is_file():
                if row["attempts"] and row["attempts"][-1].get("outcome") == "started":
                    row["attempts"][-1]["outcome"] = "complete_reconciled"
                    row["attempts"][-1]["completed_at_epoch_seconds"] = round(
                        time.time(), 3
                    )
                row["status"] = "complete"
                self._save()
                if not row['attempts']:
                    raise E18AttemptError('completion marker has no persisted dispatch intent')
                actual, _ = self._attempt_kwargs(kwargs, row['attempts'][-1])
                result = self.inner.invoke(stage=stage, call_id=call_id, retries=0, **actual)
                return replace(result, attempts=max(1, len(row["attempts"])))
            used = len(row["attempts"])
            if used >= self.maximum_attempts:
                row["status"] = "failed"
                self._save()
                raise E18AttemptError(f"{key} exhausted its two physical attempts")

        last_error: Exception | None = None
        while True:
            if last_error is not None and network_failure(last_error) and len(self.value["calls"][key]["attempts"]) < self.maximum_attempts:
                time.sleep(NETWORK_BACKOFF_SECONDS + random.uniform(0, NETWORK_JITTER_SECONDS))
            with self.lock:
                row = self.value["calls"][key]
                used = len(row["attempts"])
                if used >= self.maximum_attempts:
                    row["status"] = "failed"
                    self._save()
                    if last_error is not None:
                        raise E18AttemptError(
                            f"{key} exhausted its two physical attempts"
                        ) from last_error
                    raise E18AttemptError(f"{key} exhausted its two physical attempts")
                guard = getattr(self, "before_attempt", None)
                if guard is not None:
                    guard(stage, call_id)
                if used:
                    feedback = retry_feedback(call_dir, self.inner.output_schemas[stage])
                    prior_error = row["attempts"][-1].get("error") if stage in SEMANTIC_FEEDBACK_STAGES else None
                    if network_failure(prior_error):
                        prior_error = None
                    if prior_error:
                        feedback = {**(feedback or {}), "semantic_policy": "own_response_validation_v1",
                                    "semantic_issue": str(prior_error)[-1500:],
                                    "semantic_instruction": ("Your preceding response failed the validation above. Return a complete corrected "
                                                             "response to the original request that fixes exactly this issue. Use the reference "
                                                             "tokens for every frozen field instead of re-typing it.")}
                    self._archive_failed_attempt(call_dir, used)
                else:
                    feedback = None
                attempt = used + 1
                intent = {
                        "attempt": attempt,
                        "outcome": "started",
                        "started_at_epoch_seconds": round(time.time(), 3),
                        'structural_feedback': deepcopy(feedback),
                    }
                actual, fingerprint = self._attempt_kwargs(kwargs, intent)
                intent['request_fingerprint'] = fingerprint
                row["attempts"].append(intent)
                row["status"] = "running"
                self._save()
            try:
                result = self.inner.invoke(
                    stage=stage, call_id=call_id, retries=0, **actual
                )
            except Exception as exc:  # noqa: BLE001 - the journal owns retry policy
                last_error = exc
                with self.lock:
                    row = self.value["calls"][key]
                    current = row["attempts"][attempt - 1]
                    current["outcome"] = "failed"
                    current["error"] = str(exc)
                    current["completed_at_epoch_seconds"] = round(time.time(), 3)
                    row["status"] = (
                        "failed"
                        if attempt >= self.maximum_attempts
                        else "retry_available"
                    )
                    self._save()
                failed = getattr(self, "after_failed_attempt", None)
                if failed is not None:
                    failed(stage, call_id, exc)
                continue
            with self.lock:
                row = self.value["calls"][key]
                current = row["attempts"][attempt - 1]
                current["outcome"] = "complete"
                current["cache_source"] = result.cache_source
                current["completed_at_epoch_seconds"] = round(time.time(), 3)
                row["status"] = "complete"
                row["log_dir"] = str(result.log_dir.resolve())
                self._save()
            return replace(result, attempts=attempt)


__all__ = ["E18AttemptError", "E18AttemptInvoker"]
