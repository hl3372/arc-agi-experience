"""Stage and explicitly commit strict new cores from the V4.4/V4.5/V4.6 qualification.

The experimental pipelines and provenance-blind panel cannot mutate the good
task bank.  This maintenance helper runs only after the panel is complete and
uses its strict consensus together with the production technical/blind/board
decision.  Exact and globally palette-equivalent episodes are committed once.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from .authorized_commits import commit_authorized_manifest
from .committed_tasks import load_committed_task_registry
from .util import atomic_write_json, content_hash, read_json
from .v41_palette import palette_equivalent_hash
from .v47_quality_panel import _load_fresh, load_v47_quality_config
from .validation import validate_arc_task

COMMIT_SET_ID = "v47-baseline-qualification-strict-core-20260811"
STAGING_ROOT = "runs/v4-7-baseline-core-commit-20260811"
OUTPUT_MANIFEST = (
    "committed_tasks/commit_manifests/"
    "v47-baseline-qualification-strict-core-20260811.json"
)


class V47CommitError(ValueError):
    """Raised when a qualification core cannot be staged losslessly."""


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(record["source_run"]))
    task_id = str(record["task_id"])
    if record["source_kind"] == "v47_paired":
        state = read_json(source / "execution" / str(record["private_arm"]) / "state.json")
        value = (state.get("semantic_records") or {}).get(task_id)
        if not isinstance(value, dict):
            raise V47CommitError(f"missing paired semantic record: {record['source_key']}")
        result = deepcopy(value)
        complete = result.get("complete_task")
        palette = result.get("palette_treatment")
        if isinstance(complete, Mapping) and isinstance(palette, Mapping):
            # A targeted repair updates the terminal semantic record but the
            # inherited V4.1 runtime can leave the original realization's
            # palette hashes in state.  Normalize identity metadata against
            # the exact task being staged; this is the same report-only
            # normalization used by the artifact-clean paired export.
            normalized = deepcopy(dict(palette))
            normalized["final_exact_task_hash"] = content_hash(complete)
            normalized["palette_equivalent_identity_hash"] = (
                palette_equivalent_hash(complete)
            )
            result["palette_treatment"] = normalized
        return result
    dossier = record.get("dossier_path")
    if not dossier or not Path(str(dossier)).is_file():
        raise V47CommitError(f"missing V4.6 authoring record: {record['source_key']}")
    authoring = read_json(Path(str(dossier)))
    outer = read_json(source / "report.json")
    row = next(
        item
        for item in outer["arms"][record["private_arm"]]["tasks"]
        if str(item["task_id"]) == task_id
    )
    editorial = row.get("editorial") or {}
    return {
        **deepcopy(authoring),
        "task_id": task_id,
        "candidate_id": task_id,
        "complete_task": deepcopy(authoring.get("complete_task")),
        "concrete_plan": deepcopy(authoring.get("plan")),
        "intrinsic_quality": row.get("experimental_disposition"),
        "portfolio_placement": editorial.get("portfolio_placement") or "include_now",
        "editorial_judgments": deepcopy(editorial.get("judgments") or []),
        "editorial_reason": editorial.get("intrinsic_reason") or "",
        "technical_results": deepcopy(row.get("technical") or {}),
        "blind_results": deepcopy(row.get("blind") or {}),
        "repair_count": 0,
        "kernel": deepcopy(authoring.get("kernel")),
        "search_mode": row.get("search_mode"),
        "realization": deepcopy(authoring.get("realization")),
        "source_arm": record["private_arm"],
        "private_kernel_author_provider": row.get("private_kernel_author_provider"),
    }


def _authoring(record: Mapping[str, Any], semantic: Mapping[str, Any]) -> dict[str, Any]:
    path = record.get("dossier_path")
    if path and Path(str(path)).is_file():
        return read_json(Path(str(path)))
    return {
        "concrete_plan": deepcopy(semantic.get("concrete_plan")),
        "realization": deepcopy(semantic.get("realization")),
        "palette_treatment": deepcopy(semantic.get("palette_treatment")),
    }


def _mentions(directory: Path, needles: Sequence[bytes]) -> bool:
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle in data for needle in needles):
            return True
    return False


def _call_directories(root: Path) -> list[Path]:
    result: list[Path] = []
    for calls_root in root.rglob("model_calls"):
        if not calls_root.is_dir():
            continue
        for stage in calls_root.iterdir():
            if not stage.is_dir():
                continue
            result.extend(path for path in stage.iterdir() if path.is_dir())
    for calls_root in root.rglob("claude_model_calls"):
        if not calls_root.is_dir():
            continue
        for stage in calls_root.iterdir():
            if not stage.is_dir():
                continue
            result.extend(path for path in stage.iterdir() if path.is_dir())
    return sorted(set(result))


def _copy_calls(
    *,
    roots: Sequence[tuple[str, Path]],
    needles: Sequence[str],
    destination: Path,
    candidate_id: str,
) -> int:
    copied = 0
    encoded = [value.encode("utf-8") for value in needles]
    for prefix, root in roots:
        if not root.is_dir():
            continue
        for call in _call_directories(root):
            if not _mentions(call, encoded):
                continue
            relative = call.relative_to(root).as_posix()
            location = sha256(relative.encode("utf-8")).hexdigest()[:12]
            target = (
                destination
                / f"{prefix}__{call.parent.name}"
                / f"{location}__{call.name}"
            )
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(call, target)
            # Paired-run upstream calls are keyed by the reservoir proposal ID,
            # while the provenance-blind quality calls deliberately use opaque
            # aliases.  Add an author-only binding only inside the commit
            # staging copy so the dossier packager can retain every relevant
            # raw call for the final candidate ID without mutating a frozen
            # source artifact.
            atomic_write_json(
                target / "commit_candidate_binding.json",
                {
                    "schema_version": 1,
                    "candidate_id": candidate_id,
                    "source_kind": prefix,
                    "source_call": relative,
                    "added_after_qualification_completion": True,
                    "semantic_or_judgment_change": False,
                },
            )
            copied += 1
    return copied


def _central_relation(record: Mapping[str, Any]) -> str:
    kernel = record.get("kernel")
    if isinstance(kernel, Mapping) and isinstance(kernel.get("central_relation"), str):
        return str(kernel["central_relation"])
    plan = record.get("concrete_plan")
    if isinstance(plan, Mapping):
        semantic = plan.get("semantic_core")
        if isinstance(semantic, Mapping):
            for key in ("central_conceptual_insight", "central_rule"):
                if isinstance(semantic.get(key), str):
                    return str(semantic[key])
    raise V47CommitError("strict core has no central relation")


def _title(record: Mapping[str, Any], task_id: str) -> str:
    words = " ".join(_central_relation(record).split()).split()
    value = " ".join(words[:12]).rstrip(".,;:")
    if len(words) > 12:
        value += "…"
    return f"{value} [{task_id}]"


def _lineage_key(record: Mapping[str, Any]) -> str:
    """Return the experiment-local mechanism lineage used for commit deduping.

    Paired V4.4/V4.5 tasks deliberately realize the same selected kernel twice,
    and shared V4.5/V4.6 candidates can likewise appear in both arms.  Those
    variants are useful experimental observations, but the good-task bank must
    receive at most one representative of the same selected concept.
    """

    return f"{record['source_run']}::{record.get('lineage_id') or record['task_id']}"


def _panel_score(decision: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[Any, ...]:
    comparisons = decision.get("committed_core_reference_comparisons") or []
    wins = sum(int(item.get("wins_over_committed_core_references") or 0) for item in comparisons)
    opportunities = sum(int(item.get("reference_count") or 0) for item in comparisons)
    reference_rate = wins / opportunities if opportunities else -1.0
    votes = decision.get("votes") or []
    mean_rank = (
        sum(int(item["judgment"]["rank"]) for item in votes) / len(votes)
        if votes
        else float("inf")
    )
    version_preference = {"v4.4": 1, "v4.5": 2, "v4.6": 3}.get(
        str(record.get("private_version")), 0
    )
    return (
        int(decision.get("core_votes") or 0),
        reference_rate,
        -mean_rank,
        record.get("portfolio_placement") == "include_now",
        version_preference,
        str(record["source_key"]),
    )


def build_v47_commit_manifest(
    repository_root: str | Path,
    *,
    quality_config_path: str | Path = "config/v4_7_baseline_quality_panel.yaml",
    output_path: str | Path = OUTPUT_MANIFEST,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_path = Path(quality_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_v47_quality_config(config_path)
    quality_run = Path(config["paths"]["runs"]) / config["run"]["id"]
    quality = read_json(quality_run / "report.json")
    if quality.get("completed") is not True or quality.get("failed_calls"):
        raise V47CommitError("quality panel is incomplete or has failed calls")
    fresh = _load_fresh(config["sources"])
    by_source = {str(item["source_key"]): item for item in fresh}
    quality_aliases: dict[str, list[str]] = {}
    for ledger_name in (
        "assignment_ledger.private.json",
        "adjudication_mapping.private.json",
    ):
        ledger_path = quality_run / ledger_name
        if not ledger_path.is_file():
            continue
        ledger = read_json(ledger_path)
        mapping = ledger.get("mapping") if ledger_name.startswith("assignment") else ledger
        if not isinstance(mapping, Mapping):
            continue
        for opaque, value in mapping.items():
            if isinstance(value, Mapping) and value.get("kind") == "fresh":
                quality_aliases.setdefault(str(value["task_hash"]), []).append(str(opaque))
    staging = root / STAGING_ROOT
    if staging.exists():
        raise V47CommitError(f"staging root already exists: {staging}")
    staging.mkdir(parents=True)
    registry = load_committed_task_registry(root / "committed_tasks/registry.json")
    represented = {
        palette_equivalent_hash(read_json(Path(entry["absolute_task_path"]))): str(entry["registry_task_id"])
        for entry in registry["entries"]
    }
    entries: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    staged: list[dict[str, Any]] = []
    try:
        strict_payloads: list[dict[str, Any]] = []
        for digest, decision in sorted(quality["decisions"].items()):
            if decision.get("strict_commit_eligible") is not True:
                continue
            candidates = [
                by_source[key]
                for key in decision["source_keys"]
                if by_source[key].get("pipeline_core") is True
                and by_source[key].get("technical_clear") is True
                and by_source[key].get("blind_serious") is not True
            ]
            if not candidates:
                raise V47CommitError(
                    f"strict decision has no qualifying source record: {digest}"
                )
            non_duplicate_candidates = [
                item
                for item in candidates
                if item.get("portfolio_placement") != "same_procedure_duplicate"
            ]
            if not non_duplicate_candidates:
                exclusions.append(
                    {
                        "task_hash": digest,
                        "source_keys": decision["source_keys"],
                        "reason": (
                            "confirmed same-minimal-procedure duplicate; independent "
                            "quality does not create a new novelty-bank concept"
                        ),
                    }
                )
                continue
            candidates = non_duplicate_candidates
            candidates.sort(
                key=lambda item: (
                    item.get("portfolio_placement") != "include_now",
                    item.get("private_version") != "v4.6",
                    str(item["source_key"]),
                )
            )
            source = candidates[0]
            strict_payloads.append(
                {"task_hash": digest, "decision": decision, "source": source}
            )

        # Experimental siblings must not silently become separate novelty-bank
        # concepts.  Keep the strongest independently reviewed realization of
        # each selected-kernel lineage and retain every excluded sibling in the
        # commit audit.
        by_lineage: dict[str, dict[str, Any]] = {}
        for payload in strict_payloads:
            lineage = _lineage_key(payload["source"])
            prior = by_lineage.get(lineage)
            if prior is None or _panel_score(
                payload["decision"], payload["source"]
            ) > _panel_score(prior["decision"], prior["source"]):
                if prior is not None:
                    exclusions.append(
                        {
                            "task_hash": prior["task_hash"],
                            "source_keys": prior["decision"]["source_keys"],
                            "reason": "same selected-kernel lineage as a stronger strict sibling",
                            "represented_by_source": payload["source"]["source_key"],
                        }
                    )
                by_lineage[lineage] = payload
            else:
                exclusions.append(
                    {
                        "task_hash": payload["task_hash"],
                        "source_keys": payload["decision"]["source_keys"],
                        "reason": "same selected-kernel lineage as a stronger strict sibling",
                        "represented_by_source": prior["source"]["source_key"],
                    }
                )

        for payload in sorted(by_lineage.values(), key=lambda item: item["task_hash"]):
            digest = payload["task_hash"]
            decision = payload["decision"]
            source = payload["source"]
            semantic = _candidate_record(source)
            complete = semantic.get("complete_task")
            validation = validate_arc_task(complete, require_test_outputs=True)
            if not validation.valid:
                raise V47CommitError(f"strict task became invalid: {source['source_key']}")
            identity = palette_equivalent_hash(complete)
            if identity in represented:
                exclusions.append(
                    {
                        "task_hash": digest,
                        "source_keys": decision["source_keys"],
                        "reason": "already represented by an exact or palette-equivalent committed task",
                        "represented_by": represented[identity],
                    }
                )
                continue
            task_id = str(source["task_id"])
            staging_id = f"{_slug(Path(source['source_run']).name)}__{_slug(source['private_arm'])}__{_slug(task_id)}"
            relative = Path(STAGING_ROOT) / staging_id
            destination = root / relative
            destination.mkdir(parents=True)
            (destination / "tasks").mkdir()
            semantic = deepcopy(semantic)
            semantic.update(
                {
                    "task_id": task_id,
                    "candidate_id": task_id,
                    "complete_task": deepcopy(complete),
                    "intrinsic_quality": "core_quality",
                    "portfolio_placement": str(source.get("portfolio_placement") or "include_now"),
                    "independent_quality_reviews": deepcopy(decision["votes"]),
                    "qualification_source_keys": deepcopy(decision["source_keys"]),
                    "qualification_task_hash": digest,
                    "strict_panel_consensus": True,
                }
            )
            authoring = _authoring(source, semantic)
            atomic_write_json(destination / "tasks" / f"{task_id}.authoring.json", authoring)
            source_run = Path(str(source["source_run"]))
            roots = [("source", source_run), ("quality", quality_run)]
            opaque_needles = [task_id, *quality_aliases.get(digest, [])]
            for identifier in (
                semantic.get("source_candidate_id"),
                semantic.get("paired_lineage_id"),
                source.get("lineage_id"),
            ):
                if isinstance(identifier, str) and identifier:
                    opaque_needles.append(identifier)
            opaque_needles = list(dict.fromkeys(opaque_needles))
            copied = _copy_calls(
                roots=roots,
                needles=opaque_needles,
                destination=destination / "model_calls",
                candidate_id=task_id,
            )
            for context in (
                "config.snapshot.json",
                "config.snapshot.yaml",
                "implementation_manifest.json",
                "prompts/manifest.json",
            ):
                source_context = source_run / context
                if source_context.is_file():
                    target = destination / context
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_context, target)
            atomic_write_json(
                destination / "state.json",
                {
                    "schema_version": 1,
                    "pipeline": "arcgen-v4.7-baseline-core-commit-staging",
                    "run_id": staging_id,
                    "completed": True,
                    "semantic_records": {task_id: semantic},
                    "source_run": str(source_run.relative_to(root)),
                    "source_arm": source["private_arm"],
                    "quality_panel_run": str(quality_run.relative_to(root)),
                    "automatic_task_commit": False,
                },
            )
            atomic_write_json(
                destination / "report.json",
                {
                    "schema_version": 1,
                    "pipeline": "arcgen-v4.7-baseline-core-commit-staging",
                    "source_key": source["source_key"],
                    "source_keys_for_exact_payload": decision["source_keys"],
                    "strict_panel_consensus": True,
                    "copied_model_call_directories": copied,
                },
            )
            normalization = {
                "schema_version": 1,
                "purpose": "dossier-ready strict qualification core normalization",
                "semantic_decisions_added": False,
                "independent_panel_preserved_in_review_record": True,
                "source_implementation_manifest_preserved": (
                    destination / "implementation_manifest.json"
                ).is_file(),
            }
            atomic_write_json(destination / "normalization_manifest.json", normalization)
            if not (destination / "implementation_manifest.json").is_file():
                atomic_write_json(destination / "implementation_manifest.json", normalization)
            registry_id = f"v47-baseline-core__{_slug(source['private_version'])}__{_slug(task_id)}"
            portfolio = str(source.get("portfolio_placement") or "include_now")
            entries.append(
                {
                    "source_run": relative.as_posix(),
                    "source_pipeline": "arcgen-v4.7-baseline-core-commit-staging",
                    "source_candidate_id": task_id,
                    "registry_task_id": registry_id,
                    "title": _title(semantic, task_id),
                    "source_portfolio_status": portfolio,
                    "authorize_portfolio_override": portfolio == "hold_crowded",
                    "authorize_all_core_override": portfolio != "include_now",
                    "authorize_same_procedure_override": False,
                    "qualification_task_hash": digest,
                }
            )
            represented[identity] = registry_id
            staged.append(
                {
                    "source_key": source["source_key"],
                    "staging_run": relative.as_posix(),
                    "task_id": task_id,
                    "registry_task_id": registry_id,
                    "copied_model_call_directories": copied,
                }
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest = {
        "schema_version": 1,
        "commit_set_id": COMMIT_SET_ID,
        "committed_on": "2026-08-11",
        "authorization_record": (
            "Explicit user authorization on 2026-08-10 and autonomous continuation "
            "authorization on 2026-08-11 to commit all new core tasks "
            "from the large V4.4/V4.5/V4.6 qualification. The staging filter also "
            "requires independent provenance-blind Sol panel consensus."
        ),
        "notes": (
            "Distinct new intrinsic core-quality episodes from the fresh baseline "
            "qualification. Each passed deterministic and rule-aware verification, "
            "had no unresolved serious blind alternative, retained both production "
            "board core judgments, and received provenance-blind independent Sol "
            "quality-panel consensus. Exact and palette-equivalent repeats are excluded."
        ),
        "audit": {
            "registry_count_before": len(registry["entries"]),
            "strict_panel_core_count": sum(
                bool(value.get("strict_commit_eligible")) for value in quality["decisions"].values()
            ),
            "manifest_entry_count": len(entries),
            "lineage_sibling_exclusion_count": sum(
                item["reason"].startswith("same selected-kernel lineage")
                for item in exclusions
            ),
            "equivalent_exclusion_count": sum(
                item["reason"].startswith("already represented")
                for item in exclusions
            ),
            "same_procedure_exclusion_count": sum(
                "same-minimal-procedure duplicate" in item["reason"]
                for item in exclusions
            ),
            "exclusions": exclusions,
            "staged": staged,
            "quality_panel_run": str(quality_run.relative_to(root)),
        },
        "entries": entries,
    }
    atomic_write_json(root / output_path, manifest)
    atomic_write_json(staging / "commit_audit.json", manifest["audit"])
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "commit"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--quality-config", default="config/v4_7_baseline_quality_panel.yaml")
    parser.add_argument("--output", default=OUTPUT_MANIFEST)
    parser.add_argument("--authorize-user-decision", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = build_v47_commit_manifest(
            args.root,
            quality_config_path=args.quality_config,
            output_path=args.output,
        )
    else:
        if not args.authorize_user_decision:
            raise V47CommitError("explicit --authorize-user-decision is required")
        value = commit_authorized_manifest(
            Path(args.root) / args.output,
            authorize_user_decision=True,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMIT_SET_ID",
    "OUTPUT_MANIFEST",
    "STAGING_ROOT",
    "V47CommitError",
    "build_v47_commit_manifest",
]
