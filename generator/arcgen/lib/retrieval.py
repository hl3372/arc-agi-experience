"""Cosine retrieval and diversity sampling for ARC task descriptions.

Similarity is used only to propose calibration sets, duplicate-review groups,
and novelty references.  This module deliberately makes no accept/reject
decision from an embedding score; those decisions belong to independent model
judges in the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Generic, TypeVar
import math

from .embeddings import EmbeddingError, EmbeddingProvider, Vector, cosine_similarity


T = TypeVar("T")
TextGetter = Callable[[T], str]
IdGetter = Callable[[T], str]


class RetrievalError(ValueError):
    """Raised when a retrieval request or provider response is inconsistent."""


@dataclass(frozen=True, slots=True)
class RetrievalHit(Generic[T]):
    """One deduplicated reference and its strongest candidate similarity."""

    reference: T
    reference_id: str
    score: float
    matched_query_index: int
    query_scores: tuple[float, ...]

    @property
    def task_id(self) -> str:
        """Compatibility alias used when references are real ARC tasks."""

        return self.reference_id

    @property
    def record(self) -> T:
        """Compatibility alias for callers that call the payload a record."""

        return self.reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "score": self.score,
            "matched_query_index": self.matched_query_index,
            "query_scores": list(self.query_scores),
        }


def _default_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    names = ("rule_description", "central_relation", "intended_rule", "rule", "text")
    if isinstance(value, Mapping):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, str):
                return candidate
    else:
        for name in names:
            candidate = getattr(value, name, None)
            if isinstance(candidate, str):
                return candidate
    raise RetrievalError(
        "cannot obtain retrieval text; pass text_getter or provide rule_description/central_relation"
    )


def _default_id(value: Any, index: int) -> str:
    names = ("task_id", "candidate_id", "id", "reference_id")
    if isinstance(value, Mapping):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    else:
        for name in names:
            candidate = getattr(value, name, None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return f"reference-{index:06d}"


def _texts(values: Sequence[T], getter: TextGetter[T] | None) -> list[str]:
    function: Callable[[Any], str] = getter or _default_text
    result: list[str] = []
    for index, value in enumerate(values):
        try:
            text = function(value)
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError(f"text_getter failed for item {index}: {exc}") from exc
        if not isinstance(text, str) or not text.strip():
            raise RetrievalError(f"retrieval text for item {index} must be a non-empty string")
        result.append(text)
    return result


def _vectors(provider: EmbeddingProvider, texts: Sequence[str]) -> list[Vector]:
    try:
        vectors = provider.embed(texts)
    except (EmbeddingError, OSError) as exc:
        raise RetrievalError(f"embedding provider failed: {exc}") from exc
    if not isinstance(vectors, Sequence) or len(vectors) != len(texts):
        actual = len(vectors) if isinstance(vectors, Sequence) else "non-sequence"
        raise RetrievalError(
            f"embedding provider returned {actual} vectors for {len(texts)} texts"
        )
    # cosine_similarity performs strict component/dimension validation.  Check
    # dimensions here too so a one-item request cannot hide an empty vector.
    dimension: int | None = None
    result: list[Vector] = []
    for index, vector in enumerate(vectors):
        if isinstance(vector, (str, bytes, bytearray)) or not isinstance(vector, Sequence):
            raise RetrievalError(f"embedding vector {index} is not a numeric sequence")
        if dimension is None:
            dimension = len(vector)
            if dimension == 0:
                raise RetrievalError("embedding vectors must not be empty")
        if len(vector) != dimension:
            raise RetrievalError(
                f"embedding vector {index} has dimension {len(vector)}, expected {dimension}"
            )
        converted: Vector = []
        for component_index, component in enumerate(vector):
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise RetrievalError(
                    f"embedding vector {index}[{component_index}] is not numeric"
                )
            numeric = float(component)
            if not math.isfinite(numeric):
                raise RetrievalError(
                    f"embedding vector {index}[{component_index}] is not finite"
                )
            converted.append(numeric)
        result.append(converted)
    return result


def select_diverse(
    records: Sequence[T],
    count: int,
    provider: EmbeddingProvider,
    seed: int,
    text_getter: TextGetter[T] | None = None,
) -> list[T]:
    """Select a seeded farthest-point set with low mutual cosine similarity.

    The first item is seed-selected after stable input ordering; each subsequent
    item minimises its largest similarity to anything already selected.  This
    avoids making the context revolve around a tight semantic neighbourhood.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RetrievalError("diverse selection count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RetrievalError("diverse selection seed must be an integer")
    values = list(records)
    if count > len(values):
        raise RetrievalError(f"cannot select {count} diverse items from {len(values)} records")
    texts = _texts(values, text_getter)
    vectors = _vectors(provider, texts)

    rng = Random(seed)
    # A shuffled tie order makes the seed effective without perturbing scores.
    tie_order = list(range(len(values)))
    rng.shuffle(tie_order)
    tie_rank = {index: rank for rank, index in enumerate(tie_order)}
    first = tie_order[0]
    selected = [first]
    remaining = set(range(len(values))) - {first}
    while len(selected) < count:
        candidates: list[tuple[float, float, int, int]] = []
        for index in remaining:
            similarities = [cosine_similarity(vectors[index], vectors[chosen]) for chosen in selected]
            # Minimise the closest relation first, then average relation.  The
            # seeded rank breaks exact score ties reproducibly.
            candidates.append(
                (max(similarities), sum(similarities) / len(similarities), tie_rank[index], index)
            )
        chosen = min(candidates)[-1]
        selected.append(chosen)
        remaining.remove(chosen)
    return [values[index] for index in selected]


def pairwise_similarities(
    records: Sequence[T],
    provider: EmbeddingProvider,
    text_getter: TextGetter[T] | None = None,
) -> list[tuple[int, int, float]]:
    """Return all item-index pairs sorted from most to least similar."""

    values = list(records)
    texts = _texts(values, text_getter)
    vectors = _vectors(provider, texts)
    result = [
        (left, right, cosine_similarity(vectors[left], vectors[right]))
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return sorted(result, key=lambda item: (-item[2], item[0], item[1]))


def _reference_ids(
    references: Sequence[T], id_getter: IdGetter[T] | None
) -> list[str]:
    result: list[str] = []
    for index, reference in enumerate(references):
        if id_getter is None:
            reference_id = _default_id(reference, index)
        else:
            try:
                reference_id = id_getter(reference)
            except Exception as exc:
                raise RetrievalError(f"id_getter failed for reference {index}: {exc}") from exc
            if not isinstance(reference_id, str) or not reference_id.strip():
                raise RetrievalError(f"reference id {index} must be a non-empty string")
            reference_id = reference_id.strip()
        result.append(reference_id)
    return result


def retrieve_union(
    candidate_texts: str | Sequence[str],
    references: Sequence[T],
    provider: EmbeddingProvider,
    top_k: int = 50,
    *,
    text_getter: TextGetter[T] | None = None,
    id_getter: IdGetter[T] | None = None,
    per_query_k: int | None = None,
) -> list[RetrievalHit[T]]:
    """Retrieve, deduplicate, and retain the strongest reference union.

    A separate nearest-neighbour list is formed for every candidate text (for
    example, relation-only and relation-plus-difficulty).  Their union is
    deduplicated by reference id, scored by its strongest query match, and
    trimmed to ``top_k``.  The scores are evidence for judges, never an
    automatic novelty verdict.
    """

    if isinstance(candidate_texts, str):
        queries = [candidate_texts]
    elif isinstance(candidate_texts, Sequence):
        queries = list(candidate_texts)
    else:
        raise RetrievalError("candidate_texts must be a string or sequence of strings")
    if not queries:
        raise RetrievalError("candidate_texts must not be empty")
    for index, query in enumerate(queries):
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError(f"candidate_texts[{index}] must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise RetrievalError("top_k must be a positive integer")
    if per_query_k is None:
        per_query_k = top_k
    if isinstance(per_query_k, bool) or not isinstance(per_query_k, int) or per_query_k < 1:
        raise RetrievalError("per_query_k must be a positive integer")

    reference_values = list(references)
    if not reference_values:
        raise RetrievalError("references must not be empty")
    reference_texts = _texts(reference_values, text_getter)
    reference_ids = _reference_ids(reference_values, id_getter)

    # Deduplicate before embedding.  Repeated ids carrying different text are
    # treated as corrupt input rather than silently selecting one description.
    unique_values: list[T] = []
    unique_texts: list[str] = []
    unique_ids: list[str] = []
    seen: dict[str, str] = {}
    for reference, text, reference_id in zip(
        reference_values, reference_texts, reference_ids, strict=True
    ):
        prior = seen.get(reference_id)
        if prior is not None:
            if prior != text:
                raise RetrievalError(
                    f"duplicate reference id {reference_id!r} has conflicting descriptions"
                )
            continue
        seen[reference_id] = text
        unique_values.append(reference)
        unique_texts.append(text)
        unique_ids.append(reference_id)

    vectors = _vectors(provider, queries + unique_texts)
    query_vectors = vectors[: len(queries)]
    reference_vectors = vectors[len(queries) :]
    scores_by_query: list[list[float]] = [
        [cosine_similarity(query_vector, reference_vector) for reference_vector in reference_vectors]
        for query_vector in query_vectors
    ]

    union_indices: set[int] = set()
    query_limit = min(per_query_k, len(unique_values))
    for scores in scores_by_query:
        ranked = sorted(
            range(len(unique_values)), key=lambda index: (-scores[index], unique_ids[index])
        )
        union_indices.update(ranked[:query_limit])

    hits: list[RetrievalHit[T]] = []
    for index in union_indices:
        query_scores = tuple(scores[index] for scores in scores_by_query)
        matched_query_index = max(
            range(len(query_scores)), key=lambda query_index: query_scores[query_index]
        )
        hits.append(
            RetrievalHit(
                reference=unique_values[index],
                reference_id=unique_ids[index],
                score=query_scores[matched_query_index],
                matched_query_index=matched_query_index,
                query_scores=query_scores,
            )
        )
    hits.sort(
        key=lambda hit: (
            -hit.score,
            -(sum(hit.query_scores) / len(hit.query_scores)),
            hit.reference_id,
        )
    )
    return hits[: min(top_k, len(hits))]


def retrieve_records(*args: Any, **kwargs: Any) -> list[Any]:
    """Convenience wrapper returning only payload records from ``retrieve_union``."""

    return [hit.reference for hit in retrieve_union(*args, **kwargs)]

