"""Configurable, deterministic text embeddings with an inspectable disk cache.

The benchmark depends only on the :class:`EmbeddingProvider` interface.  The
included local provider uses stable feature hashing over word and character
n-grams, sublinear term frequency, and optional corpus-derived inverse document
frequency.  It is intentionally retrieval-oriented rather than a source of
task mechanisms.  Provider and model identifiers come from configuration; no
remote embedding model name is embedded in this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request


Vector: TypeAlias = list[float]
ProviderBuilder: TypeAlias = Callable[[Mapping[str, Any]], "EmbeddingProvider"]
_LOCAL_PROVIDER_VERSION = "local-hashing-tfidf-v1"
_OPENROUTER_PROVIDER_VERSION = "openrouter-embeddings-v1"


class EmbeddingError(ValueError):
    """Raised when embedding configuration or provider output is invalid."""


class EmbeddingCacheError(EmbeddingError):
    """Raised when an existing cache entry is malformed or inconsistent."""


class EmbeddingProvider(ABC):
    """Small provider boundary used by orchestration and retrieval code."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed each input text, preserving order and cardinality."""


def _canonicalizable(value: Any, *, context: str = "config") -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise EmbeddingError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EmbeddingError(f"{context} keys must be strings")
            result[key] = _canonicalizable(item, context=f"{context}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalizable(item, context=context) for item in value]
    raise EmbeddingError(f"{context} contains non-serializable value {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalizable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EmbeddingError(f"{name} must be a positive integer")
    return value


def _ngram_bounds(value: Any, name: str, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        low = high = value
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
    ):
        low, high = value
    else:
        raise EmbeddingError(f"{name} must be an integer or [minimum, maximum]")
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, int)
        or not isinstance(high, int)
        or low < 1
        or high < low
    ):
        raise EmbeddingError(f"{name} must contain positive ascending integers")
    return low, high


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class _Feature:
    name: str
    scale: float


class LocalHashingTfidfProvider(EmbeddingProvider):
    """Deterministic hashing-vectorizer suitable for local MVP retrieval.

    If ``idf_documents`` is supplied in the provider mapping, true smoothed IDF
    weights are fitted once from those documents.  Without that optional
    corpus, the provider remains corpus-independent (IDF is one), which makes
    vectors cacheable solely by provider configuration and text.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.dimensions = _positive_integer(
            config.get("dimensions", config.get("dimension", 1024)), "embedding.dimensions"
        )
        self.word_ngrams = _ngram_bounds(
            config.get("word_ngrams", config.get("word_ngram_range")),
            "embedding.word_ngrams",
            (1, 2),
        )
        self.char_ngrams = _ngram_bounds(
            config.get("char_ngrams", config.get("char_ngram_range")),
            "embedding.char_ngrams",
            (3, 5),
        )
        lowercase = config.get("lowercase", True)
        if not isinstance(lowercase, bool):
            raise EmbeddingError("embedding.lowercase must be boolean")
        self.lowercase = lowercase
        word_scale = config.get("word_scale", 1.0)
        char_scale = config.get("char_scale", 0.35)
        if (
            isinstance(word_scale, bool)
            or not isinstance(word_scale, (int, float))
            or not math.isfinite(word_scale)
            or word_scale <= 0
        ):
            raise EmbeddingError("embedding.word_scale must be a positive finite number")
        if (
            isinstance(char_scale, bool)
            or not isinstance(char_scale, (int, float))
            or not math.isfinite(char_scale)
            or char_scale <= 0
        ):
            raise EmbeddingError("embedding.char_scale must be a positive finite number")
        self.word_scale = float(word_scale)
        self.char_scale = float(char_scale)
        self._idf = self._fit_optional_idf(config.get("idf_documents"))

    def _normalise(self, text: str) -> str:
        value = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            value = value.casefold()
        return " ".join(value.split())

    def _features(self, text: str) -> Counter[_Feature]:
        normal = self._normalise(text)
        words = _WORD_RE.findall(normal)
        features: Counter[_Feature] = Counter()
        word_low, word_high = self.word_ngrams
        for size in range(word_low, word_high + 1):
            for index in range(max(0, len(words) - size + 1)):
                features[_Feature(f"w{size}:" + " ".join(words[index : index + size]), self.word_scale)] += 1

        # Boundary sentinels preserve starts/ends while collapsed whitespace
        # makes equivalent formatting produce the same representation.
        chars = "^" + normal + "$"
        char_low, char_high = self.char_ngrams
        for size in range(char_low, char_high + 1):
            for index in range(max(0, len(chars) - size + 1)):
                features[_Feature(f"c{size}:" + chars[index : index + size], self.char_scale)] += 1
        return features

    def _fit_optional_idf(self, documents: Any) -> dict[str, float]:
        if documents is None:
            return {}
        if isinstance(documents, (str, bytes, bytearray)) or not isinstance(documents, Sequence):
            raise EmbeddingError("embedding.idf_documents must be a list of strings")
        if not documents:
            raise EmbeddingError("embedding.idf_documents must not be empty")
        document_frequency: Counter[str] = Counter()
        for index, text in enumerate(documents):
            if not isinstance(text, str):
                raise EmbeddingError(f"embedding.idf_documents[{index}] must be a string")
            document_frequency.update({feature.name for feature in self._features(text)})
        count = len(documents)
        return {
            feature: math.log((1.0 + count) / (1.0 + frequency)) + 1.0
            for feature, frequency in document_frequency.items()
        }

    def _hash(self, feature: str) -> tuple[int, float]:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % self.dimensions
        sign = -1.0 if digest[8] & 1 else 1.0
        return index, sign

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self.dimensions
        for feature, frequency in self._features(text).items():
            index, sign = self._hash(feature.name)
            term_frequency = 1.0 + math.log(frequency)
            inverse_document_frequency = self._idf.get(feature.name, 1.0)
            vector[index] += sign * feature.scale * term_frequency * inverse_document_frequency
        norm = math.sqrt(sum(component * component for component in vector))
        if norm:
            vector = [component / norm for component in vector]
        return vector

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
            raise EmbeddingError("embed expects a sequence of strings, not a single string")
        result: list[Vector] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingError(f"texts[{index}] must be a string")
            result.append(self._embed_one(text))
        return result


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """Remote embedding provider configured entirely from the run config.

    The API key is read from a named environment variable at call time and is
    never copied into configuration snapshots, cache keys, or run artifacts.
    The model name is required in configuration; this implementation does not
    contain a model default.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        model = config.get("model")
        if not isinstance(model, str) or not model.strip():
            raise EmbeddingError("embedding.model must be configured for OpenRouter")
        self.model = model.strip()
        base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        if not isinstance(base_url, str) or not base_url.strip():
            raise EmbeddingError("embedding.base_url must be a non-empty string")
        self.endpoint = base_url.rstrip("/") + "/embeddings"
        key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
        if not isinstance(key_env, str) or not key_env.strip():
            raise EmbeddingError("embedding.api_key_env must be a non-empty string")
        self.api_key_env = key_env.strip()
        self.batch_size = _positive_integer(
            config.get("batch_size", 64), "embedding.batch_size"
        )
        self.timeout_seconds = _positive_integer(
            config.get("timeout_seconds", 120), "embedding.timeout_seconds"
        )
        retries = config.get("retries", 2)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise EmbeddingError("embedding.retries must be a non-negative integer")
        self.retries = retries
        dimensions = config.get("dimensions")
        if dimensions is not None:
            dimensions = _positive_integer(dimensions, "embedding.dimensions")
        self.dimensions: int | None = dimensions
        site_url = config.get("site_url", "https://openai.com/codex")
        app_name = config.get("app_name", "arcgen-lean-authoring-v1")
        if not isinstance(site_url, str) or not isinstance(app_name, str):
            raise EmbeddingError("embedding.site_url and app_name must be strings")
        self.site_url = site_url
        self.app_name = app_name

    def _request(self, texts: Sequence[str]) -> list[Vector]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise EmbeddingError(
                f"embedding API key environment variable {self.api_key_env!r} is not set"
            )
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            method="POST",
        )
        failures: list[str] = []
        for attempt in range(self.retries + 1):
            # Isolated program accounting only; no request, key or text content.
            observer = getattr(self, "_request_observer", None)
            if observer is not None:
                observer(attempt + 1, len(texts))
            try:
                # The socket timeout only fires on silence; OpenRouter keeps a stuck request alive with
                # keep-alive bytes, so the body is read in chunks under an overall deadline as well.
                deadline = time.monotonic() + self.timeout_seconds
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    chunks = []
                    while True:
                        if time.monotonic() > deadline:
                            raise TimeoutError("embedding response exceeded its overall deadline")
                        chunk = response.read1(65536)  # returns after one network read, so the deadline is checked often
                        if not chunk:
                            break
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                value = json.loads(raw)
                if not isinstance(value, dict) or not isinstance(value.get("data"), list):
                    raise EmbeddingError("OpenRouter returned an object without data[]")
                entries = value["data"]
                indexed: list[tuple[int, Any]] = []
                for position, entry in enumerate(entries):
                    if not isinstance(entry, dict) or "embedding" not in entry:
                        raise EmbeddingError(
                            f"OpenRouter embedding data[{position}] is malformed"
                        )
                    index = entry.get("index", position)
                    if isinstance(index, bool) or not isinstance(index, int):
                        raise EmbeddingError(
                            f"OpenRouter embedding data[{position}].index is invalid"
                        )
                    indexed.append((index, entry["embedding"]))
                indexed.sort(key=lambda item: item[0])
                return _validate_vectors(
                    [entry for _, entry in indexed],
                    expected_count=len(texts),
                    expected_dimension=self.dimensions,
                )
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                EmbeddingError,
            ) as exc:
                detail = str(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        body = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    if body:
                        detail += ": " + body[:500]
                failures.append(f"attempt {attempt + 1}: {detail}")
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 2))
        raise EmbeddingError("OpenRouter embedding request failed; " + "; ".join(failures))

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
            raise EmbeddingError("embed expects a sequence of strings, not a single string")
        values = list(texts)
        for index, text in enumerate(values):
            if not isinstance(text, str):
                raise EmbeddingError(f"texts[{index}] must be a string")
        result: list[Vector] = []
        for start in range(0, len(values), self.batch_size):
            result.extend(self._request(values[start : start + self.batch_size]))
        return result


def _validate_vectors(
    vectors: Any, *, expected_count: int, expected_dimension: int | None = None
) -> list[Vector]:
    if expected_dimension is not None and (
        isinstance(expected_dimension, bool)
        or not isinstance(expected_dimension, int)
        or expected_dimension < 1
    ):
        raise EmbeddingError("expected embedding dimension must be a positive integer")
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes, bytearray)):
        raise EmbeddingError("embedding provider returned a non-sequence")
    if len(vectors) != expected_count:
        raise EmbeddingError(
            f"embedding provider returned {len(vectors)} vectors for {expected_count} texts"
        )
    result: list[Vector] = []
    dimension = expected_dimension
    for vector_index, vector in enumerate(vectors):
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
            raise EmbeddingError(f"embedding vector {vector_index} is not a sequence")
        if dimension is None:
            dimension = len(vector)
            if dimension == 0:
                raise EmbeddingError("embedding vectors must not be empty")
        if len(vector) != dimension:
            raise EmbeddingError(
                f"embedding vector {vector_index} has dimension {len(vector)}, expected {dimension}"
            )
        converted: Vector = []
        for component_index, component in enumerate(vector):
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise EmbeddingError(
                    f"embedding vector {vector_index}[{component_index}] is not numeric"
                )
            numeric = float(component)
            if not math.isfinite(numeric):
                raise EmbeddingError(
                    f"embedding vector {vector_index}[{component_index}] is not finite"
                )
            converted.append(numeric)
        result.append(converted)
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class CachedEmbeddingProvider(EmbeddingProvider):
    """Cache successful per-text vectors by provider configuration and text."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        provider_config: Mapping[str, Any],
        cache_dir: str | Path,
    ) -> None:
        self.provider = provider
        effective_config = {
            "implementation": _LOCAL_PROVIDER_VERSION,
            "provider_config": _canonicalizable(provider_config),
        }
        self._config_json = _canonical_json(effective_config)
        self._config_hash = _sha256(self._config_json)
        self.cache_dir = Path(cache_dir) / "embeddings" / self._config_hash

    def _key(self, text: str) -> str:
        return _sha256(self._config_json + "\x00" + text)

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read(self, path: Path, *, key: str, text: str) -> Vector:
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EmbeddingCacheError(f"malformed embedding cache entry {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise EmbeddingCacheError(f"embedding cache entry root is not an object: {path}")
        expected = {
            "version": 1,
            "key": key,
            "provider_config_hash": self._config_hash,
            "text_sha256": _sha256(text),
        }
        for field, expected_value in expected.items():
            if value.get(field) != expected_value:
                raise EmbeddingCacheError(
                    f"embedding cache entry {path} has inconsistent {field}"
                )
        dimension = value.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise EmbeddingCacheError(
                f"embedding cache entry {path} has invalid or missing dimension"
            )
        try:
            vectors = _validate_vectors(
                [value.get("vector")],
                expected_count=1,
                expected_dimension=dimension,
            )
        except EmbeddingError as exc:
            raise EmbeddingCacheError(f"invalid vector in embedding cache entry {path}: {exc}") from exc
        return vectors[0]

    def _write(self, path: Path, *, key: str, text: str, vector: Vector) -> None:
        _atomic_json(
            path,
            {
                "version": 1,
                "key": key,
                "provider_config_hash": self._config_hash,
                "text_sha256": _sha256(text),
                "dimension": len(vector),
                "vector": vector,
            },
        )

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
            raise EmbeddingError("embed expects a sequence of strings, not a single string")
        values = list(texts)
        for index, text in enumerate(values):
            if not isinstance(text, str):
                raise EmbeddingError(f"texts[{index}] must be a string")
        results: list[Vector | None] = [None] * len(values)
        missing_positions: dict[str, list[int]] = {}
        missing_text: dict[str, str] = {}
        for index, text in enumerate(values):
            key = self._key(text)
            path = self._path(key)
            if path.exists():
                results[index] = self._read(path, key=key, text=text)
            else:
                missing_positions.setdefault(key, []).append(index)
                missing_text[key] = text

        if missing_positions:
            keys = list(missing_positions)
            generated = _validate_vectors(
                self.provider.embed([missing_text[key] for key in keys]),
                expected_count=len(keys),
            )
            for key, vector in zip(keys, generated, strict=True):
                text = missing_text[key]
                self._write(self._path(key), key=key, text=text, vector=vector)
                for index in missing_positions[key]:
                    results[index] = list(vector)

        if any(vector is None for vector in results):
            raise AssertionError("internal error: embedding result was not populated")
        return [list(vector) for vector in results if vector is not None]


# Concise compatibility name for callers that do not need to distinguish the
# bundled implementation from externally registered providers.
HashingTfidfProvider = LocalHashingTfidfProvider


_PROVIDER_BUILDERS: dict[str, ProviderBuilder] = {}


def register_embedding_provider(
    name: str, builder: ProviderBuilder, *, replace: bool = False
) -> None:
    """Register a configured provider implementation for external integrations."""

    if not isinstance(name, str) or not name.strip():
        raise EmbeddingError("provider registration name must be a non-empty string")
    if not callable(builder):
        raise EmbeddingError("provider builder must be callable")
    key = name.strip().casefold()
    if key in _PROVIDER_BUILDERS and not replace:
        raise EmbeddingError(f"embedding provider already registered: {name}")
    _PROVIDER_BUILDERS[key] = builder


def _build_local(config: Mapping[str, Any]) -> EmbeddingProvider:
    return LocalHashingTfidfProvider(config)


def _build_openrouter(config: Mapping[str, Any]) -> EmbeddingProvider:
    return OpenRouterEmbeddingProvider(config)


for _local_name in ("hashing_tfidf", "local_hashing_tfidf", "local", "hashing"):
    register_embedding_provider(_local_name, _build_local)
for _remote_name in ("openrouter", "openrouter_embeddings"):
    register_embedding_provider(_remote_name, _build_openrouter)


def build_embedding_provider(
    config: Mapping[str, Any], cache_dir: str | Path | None = None
) -> EmbeddingProvider:
    """Build an embedding provider entirely from a configuration mapping.

    Third-party providers can be installed with :func:`register_embedding_provider`.
    The full mapping—including any configured model identifier—is part of each
    cache key.  Unknown providers fail rather than silently falling back.
    """

    if not isinstance(config, Mapping):
        raise EmbeddingError("embedding configuration must be a mapping")
    provider_name = config.get("provider", config.get("type"))
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise EmbeddingError("embedding.provider must be a non-empty string")
    provider_key = provider_name.strip().casefold()
    try:
        builder = _PROVIDER_BUILDERS[provider_key]
    except KeyError as exc:
        available = ", ".join(sorted(_PROVIDER_BUILDERS))
        raise EmbeddingError(
            f"unknown embedding provider {provider_name!r}; registered providers: {available}"
        ) from exc
    safe_config = _canonicalizable(config)
    provider = builder(safe_config)
    if cache_dir is None:
        return provider
    return CachedEmbeddingProvider(provider, safe_config, cache_dir)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity with strict dimension/numeric validation."""

    if isinstance(left, (str, bytes, bytearray)) or isinstance(right, (str, bytes, bytearray)):
        raise EmbeddingError("cosine inputs must be numeric sequences")
    if len(left) != len(right):
        raise EmbeddingError(
            f"cosine dimension mismatch: left has {len(left)}, right has {len(right)}"
        )
    if not left:
        raise EmbeddingError("cosine inputs must not be empty")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
        if (
            isinstance(left_item, bool)
            or isinstance(right_item, bool)
            or not isinstance(left_item, (int, float))
            or not isinstance(right_item, (int, float))
        ):
            raise EmbeddingError(f"cosine component {index} is not numeric")
        left_value = float(left_item)
        right_value = float(right_item)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise EmbeddingError(f"cosine component {index} is not finite")
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = dot / math.sqrt(left_norm * right_norm)
    # Floating-point roundoff can escape the mathematical interval by a few ulps.
    return max(-1.0, min(1.0, similarity))
