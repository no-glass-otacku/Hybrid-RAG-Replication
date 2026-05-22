from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Protocol

from .config import EmbeddingRuntimeSettings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9\-\+]+", text.lower())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class TokenOverlapEmbeddingProvider:
    model_name = "token-overlap"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        counts = Counter(_tokenize(text))
        vector = [0.0] * self.dimensions
        for token, count in counts.items():
            vector[hash(token) % self.dimensions] += float(count)
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        runtime: EmbeddingRuntimeSettings,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.runtime = runtime
        self.cache_dir = Path(runtime.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._last_request_at = 0.0

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model_name}\n{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [float(value) for value in payload["embedding"]]

    def _write_cache(self, text: str, embedding: list[float]) -> None:
        path = self._cache_path(text)
        path.write_text(
            json.dumps({"model": self.model_name, "embedding": embedding}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        wait_for = self.runtime.min_seconds_between_requests - elapsed
        if wait_for > 0:
            time.sleep(wait_for)

    def _request_batch(self, batch: list[str]) -> list[list[float]]:
        from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

        last_error: Exception | None = None
        for attempt in range(self.runtime.max_retries):
            try:
                self._throttle()
                response = self._client.embeddings.create(model=self.model_name, input=batch)
                self._last_request_at = time.time()
                return [[float(value) for value in item.embedding] for item in response.data]
            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
                last_error = exc
                delay = self.runtime.retry_base_seconds * (2**attempt)
                time.sleep(delay)
        raise RuntimeError(f"OpenAI embedding request failed after retries: {last_error}") from last_error

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for index, text in enumerate(texts):
            cached = self._read_cache(text)
            if cached is not None:
                embeddings[index] = cached
            else:
                uncached_indices.append(index)
                uncached_texts.append(text)

        for start in range(0, len(uncached_texts), self.runtime.batch_size):
            batch = uncached_texts[start : start + self.runtime.batch_size]
            batch_embeddings = self._request_batch(batch)
            for text, embedding, index in zip(
                batch,
                batch_embeddings,
                uncached_indices[start : start + self.runtime.batch_size],
            ):
                self._write_cache(text, embedding)
                embeddings[index] = embedding

        return [embedding or [] for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        cached = self._read_cache(text)
        if cached is not None:
            return cached
        return self.embed_texts([text])[0]
