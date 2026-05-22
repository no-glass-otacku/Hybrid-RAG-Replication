from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class IndexSettings:
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class GenerationSettings:
    llm_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 1024
    prompt_template: str = (
        "Answer the question using only the provided context.\n\n"
        "Question: {q}\n"
        "Context: {context}"
    )


@dataclass(slots=True)
class EvaluationSettings:
    evaluator_name: str = "gpt-4o"
    framework: str = "ragas"
    significance_level: float = 0.05
    statistical_test: str = "paired_t_test"
    use_offline_fallback: bool = True


@dataclass(slots=True)
class EmbeddingRuntimeSettings:
    provider: str = "token_overlap"
    cache_dir: str = "artifacts/cache/embeddings"
    batch_size: int = 16
    max_retries: int = 6
    retry_base_seconds: float = 2.0
    min_seconds_between_requests: float = 0.25


@dataclass(slots=True)
class GraphRuntimeSettings:
    backend: str = "in_memory"
    uri_env: str = "NEO4J_URI_GRAPH"
    username_env: str = "NEO4J_USERNAME_GRAPH"
    password_env: str = "NEO4J_PASSWORD_GRAPH"
    database_env: str = "NEO4J_DATABASE_GRAPH"


@dataclass(slots=True)
class HybridGraphRuntimeSettings:
    backend: str = "in_memory"
    uri_env: str = "NEO4J_URI_HYBRID"
    username_env: str = "NEO4J_USERNAME_HYBRID"
    password_env: str = "NEO4J_PASSWORD_HYBRID"
    database_env: str = "NEO4J_DATABASE_HYBRID"


@dataclass(slots=True)
class GraphSettings:
    builder_name: str = "neo4j_llm_graph_builder"
    graph_llm: str = "gpt-3.5-turbo"
    query_llm: str = "gpt-3.5-turbo"
    database: str = "neo4j"
    hop_depth: int = 1
    max_paths_per_chunk: int = 10


@dataclass(slots=True)
class ExperimentSettings:
    embedding_model: str = "text-embedding-3-large"
    embedding_runtime: EmbeddingRuntimeSettings = field(default_factory=EmbeddingRuntimeSettings)
    vector_index: IndexSettings = field(default_factory=lambda: IndexSettings(chunk_size=512, chunk_overlap=0))
    graph_index: IndexSettings = field(default_factory=lambda: IndexSettings(chunk_size=1024, chunk_overlap=204))
    hybrid_index: IndexSettings = field(default_factory=lambda: IndexSettings(chunk_size=1024, chunk_overlap=204))
    vector_top_k: int = 4
    graph_top_k: int = 4
    hybrid_top_k: int = 4
    vector_database: str = "chromadb"
    vector_index_type: str = "cosine_similarity"
    graph: GraphSettings = field(default_factory=GraphSettings)
    graph_runtime: GraphRuntimeSettings = field(default_factory=GraphRuntimeSettings)
    hybrid_graph_runtime: HybridGraphRuntimeSettings = field(default_factory=HybridGraphRuntimeSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    def validate(self) -> None:
        if self.vector_top_k != self.hybrid_top_k:
            raise ValueError("Vector RAG and Hybrid RAG must use the same top_k for fair comparison.")
        if self.graph_index.chunk_size != self.hybrid_index.chunk_size:
            raise ValueError("Graph RAG and Hybrid RAG must use the same chunk_size.")
        if self.graph_index.chunk_overlap != self.hybrid_index.chunk_overlap:
            raise ValueError("Graph RAG and Hybrid RAG must use the same chunk_overlap.")
        if self.generation.temperature != 0.0:
            raise ValueError("temperature must remain 0.0 for reproducible comparisons.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
