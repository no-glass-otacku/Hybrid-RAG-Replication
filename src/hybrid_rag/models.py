from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Document:
    doc_id: str
    source_path: str
    text: str
    page: int | None = None
    section_path: str | None = None


@dataclass(slots=True)
class Chunk:
    doc_id: str
    chunk_id: str
    source_path: str
    text: str
    token_count: int
    page: int | None = None
    section_path: str | None = None


@dataclass(slots=True)
class Triple:
    subject: str
    relation: str
    object: str
    source_chunk_id: str
    evidence_text: str
    confidence: float | None = None


@dataclass(slots=True)
class RetrievedContext:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    score: float
    retrieval_method: str
    graph_node_id: str | None = None
    graph_edge_id: str | None = None
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    settings: dict[str, str | int | float]
