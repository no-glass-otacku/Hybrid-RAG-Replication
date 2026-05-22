from __future__ import annotations

import re
from collections import defaultdict

from .embeddings import EmbeddingProvider, cosine_similarity
from .graph_backend import GraphStore
from .models import Chunk, RetrievedContext, Triple


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9\-\+]+", text.lower())


class VectorRetriever:
    def __init__(self, chunks: list[Chunk], embedding_provider: EmbeddingProvider) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._embedding_provider = embedding_provider
        vectors = embedding_provider.embed_texts([chunk.text for chunk in chunks])
        self._vectors = {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}

    def retrieve(self, question: str, top_k: int) -> list[RetrievedContext]:
        query_vector = self._embedding_provider.embed_query(question)
        scored = sorted(
            (
                (
                    cosine_similarity(query_vector, vector),
                    self._chunks[chunk_id],
                )
                for chunk_id, vector in self._vectors.items()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            RetrievedContext(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_path=chunk.source_path,
                text=chunk.text,
                score=score,
                retrieval_method="vector",
                metadata={"token_count": chunk.token_count, "page": chunk.page},
            )
            for score, chunk in scored[:top_k]
            if score > 0
        ]


class GraphRetriever:
    def __init__(self, chunks: list[Chunk], triples: list[Triple], graph_store: GraphStore) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._triples = triples
        self._graph_store = graph_store
        self._entities_by_chunk: dict[str, set[str]] = defaultdict(set)

        for triple in triples:
            self._entities_by_chunk[triple.source_chunk_id].update(
                {triple.subject.lower(), triple.object.lower()}
            )

    def retrieve(self, question: str, top_k: int, depth: int) -> list[RetrievedContext]:
        if hasattr(self._graph_store, "query_graph"):
            return self._graph_store.query_graph(question, self._chunks, top_k)
        tokens = set(_tokenize(question))
        return self._graph_store.get_relationship_contexts(
            seed_entities=sorted(tokens),
            chunks_by_id=self._chunks,
            top_k=top_k,
            depth=depth,
        )


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], triples: list[Triple], embedding_provider: EmbeddingProvider, graph_store: GraphStore) -> None:
        self._vector = VectorRetriever(chunks, embedding_provider)
        self._graph = GraphRetriever(chunks, triples, graph_store)
        self._chunk_entities: dict[str, set[str]] = defaultdict(set)
        for triple in triples:
            self._chunk_entities[triple.source_chunk_id].update(
                {triple.subject.lower(), triple.object.lower()}
            )

    def retrieve(self, question: str, vector_top_k: int, top_k: int, depth: int) -> list[RetrievedContext]:
        if hasattr(self._graph, "_graph_store") and hasattr(self._graph._graph_store, "query_hybrid"):
            return self._graph._graph_store.query_hybrid(
                question=question,
                chunks_by_id=self._graph._chunks,
                embedding_provider=self._vector._embedding_provider,
                top_k=top_k,
                depth=depth,
            )
        vector_results = self._vector.retrieve(question, vector_top_k)
        seed_tokens = set()
        for context in vector_results:
            seed_tokens.update(self._chunk_entities.get(context.chunk_id, set()))

        graph_results = self._graph.retrieve(" ".join(sorted(seed_tokens)) or question, top_k, depth)

        merged: dict[str, RetrievedContext] = {item.chunk_id: item for item in vector_results}
        for item in graph_results:
            if item.chunk_id in merged:
                prior = merged[item.chunk_id]
                prior.score = max(prior.score, item.score)
                prior.retrieval_method = "hybrid"
                prior.graph_edge_id = item.graph_edge_id
                prior.graph_node_id = item.graph_node_id
                prior.metadata.update(item.metadata)
            else:
                item.retrieval_method = "hybrid"
                merged[item.chunk_id] = item

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
