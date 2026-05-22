from __future__ import annotations

from dataclasses import asdict

from dotenv import load_dotenv

from .chunking import chunk_document
from .config import ExperimentSettings
from .embeddings import OpenAIEmbeddingProvider, TokenOverlapEmbeddingProvider
from .graph_backend import InMemoryGraphStore, Neo4jGraphStore
from .models import Document, QueryResult, RetrievedContext
from .retrievers import GraphRetriever, HybridRetriever, VectorRetriever
from .triples import extract_triples


def _extractive_answer(question: str, contexts: list[RetrievedContext]) -> str:
    if not contexts:
        return "No supporting context retrieved."
    question_terms = set(question.lower().split())
    best_sentence = contexts[0].text
    best_score = -1
    for context in contexts:
        for sentence in context.text.split(". "):
            score = len(question_terms & set(sentence.lower().split()))
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
    return best_sentence if best_sentence.endswith(".") else f"{best_sentence}."


class HybridRAGPipeline:
    def __init__(self, documents: list[Document], settings: ExperimentSettings | None = None) -> None:
        load_dotenv()
        self.settings = settings or ExperimentSettings()
        self.settings.validate()
        self.embedding_provider = self._build_embedding_provider()
        self.vector_chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document,
                chunk_size=self.settings.vector_index.chunk_size,
                chunk_overlap=self.settings.vector_index.chunk_overlap,
            )
        ]
        self.graph_chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document,
                chunk_size=self.settings.graph_index.chunk_size,
                chunk_overlap=self.settings.graph_index.chunk_overlap,
            )
        ]
        self.hybrid_chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document,
                chunk_size=self.settings.hybrid_index.chunk_size,
                chunk_overlap=self.settings.hybrid_index.chunk_overlap,
            )
        ]
        self.chunks = self.hybrid_chunks
        use_llm_builder = self.settings.graph.builder_name == "neo4j_llm_graph_builder"
        self.graph_triples = extract_triples(
            self.graph_chunks,
            graph_llm=self.settings.graph.graph_llm,
            max_paths_per_chunk=self.settings.graph.max_paths_per_chunk,
            use_llm=use_llm_builder,
        )
        self.hybrid_triples = extract_triples(
            self.hybrid_chunks,
            graph_llm=self.settings.graph.graph_llm,
            max_paths_per_chunk=self.settings.graph.max_paths_per_chunk,
            use_llm=use_llm_builder,
        )
        self.graph_store = self._build_graph_store(self.graph_triples, method="graph")
        self.hybrid_graph_store = self._build_graph_store(self.hybrid_triples, method="hybrid")
        self.vector_retriever = VectorRetriever(self.vector_chunks, self.embedding_provider)
        self.graph_retriever = GraphRetriever(self.graph_chunks, self.graph_triples, self.graph_store)
        self.hybrid_retriever = HybridRetriever(
            self.hybrid_chunks,
            self.hybrid_triples,
            self.embedding_provider,
            self.hybrid_graph_store,
        )

    def _build_embedding_provider(self) -> OpenAIEmbeddingProvider | TokenOverlapEmbeddingProvider:
        runtime = self.settings.embedding_runtime
        if runtime.provider == "openai":
            return OpenAIEmbeddingProvider(self.settings.embedding_model, runtime)
        return TokenOverlapEmbeddingProvider()

    def _build_graph_store(self, triples, method: str):
        runtime = self.settings.graph_runtime if method == "graph" else self.settings.hybrid_graph_runtime
        if runtime.backend == "neo4j":
            store = Neo4jGraphStore(
                uri_env=runtime.uri_env,
                username_env=runtime.username_env,
                password_env=runtime.password_env,
                database_env=runtime.database_env,
                query_llm_model=self.settings.graph.query_llm,
            )
            chunks = self.graph_chunks if method == "graph" else self.hybrid_chunks
            store.upsert_triples(
                triples,
                chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
                embedding_provider=self.embedding_provider,
            )
            return store
        return InMemoryGraphStore(triples)

    def query(self, method: str, question: str) -> QueryResult:
        if method == "vector":
            contexts = self.vector_retriever.retrieve(question, self.settings.vector_top_k)
        elif method == "graph":
            contexts = self.graph_retriever.retrieve(
                question,
                top_k=self.settings.graph_top_k,
                depth=self.settings.graph.hop_depth,
            )
        elif method == "hybrid":
            contexts = self.hybrid_retriever.retrieve(
                question,
                vector_top_k=self.settings.vector_top_k,
                top_k=self.settings.hybrid_top_k,
                depth=self.settings.graph.hop_depth,
            )
        else:
            raise ValueError(f"Unsupported retrieval method: {method}")

        answer = _extractive_answer(question, contexts)
        return QueryResult(
            question=question,
            answer=answer,
            contexts=contexts,
            settings=asdict(self.settings),
        )
