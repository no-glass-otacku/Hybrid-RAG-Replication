from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Protocol, Sequence

from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from .embeddings import EmbeddingProvider
from .models import Chunk, RetrievedContext, Triple

_CYPHER_QUERY_TEMPLATE = PromptTemplate(
    """You generate read-only Neo4j Cypher for the provided schema.
Return only a Cypher query with no prose and no code fences.

Requirements:
- Use only MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, CALL db.index.vector.queryNodes.
- Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD CSV, APOC write procedures, or CALL procedures except db.index.vector.queryNodes.
- The query must answer the user's question using the graph schema.
- The query must return these exact aliases when available:
  start_name, relation, end_name, source_chunk_id, evidence_text
- The query must limit results to at most {top_k}.

Schema:
{schema}

Question:
{question}

Cypher:"""
)


class GraphStore(Protocol):
    def upsert_triples(
        self,
        triples: list[Triple],
        chunks_by_id: dict[str, Chunk] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        ...

    def get_relationship_contexts(
        self,
        seed_entities: list[str],
        chunks_by_id: dict[str, Chunk],
        top_k: int,
        depth: int,
    ) -> list[RetrievedContext]:
        ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9\-\+]+", text.lower())


class InMemoryGraphStore:
    def __init__(self, triples: list[Triple] | None = None) -> None:
        self._triples = triples or []
        self._adjacency: dict[str, list[tuple[str, Triple]]] = defaultdict(list)
        if triples:
            self.upsert_triples(triples)

    def upsert_triples(
        self,
        triples: list[Triple],
        chunks_by_id: dict[str, Chunk] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if not self._triples:
            self._triples = []
        start_index = len(self._triples)
        self._triples.extend(triples)
        for offset, triple in enumerate(triples):
            edge_id = f"edge-{start_index + offset}"
            self._adjacency[triple.subject.lower()].append((edge_id, triple))
            self._adjacency[triple.object.lower()].append((edge_id, triple))

    def get_relationship_contexts(
        self,
        seed_entities: list[str],
        chunks_by_id: dict[str, Chunk],
        top_k: int,
        depth: int,
    ) -> list[RetrievedContext]:
        lowered = [entity.lower() for entity in seed_entities if entity]
        queue = deque((entity, 0) for entity in lowered)
        visited_entities: set[str] = set()
        scored_contexts: dict[str, RetrievedContext] = {}

        while queue:
            entity, current_depth = queue.popleft()
            if entity in visited_entities or current_depth > depth:
                continue
            visited_entities.add(entity)
            for edge_id, triple in self._adjacency.get(entity, []):
                chunk = chunks_by_id[triple.source_chunk_id]
                score = 1.0 / (current_depth + 1)
                existing = scored_contexts.get(chunk.chunk_id)
                context = RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_path=chunk.source_path,
                    text=chunk.text,
                    score=score,
                    retrieval_method="graph",
                    graph_node_id=entity,
                    graph_edge_id=edge_id,
                    metadata={
                        "token_count": chunk.token_count,
                        "page": chunk.page,
                        "evidence_text": triple.evidence_text,
                        "triple_subject": triple.subject,
                        "triple_relation": triple.relation,
                        "triple_object": triple.object,
                    },
                )
                if existing is None or context.score > existing.score:
                    scored_contexts[chunk.chunk_id] = context
                if current_depth < depth:
                    queue.append((triple.subject.lower(), current_depth + 1))
                    queue.append((triple.object.lower(), current_depth + 1))
        return sorted(scored_contexts.values(), key=lambda item: item.score, reverse=True)[:top_k]


class Neo4jGraphStore:
    def __init__(
        self,
        uri_env: str,
        username_env: str,
        password_env: str,
        database_env: str,
        query_llm_model: str = "gpt-3.5-turbo",
    ) -> None:
        import os

        uri = os.getenv(uri_env)
        username = os.getenv(username_env)
        password = os.getenv(password_env)
        database = os.getenv(database_env, "neo4j")
        if not uri or not username or not password:
            raise ValueError(
                f"Neo4j configuration missing. Expected env vars: {uri_env}, {username_env}, {password_env}."
            )
        self.database = database
        self._query_llm_model = query_llm_model
        self._property_graph = Neo4jPropertyGraphStore(
            username=username,
            password=password,
            url=uri,
            database=database,
            refresh_schema=False,
        )

    @property
    def _driver(self):
        return self._property_graph.client

    def close(self) -> None:
        self._property_graph.close()

    def upsert_triples(
        self,
        triples: list[Triple],
        chunks_by_id: dict[str, Chunk] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if chunks_by_id is None or embedding_provider is None or not triples:
            return

        chunk_ids = sorted({triple.source_chunk_id for triple in triples if triple.source_chunk_id in chunks_by_id})
        chunk_texts = [chunks_by_id[chunk_id].text for chunk_id in chunk_ids]
        chunk_embeddings = embedding_provider.embed_texts(chunk_texts)
        chunk_nodes = [
            TextNode(
                id_=chunk_id,
                text=chunks_by_id[chunk_id].text,
                metadata={
                    "doc_id": chunks_by_id[chunk_id].doc_id,
                    "source_path": chunks_by_id[chunk_id].source_path,
                    "page": chunks_by_id[chunk_id].page,
                    "section_path": chunks_by_id[chunk_id].section_path,
                },
                embedding=embedding,
            )
            for chunk_id, embedding in zip(chunk_ids, chunk_embeddings)
        ]
        self._property_graph.upsert_llama_nodes(chunk_nodes)

        entity_descriptions: dict[str, set[str]] = defaultdict(set)
        for triple in triples:
            entity_descriptions[triple.subject].add(triple.evidence_text)
            entity_descriptions[triple.object].add(triple.evidence_text)

        entity_names = sorted(entity_descriptions)
        entity_texts = [
            f"Entity: {name}\nEvidence: {' '.join(sorted(entity_descriptions[name]))}"
            for name in entity_names
        ]
        entity_embeddings = embedding_provider.embed_texts(entity_texts)
        entity_nodes = [
            EntityNode(
                name=name,
                label="entity",
                properties={"description": text},
                embedding=embedding,
            )
            for name, text, embedding in zip(entity_names, entity_texts, entity_embeddings)
        ]
        self._property_graph.upsert_nodes(entity_nodes)

        relations: list[Relation] = []
        for triple in triples:
            chunk = chunks_by_id[triple.source_chunk_id]
            relations.append(
                Relation(
                    label=self._normalize_relation_label(triple.relation),
                    source_id=triple.subject,
                    target_id=triple.object,
                    properties={
                        "source_chunk_id": triple.source_chunk_id,
                        "evidence_text": triple.evidence_text,
                        "confidence": triple.confidence,
                        "doc_id": chunk.doc_id,
                        "source_path": chunk.source_path,
                        "page": chunk.page,
                    },
                )
            )
        self._property_graph.upsert_relations(relations)
        self._property_graph.get_schema(refresh=True)

    def get_relationship_contexts(
        self,
        seed_entities: list[str],
        chunks_by_id: dict[str, Chunk],
        top_k: int,
        depth: int,
    ) -> list[RetrievedContext]:
        return self._contexts_from_seed_entities(seed_entities, chunks_by_id, top_k, depth)

    def query_graph(
        self,
        question: str,
        chunks_by_id: dict[str, Chunk],
        top_k: int,
    ) -> list[RetrievedContext]:
        llm = LlamaIndexOpenAI(model=self._query_llm_model, temperature=0.0)
        schema = self._property_graph.get_schema_str(refresh=True)
        raw_query = llm.predict(
            _CYPHER_QUERY_TEMPLATE,
            schema=schema,
            question=question,
            top_k=top_k,
        )
        cypher = self._sanitize_cypher(raw_query)
        rows = self._property_graph.structured_query(cypher)
        return self._contexts_from_cypher_rows(rows, chunks_by_id, top_k)

    def query_hybrid(
        self,
        question: str,
        chunks_by_id: dict[str, Chunk],
        embedding_provider: EmbeddingProvider,
        top_k: int,
        depth: int,
    ) -> list[RetrievedContext]:
        query_embedding = embedding_provider.embed_query(question)
        kg_nodes, scores = self._property_graph.vector_query(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
            )
        )
        if not kg_nodes:
            return []

        score_by_node_id = {node.id: float(score) for node, score in zip(kg_nodes, scores)}
        triplets = self._property_graph.get_rel_map(
            kg_nodes,
            depth=depth,
            limit=max(top_k * 5, top_k),
            ignore_rels=["MENTIONS"],
        )
        return self._contexts_from_triplets(
            triplets=triplets,
            chunks_by_id=chunks_by_id,
            default_method="hybrid",
            top_k=top_k,
            score_by_node_id=score_by_node_id,
        )

    def _contexts_from_seed_entities(
        self,
        seed_entities: Sequence[str],
        chunks_by_id: dict[str, Chunk],
        top_k: int,
        depth: int,
    ) -> list[RetrievedContext]:
        entity_names = [entity for entity in seed_entities if entity]
        if not entity_names:
            return []
        kg_nodes = self._property_graph.get(ids=entity_names)
        if not kg_nodes:
            return []
        triplets = self._property_graph.get_rel_map(
            kg_nodes,
            depth=depth,
            limit=max(top_k * 5, top_k),
            ignore_rels=["MENTIONS"],
        )
        score_by_node_id = {node.id: 1.0 for node in kg_nodes}
        return self._contexts_from_triplets(
            triplets=triplets,
            chunks_by_id=chunks_by_id,
            default_method="graph",
            top_k=top_k,
            score_by_node_id=score_by_node_id,
        )

    def _contexts_from_cypher_rows(
        self,
        rows: list[dict[str, object]],
        chunks_by_id: dict[str, Chunk],
        top_k: int,
    ) -> list[RetrievedContext]:
        contexts: dict[str, RetrievedContext] = {}
        for row in rows:
            chunk_id = str(row.get("source_chunk_id") or "")
            if chunk_id not in chunks_by_id:
                continue
            chunk = chunks_by_id[chunk_id]
            relation = str(row.get("relation") or "")
            start_name = str(row.get("start_name") or "")
            end_name = str(row.get("end_name") or "")
            context = RetrievedContext(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_path=chunk.source_path,
                text=chunk.text,
                score=1.0,
                retrieval_method="graph",
                graph_node_id=start_name.lower() or None,
                graph_edge_id=f"{start_name}::{chunk_id}::{relation}" if start_name and relation else None,
                metadata={
                    "token_count": chunk.token_count,
                    "page": chunk.page,
                    "evidence_text": str(row.get("evidence_text") or ""),
                    "triple_subject": start_name or None,
                    "triple_relation": relation or None,
                    "triple_object": end_name or None,
                },
            )
            contexts[chunk_id] = context
        return sorted(contexts.values(), key=lambda item: item.score, reverse=True)[:top_k]

    def _contexts_from_triplets(
        self,
        triplets: Sequence[Sequence[object]],
        chunks_by_id: dict[str, Chunk],
        default_method: str,
        top_k: int,
        score_by_node_id: dict[str, float],
    ) -> list[RetrievedContext]:
        contexts: dict[str, RetrievedContext] = {}
        for triplet in triplets:
            source, relation, target = triplet
            if not isinstance(source, EntityNode) or not isinstance(relation, Relation) or not isinstance(target, EntityNode):
                continue
            chunk_id = str(relation.properties.get("source_chunk_id") or "")
            if chunk_id not in chunks_by_id:
                continue
            chunk = chunks_by_id[chunk_id]
            score = max(score_by_node_id.get(source.id, 0.0), score_by_node_id.get(target.id, 0.0), 1.0 if default_method == "graph" else 0.0)
            context = RetrievedContext(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_path=chunk.source_path,
                text=chunk.text,
                score=score,
                retrieval_method=default_method,
                graph_node_id=source.id.lower(),
                graph_edge_id=f"{source.id}::{chunk_id}::{relation.label}",
                metadata={
                    "token_count": chunk.token_count,
                    "page": chunk.page,
                    "evidence_text": str(relation.properties.get("evidence_text") or ""),
                    "triple_subject": source.name,
                    "triple_relation": relation.label,
                    "triple_object": target.name,
                },
            )
            prior = contexts.get(chunk.chunk_id)
            if prior is None or context.score > prior.score:
                contexts[chunk.chunk_id] = context
        return sorted(contexts.values(), key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _normalize_relation_label(relation: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", relation.strip()).strip("_")
        return cleaned.upper() or "RELATED"

    @staticmethod
    def _sanitize_cypher(raw_query: str) -> str:
        query = raw_query.strip()
        if query.startswith("```"):
            lines = [line for line in query.splitlines() if not line.strip().startswith("```")]
            query = "\n".join(lines).strip()
        upper_query = query.upper()
        forbidden = ["CREATE ", "MERGE ", "DELETE ", "SET ", "REMOVE ", "DROP ", "LOAD CSV", "CALL APOC."]
        if any(token in upper_query for token in forbidden):
            raise ValueError(f"Unsafe Cypher generated: {query}")
        return query
