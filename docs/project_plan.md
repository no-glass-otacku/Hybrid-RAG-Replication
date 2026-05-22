# Hybrid RAG Project Plan

## Goal

Reproduce the paper's comparison structure for `vector`, `graph`, and `hybrid` retrieval while enforcing the shared controls called out in the experiment diagram.

## Controlled Parameters

### [A] Embedding Model

- `embedding_model = text-embedding-3-large`
- Must be identical for vector chunk embeddings, hybrid node embeddings, and query embeddings.

### [B] Top-K

- `top_k = 4`
- Must remain identical for `vector` and `hybrid` retrieval.

### [C] Generation

- `llm = gpt-3.5-turbo`
- `temperature = 0`
- `max_tokens = 1024`
- All three methods must share the same prompt template.

## Indexing Defaults

1. Vector RAG
- `chunk_size = 512`
- `chunk_overlap = 0`
- `vector_db = chromadb`

2. Graph RAG
- `chunk_size = 1024`
- `chunk_overlap = 204`
- `graph_db = neo4j`
- `graph_llm = gpt-3.5-turbo`

3. Hybrid RAG
- `chunk_size = 1024`
- `chunk_overlap = 204`
- `graph_db = neo4j`
- `graph_llm = gpt-3.5-turbo`
- Node embeddings must use the same model as Vector RAG.

## Current Implementation Status

1. Method-specific indexing settings are now modeled separately in code.
2. Fair-comparison validation now rejects mismatched `top_k`, graph/hybrid chunk settings, or non-deterministic generation temperature.
3. Experiment runs now save a manifest with retrieval, generation, and evaluation settings.
4. PDF ingestion and benchmark loading are now part of the codebase, so repository documents can be used as experiment inputs.
5. Retrieval remains offline and heuristic for now, so the next step is backend integration rather than more scaffolding.

## Next Milestones

1. Replace heuristic embedding similarity with `text-embedding-3-large` integration and persistent vector storage.
2. Replace heuristic triple extraction with Neo4j graph-builder backed extraction while preserving `evidence_text`.
3. Add Cypher-query based Graph RAG retrieval instead of token overlap traversal.
4. Add RAGAS evaluation runner and paired t-test reporting over stored run outputs.
5. Add a document manifest so the exact 13 benchmark papers and any extra comparison corpus are explicitly versioned.
