from __future__ import annotations

import re
from typing import Any

from llama_index.core.graph_stores.types import KG_RELATIONS_KEY
from llama_index.core.indices.property_graph.transformations import SimpleLLMPathExtractor
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from .models import Chunk, Triple

_PATTERNS = [
    re.compile(
        r"(?P<subject>[A-Z][A-Za-z0-9\-\+ ]+?)\s+is\s+(?P<object>[A-Za-z0-9\-\+ ,]+?)(?:[.;]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<subject>[A-Z][A-Za-z0-9\-\+ ]+?)\s+uses\s+(?P<object>[A-Za-z0-9\-\+ ,]+?)(?:[.;]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<subject>[A-Z][A-Za-z0-9\-\+ ]+?)\s+includes\s+(?P<object>[A-Za-z0-9\-\+ ,]+?)(?:[.;]|$)",
        re.IGNORECASE,
    ),
]


def _extract_triples_with_regex(chunks: list[Chunk]) -> list[Triple]:
    triples: list[Triple] = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            evidence = sentence.strip()
            if not evidence:
                continue
            for pattern in _PATTERNS:
                match = pattern.search(evidence)
                if not match:
                    continue
                relation = pattern.pattern.split(r"\s+")[1].strip("\\")
                triples.append(
                    Triple(
                        subject=match.group("subject").strip(),
                        relation=relation,
                        object=match.group("object").strip(),
                        source_chunk_id=chunk.chunk_id,
                        evidence_text=evidence,
                        confidence=0.5,
                    )
                )
                break
    return triples


def _deduplicate_triples(triples: list[Triple]) -> list[Triple]:
    seen: set[tuple[str, str, str, str]] = set()
    deduplicated: list[Triple] = []
    for triple in triples:
        key = (
            triple.subject.strip().lower(),
            triple.relation.strip().lower(),
            triple.object.strip().lower(),
            triple.source_chunk_id,
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(triple)
    return deduplicated


def _extract_triples_with_llm(
    chunks: list[Chunk],
    graph_llm: str,
    max_paths_per_chunk: int,
) -> list[Triple]:
    llm = LlamaIndexOpenAI(model=graph_llm, temperature=0.0)
    extractor = SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=max_paths_per_chunk,
        num_workers=4,
    )
    nodes = [
        TextNode(
            id_=chunk.chunk_id,
            text=chunk.text,
            metadata={
                "doc_id": chunk.doc_id,
                "source_path": chunk.source_path,
                "page": chunk.page,
                "section_path": chunk.section_path,
            },
        )
        for chunk in chunks
    ]
    extracted_nodes = extractor(nodes)

    triples: list[Triple] = []
    for chunk, node in zip(chunks, extracted_nodes):
        relations: list[Any] = node.metadata.get(KG_RELATIONS_KEY, [])
        for relation in relations:
            triples.append(
                Triple(
                    subject=str(relation.source_id).strip(),
                    relation=str(relation.label).strip(),
                    object=str(relation.target_id).strip(),
                    source_chunk_id=chunk.chunk_id,
                    evidence_text=chunk.text,
                    confidence=1.0,
                )
            )
    return _deduplicate_triples(triples)


def extract_triples(
    chunks: list[Chunk],
    graph_llm: str = "gpt-3.5-turbo",
    max_paths_per_chunk: int = 10,
    use_llm: bool = True,
) -> list[Triple]:
    if use_llm:
        try:
            triples = _extract_triples_with_llm(
                chunks,
                graph_llm=graph_llm,
                max_paths_per_chunk=max_paths_per_chunk,
            )
            if triples:
                return triples
        except Exception:
            pass
    return _extract_triples_with_regex(chunks)
