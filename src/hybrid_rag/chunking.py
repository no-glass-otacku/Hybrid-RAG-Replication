from __future__ import annotations

from .models import Chunk, Document


def _token_count(text: str) -> int:
    return len(text.split())


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    words = document.text.split()
    if not words:
        raise ValueError(f"Document {document.doc_id} has empty text.")

    chunks: list[Chunk] = []
    step = max(1, chunk_size - chunk_overlap)

    for index, start in enumerate(range(0, len(words), step)):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        text = " ".join(chunk_words).strip()
        chunks.append(
            Chunk(
                doc_id=document.doc_id,
                chunk_id=f"{document.doc_id}-chunk-{index}",
                source_path=document.source_path,
                page=document.page,
                section_path=document.section_path,
                text=text,
                token_count=_token_count(text),
            )
        )
        if end >= len(words):
            break

    return chunks
