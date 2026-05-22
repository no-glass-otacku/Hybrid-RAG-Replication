from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Document


def _safe_doc_id(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", path.stem).strip("-").lower()


def load_benchmark(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "questions" not in payload:
        raise ValueError("Benchmark JSON must include a 'questions' field.")
    return payload


def load_documents_from_pdf_directory(directory: str | Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pypdf is required for PDF ingestion. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    pdf_dir = Path(directory)
    documents: list[Document] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            documents.append(
                Document(
                    doc_id=f"{_safe_doc_id(pdf_path)}-page-{page_index}",
                    source_path=str(pdf_path.relative_to(pdf_dir.parent)),
                    page=page_index,
                    section_path=None,
                    text=page_text,
                )
            )

    if not documents:
        raise ValueError(f"No readable PDF text found in {pdf_dir}.")
    return documents


def build_corpus_manifest(documents: list[Document]) -> dict[str, object]:
    by_source: dict[str, dict[str, object]] = {}
    total_characters = 0
    for document in documents:
        total_characters += len(document.text)
        entry = by_source.setdefault(
            document.source_path,
            {
                "source_path": document.source_path,
                "pages": 0,
                "document_ids": [],
                "characters": 0,
            },
        )
        entry["pages"] = int(entry["pages"]) + 1
        entry["characters"] = int(entry["characters"]) + len(document.text)
        entry["document_ids"].append(document.doc_id)

    return {
        "document_count": len(documents),
        "source_count": len(by_source),
        "total_characters": total_characters,
        "sources": list(by_source.values()),
    }
