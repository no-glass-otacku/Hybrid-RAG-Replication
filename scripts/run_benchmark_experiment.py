from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

from hybrid_rag import (
    ExperimentRunner,
    ExperimentSettings,
    build_corpus_manifest,
    load_benchmark,
    load_documents_from_pdf_directory,
)


def main() -> None:
    load_dotenv()
    method = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    benchmark = load_benchmark(REPO_ROOT / "eval_questions" / "benchmark.json")
    documents = load_documents_from_pdf_directory(REPO_ROOT / "papers_for_questions")
    corpus_manifest = build_corpus_manifest(documents)
    settings = ExperimentSettings()
    if method in {"vector", "hybrid"}:
        settings.embedding_runtime.provider = "openai"
    if method == "graph":
        settings.graph_runtime.backend = "neo4j"
    elif method == "hybrid":
        settings.hybrid_graph_runtime.backend = "neo4j"
    runner = ExperimentRunner(documents=documents, settings=settings)
    output_dir = REPO_ROOT / "artifacts" / "benchmark" / method
    output = runner.run(
        questions=benchmark["questions"][:limit],
        ground_truths=benchmark.get("ground_truths", [])[:limit],
        dataset_name="eval_questions/benchmark.json",
        corpus_manifest=corpus_manifest,
        method=method,
        output_dir=output_dir,
    )
    print(output)


if __name__ == "__main__":
    main()
