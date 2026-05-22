from __future__ import annotations

import json
from pathlib import Path

from llama_index.core.graph_stores.types import KG_RELATIONS_KEY, EntityNode, Relation

from hybrid_rag.config import ExperimentSettings
from hybrid_rag.evaluation import EvaluationRunner
from hybrid_rag.experiments import ExperimentRunner
from hybrid_rag.loaders import build_corpus_manifest, load_benchmark, load_documents_from_pdf_directory
from hybrid_rag.models import Document
from hybrid_rag.pipeline import HybridRAGPipeline
from hybrid_rag.triples import extract_triples


def _documents() -> list[Document]:
    return [
        Document(
            doc_id="bert",
            source_path="papers_for_questions/bert.pdf",
            text=(
                "BERT is pre-trained on masked language modeling and next sentence prediction. "
                "BERT includes BERTBASE and BERTLARGE."
            ),
        ),
        Document(
            doc_id="llama",
            source_path="papers_for_questions/llama.pdf",
            text=(
                "LLaMA uses RMSNorm and rotary embeddings. "
                "LLaMA includes 7B, 13B, 33B, and 65B models."
            ),
        ),
    ]


class _FakeNeo4jPropertyGraphStore:
    def __init__(self, *args, **kwargs) -> None:
        self.client = object()
        self.structured_queries: list[str] = []
        self.vector_queries = 0
        self.relations: list[Relation] = []

    def close(self) -> None:
        return

    def upsert_llama_nodes(self, nodes) -> None:
        self.chunk_nodes = nodes

    def upsert_nodes(self, nodes) -> None:
        self.entity_nodes = nodes

    def upsert_relations(self, relations) -> None:
        self.relations = relations

    def get_schema(self, refresh: bool = False):
        return {}

    def get_schema_str(self, refresh: bool = False) -> str:
        return "(:__Entity__)-[:USES]->(:__Entity__)"

    def structured_query(self, query: str):
        self.structured_queries.append(query)
        return [
            {
                "start_name": "BERT",
                "relation": "USES",
                "end_name": "transformer encoder",
                "source_chunk_id": "bert-chunk-0",
                "evidence_text": "BERT uses transformer encoder.",
            }
        ]

    def vector_query(self, query):
        self.vector_queries += 1
        return ([EntityNode(name="BERT", label="entity")], [0.91])

    def get_rel_map(self, graph_nodes, depth: int = 2, limit: int = 30, ignore_rels=None):
        return [
            [
                EntityNode(name="BERT", label="entity"),
                Relation(
                    label="USES",
                    source_id="BERT",
                    target_id="transformer encoder",
                    properties={
                        "source_chunk_id": "bert-chunk-0",
                        "evidence_text": "BERT uses transformer encoder.",
                    },
                ),
                EntityNode(name="transformer encoder", label="entity"),
            ]
        ]

    def get(self, ids=None, properties=None):
        ids = ids or []
        return [EntityNode(name=id_, label="entity") for id_ in ids]


class _FakeCypherLLM:
    def __init__(self, *args, **kwargs) -> None:
        return

    def predict(self, *args, **kwargs) -> str:
        return (
            "MATCH (s:__Entity__)-[r:USES]->(o:__Entity__) "
            "RETURN s.name AS start_name, type(r) AS relation, o.name AS end_name, "
            "r.source_chunk_id AS source_chunk_id, r.evidence_text AS evidence_text LIMIT 4"
        )


def test_pipeline_preserves_traceable_chunk_metadata() -> None:
    settings = ExperimentSettings()
    settings.embedding_runtime.provider = "token_overlap"
    settings.vector_index.chunk_size = 32
    settings.vector_index.chunk_overlap = 4
    settings.graph_index.chunk_size = 32
    settings.graph_index.chunk_overlap = 4
    settings.hybrid_index.chunk_size = 32
    settings.hybrid_index.chunk_overlap = 4
    pipeline = HybridRAGPipeline(_documents(), settings)

    assert pipeline.vector_chunks
    chunk = pipeline.vector_chunks[0]
    assert chunk.doc_id == "bert"
    assert chunk.chunk_id.startswith("bert-chunk-")
    assert chunk.source_path.endswith("bert.pdf")
    assert chunk.token_count > 0


def test_extract_triples_uses_llm_path_extractor(monkeypatch) -> None:
    class _FakeExtractor:
        def __init__(self, *args, **kwargs) -> None:
            return

        def __call__(self, nodes):
            nodes[0].metadata[KG_RELATIONS_KEY] = [
                Relation(label="USES", source_id="BERT", target_id="transformer encoder")
            ]
            return nodes

    monkeypatch.setattr("hybrid_rag.triples.LlamaIndexOpenAI", _FakeCypherLLM)
    monkeypatch.setattr("hybrid_rag.triples.SimpleLLMPathExtractor", _FakeExtractor)

    chunks = [
        type("ChunkLike", (), {
            "chunk_id": "bert-chunk-0",
            "doc_id": "bert",
            "source_path": "papers_for_questions/bert.pdf",
            "text": "BERT uses transformer encoder.",
            "page": 1,
            "section_path": None,
        })()
    ]

    triples = extract_triples(chunks, graph_llm="gpt-3.5-turbo", use_llm=True)

    assert len(triples) == 1
    assert triples[0].subject == "BERT"
    assert triples[0].relation == "USES"
    assert triples[0].object == "transformer encoder"


def test_graph_query_uses_neo4j_cypher_path(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI_GRAPH", "bolt://fake")
    monkeypatch.setenv("NEO4J_USERNAME_GRAPH", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD_GRAPH", "password")
    monkeypatch.setenv("NEO4J_DATABASE_GRAPH", "neo4j")
    monkeypatch.setattr("hybrid_rag.graph_backend.Neo4jPropertyGraphStore", _FakeNeo4jPropertyGraphStore)
    monkeypatch.setattr("hybrid_rag.graph_backend.LlamaIndexOpenAI", _FakeCypherLLM)
    monkeypatch.setattr(
        "hybrid_rag.pipeline.extract_triples",
        lambda chunks, graph_llm, max_paths_per_chunk, use_llm: [
            type("TripleLike", (), {
                "subject": "BERT",
                "relation": "uses",
                "object": "transformer encoder",
                "source_chunk_id": "bert-chunk-0",
                "evidence_text": "BERT uses transformer encoder.",
                "confidence": 1.0,
            })()
        ],
    )

    settings = ExperimentSettings()
    settings.embedding_runtime.provider = "token_overlap"
    settings.graph_runtime.backend = "neo4j"
    pipeline = HybridRAGPipeline(_documents(), settings)

    result = pipeline.query("graph", "What does BERT use?")

    assert type(pipeline.graph_store).__name__ == "Neo4jGraphStore"
    assert result.contexts
    assert result.contexts[0].graph_edge_id == "BERT::bert-chunk-0::USES"
    assert pipeline.graph_store._property_graph.structured_queries


def test_hybrid_query_uses_neo4j_vector_start_nodes(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI_HYBRID", "bolt://fake")
    monkeypatch.setenv("NEO4J_USERNAME_HYBRID", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD_HYBRID", "password")
    monkeypatch.setenv("NEO4J_DATABASE_HYBRID", "neo4j")
    monkeypatch.setattr("hybrid_rag.graph_backend.Neo4jPropertyGraphStore", _FakeNeo4jPropertyGraphStore)
    monkeypatch.setattr("hybrid_rag.graph_backend.LlamaIndexOpenAI", _FakeCypherLLM)
    monkeypatch.setattr(
        "hybrid_rag.pipeline.extract_triples",
        lambda chunks, graph_llm, max_paths_per_chunk, use_llm: [
            type("TripleLike", (), {
                "subject": "BERT",
                "relation": "uses",
                "object": "transformer encoder",
                "source_chunk_id": "bert-chunk-0",
                "evidence_text": "BERT uses transformer encoder.",
                "confidence": 1.0,
            })()
        ],
    )

    settings = ExperimentSettings()
    settings.embedding_runtime.provider = "token_overlap"
    settings.hybrid_graph_runtime.backend = "neo4j"
    pipeline = HybridRAGPipeline(_documents(), settings)

    result = pipeline.query("hybrid", "What does BERT use?")

    assert type(pipeline.hybrid_graph_store).__name__ == "Neo4jGraphStore"
    assert result.contexts
    assert result.contexts[0].graph_node_id == "bert"
    assert pipeline.hybrid_graph_store._property_graph.vector_queries == 1


def test_hybrid_retrieval_returns_traceable_contexts() -> None:
    settings = ExperimentSettings()
    settings.embedding_runtime.provider = "token_overlap"
    pipeline = HybridRAGPipeline(_documents(), settings)

    result = pipeline.query("hybrid", "What are the two main tasks BERT is pre-trained on?")

    assert result.contexts
    first = result.contexts[0]
    assert first.chunk_id
    assert first.doc_id == "bert"
    assert first.source_path.endswith(".pdf")
    assert first.retrieval_method in {"vector", "hybrid"}


def test_experiment_runner_writes_results_and_settings(tmp_path: Path) -> None:
    settings = ExperimentSettings()
    settings.embedding_runtime.provider = "token_overlap"
    runner = ExperimentRunner(_documents(), settings)

    output = runner.run(
        questions=["What are the two main tasks BERT is pre-trained on?"],
        ground_truths=["Masked LM and NSP."],
        dataset_name="test-benchmark",
        corpus_manifest={"document_count": 2},
        method="vector",
        output_dir=tmp_path,
    )

    settings_file = tmp_path / "settings.json"
    manifest_file = tmp_path / "experiment_manifest.json"
    dataset_file = tmp_path / "dataset_snapshot.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))

    assert output.exists()
    assert settings_file.exists()
    assert manifest_file.exists()
    assert dataset_file.exists()
    assert manifest["embedding_model"] == "text-embedding-3-large"
    assert manifest["dataset_name"] == "test-benchmark"
    assert dataset["ground_truths"] == ["Masked LM and NSP."]
    assert payload[0]["ground_truth"] == "Masked LM and NSP."
    assert payload[0]["contexts"][0]["chunk_id"].startswith("bert-chunk-")


def test_settings_enforce_fair_comparison_constraints() -> None:
    settings = ExperimentSettings()
    settings.hybrid_top_k = 3

    try:
        settings.validate()
    except ValueError as exc:
        assert "same top_k" in str(exc)
    else:
        raise AssertionError("Expected fair-comparison validation to fail.")


def test_load_benchmark_reads_question_list(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps({"questions": ["q1"], "ground_truths": ["a1"]}), encoding="utf-8")

    payload = load_benchmark(benchmark_path)

    assert payload["questions"] == ["q1"]


def test_load_documents_from_pdf_directory_uses_pdf_reader(monkeypatch, tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "sample.pdf").write_bytes(b"%PDF-1.4 fake")

    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakeReader:
        def __init__(self, path: str) -> None:
            self.pages = [_FakePage("Page one text"), _FakePage("Page two text")]

    import types
    import sys

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=_FakeReader))

    documents = load_documents_from_pdf_directory(pdf_dir)

    assert len(documents) == 2
    assert documents[0].doc_id == "sample-page-1"
    assert documents[0].source_path == "pdfs\\sample.pdf" or documents[0].source_path == "pdfs/sample.pdf"
    assert documents[1].page == 2


def test_build_corpus_manifest_summarizes_documents() -> None:
    manifest = build_corpus_manifest(_documents())

    assert manifest["document_count"] == 2
    assert manifest["source_count"] == 2
    assert manifest["sources"][0]["pages"] == 1


def test_evaluation_runner_writes_metric_report(tmp_path: Path) -> None:
    results_file = tmp_path / "vector_results.json"
    results_file.write_text(
        json.dumps(
            [
                {
                    "question": "What are the two main tasks BERT is pre-trained on?",
                    "answer": "Masked LM and next sentence prediction.",
                    "ground_truth": "Masked LM and Next Sentence Prediction.",
                    "contexts": [{"text": "BERT is pre-trained on masked language modeling and next sentence prediction."}],
                }
            ]
        ),
        encoding="utf-8",
    )
    runner = EvaluationRunner(ExperimentSettings())

    report_file = runner.evaluate_results_file(results_file, tmp_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert report["sample_count"] == 1
    assert report["metric_means"]["answer_correctness"] > 0


def test_evaluation_runner_writes_paired_t_test(tmp_path: Path) -> None:
    baseline = tmp_path / "vector_results.json"
    candidate = tmp_path / "hybrid_results.json"
    baseline.write_text(
        json.dumps(
            [
                {"question": "q1", "answer": "wrong", "ground_truth": "right", "contexts": [{"text": "right"}]},
                {"question": "q2", "answer": "bad", "ground_truth": "good", "contexts": [{"text": "good"}]},
            ]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            [
                {"question": "q1", "answer": "right", "ground_truth": "right", "contexts": [{"text": "right"}]},
                {"question": "q2", "answer": "good", "ground_truth": "good", "contexts": [{"text": "good"}]},
            ]
        ),
        encoding="utf-8",
    )
    runner = EvaluationRunner(ExperimentSettings())

    report_file = runner.compare_results_files(baseline, candidate, tmp_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert report["candidate_method"] == "hybrid"
    assert "answer_correctness" in report["metrics"]
