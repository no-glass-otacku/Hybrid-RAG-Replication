from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hybrid_rag.config import ExperimentSettings
from hybrid_rag.experiments import ExperimentRunner
from hybrid_rag.loaders import load_benchmark
from hybrid_rag.models import Document


def main() -> None:
    benchmark = load_benchmark(REPO_ROOT / "eval_questions" / "benchmark.json")
    documents = [
        Document(
            doc_id="bert",
            source_path="papers_for_questions/bert.pdf",
            text=(
                "BERT is pre-trained on masked language modeling and next sentence prediction. "
                "BERTBASE uses 12 layers, hidden size 768, 12 attention heads, and 110M parameters. "
                "BERTLARGE uses 24 layers, hidden size 1024, 16 attention heads, and 340M parameters."
            ),
        ),
        Document(
            doc_id="llama",
            source_path="papers_for_questions/llama.pdf",
            text=(
                "LLaMA uses RMSNorm, SwiGLU, and rotary embeddings. "
                "LLaMA models include 7B, 13B, 33B, and 65B parameter variants. "
                "LLaMA is trained on publicly available data."
            ),
        ),
    ]
    settings = ExperimentSettings(vector_top_k=4, graph_top_k=4, hybrid_top_k=4)
    runner = ExperimentRunner(documents=documents, settings=settings)
    questions = benchmark["questions"][:3]
    output = runner.run(
        questions=questions,
        ground_truths=benchmark.get("ground_truths", [])[:3],
        dataset_name="eval_questions/benchmark.json",
        method="hybrid",
        output_dir=REPO_ROOT / "artifacts" / "smoke",
    )
    print(output)


if __name__ == "__main__":
    main()
