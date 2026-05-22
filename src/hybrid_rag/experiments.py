from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import ExperimentSettings
from .models import Document
from .pipeline import HybridRAGPipeline


class ExperimentRunner:
    def __init__(self, documents: list[Document], settings: ExperimentSettings | None = None) -> None:
        self.settings = settings or ExperimentSettings()
        self.settings.validate()
        self.pipeline = HybridRAGPipeline(documents, self.settings)

    def run(
        self,
        questions: list[str],
        method: str,
        output_dir: str | Path,
        ground_truths: list[str] | None = None,
        dataset_name: str | None = None,
        corpus_manifest: dict[str, object] | None = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for index, question in enumerate(questions):
            result = self.pipeline.query(method=method, question=question)
            row = {
                "question": result.question,
                "answer": result.answer,
                "settings": result.settings,
                "contexts": [asdict(context) for context in result.contexts],
            }
            if ground_truths is not None and index < len(ground_truths):
                row["ground_truth"] = ground_truths[index]
            results.append(row)

        result_file = output_path / f"{method}_results.json"
        settings_file = output_path / "settings.json"
        manifest_file = output_path / "experiment_manifest.json"
        dataset_file = output_path / "dataset_snapshot.json"
        result_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        settings_file.write_text(
            json.dumps(self.settings.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest_file.write_text(
            json.dumps(
                {
                    "method": method,
                    "embedding_model": self.settings.embedding_model,
                    "top_k": self.settings.vector_top_k if method in {"vector", "hybrid"} else self.settings.graph_top_k,
                    "generation": asdict(self.settings.generation),
                    "evaluation": asdict(self.settings.evaluation),
                    "graph": asdict(self.settings.graph),
                    "dataset_name": dataset_name,
                    "question_count": len(questions),
                    "has_ground_truths": ground_truths is not None,
                    "corpus_manifest": corpus_manifest,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        dataset_file.write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "questions": questions,
                    "ground_truths": ground_truths,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return result_file
