from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from scipy.stats import ttest_rel

from .config import ExperimentSettings


def _normalize(text: str) -> set[str]:
    return {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()}


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(slots=True)
class EvaluationReport:
    method: str
    sample_count: int
    metric_means: dict[str, float]
    per_question_metrics: list[dict[str, float]]


class EvaluationRunner:
    def __init__(self, settings: ExperimentSettings) -> None:
        self.settings = settings

    def _score_row(self, row: dict[str, object]) -> dict[str, float]:
        answer = str(row.get("answer", ""))
        ground_truth = str(row.get("ground_truth", ""))
        question = str(row.get("question", ""))
        contexts = row.get("contexts", [])
        context_text = " ".join(str(context.get("text", "")) for context in contexts if isinstance(context, dict))

        answer_tokens = _normalize(answer)
        ground_truth_tokens = _normalize(ground_truth)
        context_tokens = _normalize(context_text)
        question_tokens = _normalize(question)

        overlap_context = answer_tokens & context_tokens
        overlap_answer = answer_tokens & ground_truth_tokens
        overlap_question = answer_tokens & question_tokens
        context_hits = ground_truth_tokens & context_tokens

        precision = _safe_divide(len(context_hits), len(context_tokens))
        recall = _safe_divide(len(context_hits), len(ground_truth_tokens))
        faithfulness = _safe_divide(len(overlap_context), len(answer_tokens))
        answer_relevancy = _safe_divide(len(overlap_question), len(question_tokens))
        if answer_tokens or ground_truth_tokens:
            answer_correctness = _safe_divide(
                2 * len(overlap_answer),
                len(answer_tokens) + len(ground_truth_tokens),
            )
        else:
            answer_correctness = 0.0
        answer_similarity = _safe_divide(len(overlap_answer), len(answer_tokens | ground_truth_tokens))

        return {
            "context_precision": precision,
            "context_recall": recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity,
        }

    def evaluate_results_file(self, results_file: str | Path, output_dir: str | Path) -> Path:
        rows = json.loads(Path(results_file).read_text(encoding="utf-8"))
        per_question = [self._score_row(row) for row in rows]
        metric_names = list(per_question[0].keys()) if per_question else []
        report = EvaluationReport(
            method=Path(results_file).stem.replace("_results", ""),
            sample_count=len(per_question),
            metric_means={metric: mean(item[metric] for item in per_question) for metric in metric_names},
            per_question_metrics=per_question,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / f"{report.method}_evaluation.json"
        report_file.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
        return report_file

    def compare_results_files(
        self,
        baseline_results_file: str | Path,
        candidate_results_file: str | Path,
        output_dir: str | Path,
    ) -> Path:
        baseline_rows = json.loads(Path(baseline_results_file).read_text(encoding="utf-8"))
        candidate_rows = json.loads(Path(candidate_results_file).read_text(encoding="utf-8"))
        baseline_scores = [self._score_row(row) for row in baseline_rows]
        candidate_scores = [self._score_row(row) for row in candidate_rows]
        if len(baseline_scores) != len(candidate_scores):
            raise ValueError("Paired comparison requires equal numbers of examples.")

        metric_names = list(baseline_scores[0].keys()) if baseline_scores else []
        comparison = {
            "baseline_method": Path(baseline_results_file).stem.replace("_results", ""),
            "candidate_method": Path(candidate_results_file).stem.replace("_results", ""),
            "significance_level": self.settings.evaluation.significance_level,
            "metrics": {},
        }
        for metric in metric_names:
            baseline_values = [row[metric] for row in baseline_scores]
            candidate_values = [row[metric] for row in candidate_scores]
            if len(candidate_values) < 2:
                statistic = 0.0
                p_value = 1.0
                is_significant = False
                note = "paired_t_test_requires_at_least_two_examples"
            else:
                statistic, p_value = ttest_rel(candidate_values, baseline_values)
                is_significant = bool(p_value < self.settings.evaluation.significance_level) if p_value == p_value else False
                note = ""
            comparison["metrics"][metric] = {
                "baseline_mean": mean(baseline_values),
                "candidate_mean": mean(candidate_values),
                "mean_delta": mean(candidate_values) - mean(baseline_values),
                "t_statistic": float(statistic) if statistic == statistic else 0.0,
                "p_value": float(p_value) if p_value == p_value else 1.0,
                "is_significant": is_significant,
                "note": note,
            }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / "paired_t_test.json"
        report_file.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_file
