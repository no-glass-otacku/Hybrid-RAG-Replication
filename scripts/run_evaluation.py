from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv

from hybrid_rag import EvaluationRunner, ExperimentSettings


def main() -> None:
    load_dotenv()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_evaluation.py <results_file> [candidate_results_file]")

    settings = ExperimentSettings()
    runner = EvaluationRunner(settings)
    first = Path(sys.argv[1])
    output_dir = first.parent / "evaluation"
    if len(sys.argv) == 2:
        output = runner.evaluate_results_file(first, output_dir=output_dir)
    else:
        second = Path(sys.argv[2])
        output = runner.compare_results_files(first, second, output_dir=output_dir)
    print(output)


if __name__ == "__main__":
    main()
