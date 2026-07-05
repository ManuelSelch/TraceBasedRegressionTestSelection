from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.experiments import run_experiments, run_selection_pipeline
from src.visualize_function_web import export

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "data" / "model"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
OUT_DIR = PROJECT_ROOT / "out"



def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)



def main() -> None:
    ecus = load_yaml(MODEL_DIR / "ecus.yaml")["ecus"]
    signals = load_yaml(MODEL_DIR / "signals.yaml")["signals"]
    keywords = load_yaml(MODEL_DIR / "keywords.yaml")["keywords"]
    scenarios = load_yaml(MODEL_DIR / "scenarios.yaml")["scenarios"]
    changes = load_yaml(MODEL_DIR / "changes.yaml")["changes"]
    test_cases = load_yaml(BENCHMARK_DIR / "test_cases.yaml")["test_cases"]
    experiments = load_yaml(BENCHMARK_DIR / "experiments.yaml")["experiments"]

    baseline_pipeline = run_selection_pipeline(
        ecus,
        signals,
        keywords,
        scenarios,
        changes,
        test_cases,
    )
    function_web = baseline_pipeline["function_web"]
    keyword_traces = baseline_pipeline["keyword_traces"]
    test_case_traces = baseline_pipeline["test_case_traces"]
    ecu_to_test_cases = baseline_pipeline["ecu_to_test_cases"]
    selection_results = baseline_pipeline["selection_results"]

    experiment_results = run_experiments(
        ecus,
        signals,
        keywords,
        scenarios,
        changes,
        test_cases,
        experiments,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "keyword_traces.json": keyword_traces,
        "test_case_traces.json": test_case_traces,
        "ecu_to_test_cases.json": ecu_to_test_cases,
        "selection_results.json": selection_results,
        "experiment_results.json": experiment_results,
    }

    for filename, content in outputs.items():
        output_path = OUT_DIR / filename
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(content, file, indent=2)

    print("\nSelection summary:")
    max_change_id_len = max(len(change_id) for change_id in selection_results)
    for change_id, result in selection_results.items():
        print(
            f"- {change_id:<{max_change_id_len}}  "
            f"{result['selected_count']:>2}/{result['all_test_cases']:<2} selected  "
            f"reduction: {result['reduction_rate']:.2%}"
        )

    print("\nExperiment summary:")
    experiment_summaries = [
        result["summary"] for result in experiment_results["experiments"].values()
    ]
    max_experiment_id_len = max(len(summary["experiment"]) for summary in experiment_summaries)
    max_worst_change_len = max(
        len(summary["worst_change"] or "-") for summary in experiment_summaries
    )
    print(
        f"  {'experiment':<{max_experiment_id_len}}  "
        f"{'avg recall':>10}  "
        f"{'missed tests':>12}  "
        f"{'worst change':<{max_worst_change_len}}"
    )
    for summary in experiment_summaries:
        worst_change = summary["worst_change"] or "-"
        print(
            f"- {summary['experiment']:<{max_experiment_id_len}}  "
            f"{summary['avg_test_case_recall']:>10.2%}  "
            f"{summary['total_missed_affected_test_cases']:>12}  "
            f"{worst_change:<{max_worst_change_len}}"
        )

    export(function_web)

if __name__ == "__main__":
    main()
