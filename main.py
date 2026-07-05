from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.graph_builder import build_function_web, generate_all_keyword_traces
from src.selectors import (
    build_ecu_to_test_cases_map,
    generate_all_test_case_traces,
    select_test_cases_for_all_changes,
)
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

    scenarios_by_id = {scenario["id"]: scenario for scenario in scenarios}

    function_web = build_function_web(ecus, signals)
    keyword_traces = {
        scenario["id"]: generate_all_keyword_traces(
            function_web,
            keywords,
            active_features=scenario.get("active_features", []),
        )
        for scenario in scenarios
    }
    test_case_traces = generate_all_test_case_traces(
        test_cases,
        scenarios_by_id,
        keyword_traces,
    )
    ecu_to_test_cases = build_ecu_to_test_cases_map(test_case_traces)
    selection_results = select_test_cases_for_all_changes(changes, test_case_traces)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "keyword_traces.json": keyword_traces,
        "test_case_traces.json": test_case_traces,
        "ecu_to_test_cases.json": ecu_to_test_cases,
        "selection_results.json": selection_results,
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

    export(function_web)

if __name__ == "__main__":
    main()
