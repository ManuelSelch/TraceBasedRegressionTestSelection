from __future__ import annotations

from copy import deepcopy
from typing import Any

import networkx as nx

from src.graph_builder import build_function_web, generate_all_keyword_traces
from src.selectors import (
    build_ecu_to_test_cases_map,
    generate_all_test_case_traces,
    select_test_cases_for_all_changes,
)


def apply_experiment_to_signals(
    signals: list[dict[str, Any]], experiment: dict[str, Any]
) -> list[dict[str, Any]]:
    removed_signals = set(experiment.get("removed_signals", []))

    return [
        deepcopy(signal)
        for signal in signals
        if signal["id"] not in removed_signals
    ]


def apply_experiment_to_graph(
    graph: nx.DiGraph, experiment: dict[str, Any]
) -> nx.DiGraph:
    degraded_graph = graph.copy()

    for source, target in experiment.get("removed_edges", []):
        if degraded_graph.has_edge(source, target):
            degraded_graph.remove_edge(source, target)

    return degraded_graph


def run_selection_pipeline(
    ecus: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    keywords: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenarios_by_id = {scenario["id"]: scenario for scenario in scenarios}

    effective_signals = apply_experiment_to_signals(signals, experiment or {})
    function_web = build_function_web(ecus, effective_signals)
    function_web = apply_experiment_to_graph(function_web, experiment or {})

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

    return {
        "function_web": function_web,
        "keyword_traces": keyword_traces,
        "test_case_traces": test_case_traces,
        "ecu_to_test_cases": ecu_to_test_cases,
        "selection_results": selection_results,
    }


def compare_selection_results(
    baseline_results: dict[str, dict[str, Any]],
    experiment_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}

    for change_id, baseline in baseline_results.items():
        experiment = experiment_results[change_id]

        baseline_tests = set(baseline.get("selected_test_cases", []))
        experiment_tests = set(experiment.get("selected_test_cases", []))
        baseline_scenarios = set(baseline.get("selected_logical_scenarios", []))
        experiment_scenarios = set(experiment.get("selected_logical_scenarios", []))

        matched_tests = baseline_tests.intersection(experiment_tests)
        matched_scenarios = baseline_scenarios.intersection(experiment_scenarios)
        missed_tests = baseline_tests - experiment_tests
        extra_tests = experiment_tests - baseline_tests
        missed_scenarios = baseline_scenarios - experiment_scenarios
        extra_scenarios = experiment_scenarios - baseline_scenarios

        test_recall = len(matched_tests) / len(baseline_tests) if baseline_tests else 1.0
        scenario_recall = (
            len(matched_scenarios) / len(baseline_scenarios) if baseline_scenarios else 1.0
        )

        comparison[change_id] = {
            "change": change_id,
            "baseline_selected_count": baseline["selected_count"],
            "experiment_selected_count": experiment["selected_count"],
            "baseline_reduction_rate": baseline["reduction_rate"],
            "experiment_reduction_rate": experiment["reduction_rate"],
            "test_case_recall": test_recall,
            "scenario_recall": scenario_recall,
            "missed_affected_test_cases": sorted(missed_tests),
            "extra_selected_test_cases": sorted(extra_tests),
            "missed_affected_scenarios": sorted(missed_scenarios),
            "extra_selected_scenarios": sorted(extra_scenarios),
            "missed_affected_test_case_count": len(missed_tests),
            "extra_selected_test_case_count": len(extra_tests),
            "missed_affected_scenario_count": len(missed_scenarios),
            "extra_selected_scenario_count": len(extra_scenarios),
        }

    return comparison


def summarize_experiment_comparison(
    experiment: dict[str, Any], comparison_by_change: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparisons = list(comparison_by_change.values())

    change_count = len(comparisons)
    avg_test_case_recall = (
        sum(item["test_case_recall"] for item in comparisons) / change_count
        if change_count
        else 1.0
    )
    avg_scenario_recall = (
        sum(item["scenario_recall"] for item in comparisons) / change_count
        if change_count
        else 1.0
    )
    total_missed_test_cases = sum(
        item["missed_affected_test_case_count"] for item in comparisons
    )
    total_missed_scenarios = sum(
        item["missed_affected_scenario_count"] for item in comparisons
    )

    has_any_miss = any(
        item["missed_affected_test_case_count"] > 0
        or item["missed_affected_scenario_count"] > 0
        for item in comparisons
    )
    worst_change = (
        min(
            comparisons,
            key=lambda item: (item["test_case_recall"], item["scenario_recall"]),
            default=None,
        )
        if has_any_miss
        else None
    )

    return {
        "experiment": experiment["id"],
        "description": experiment.get("description", ""),
        "removed_signals": list(experiment.get("removed_signals", [])),
        "removed_edges": [list(edge) for edge in experiment.get("removed_edges", [])],
        "avg_test_case_recall": avg_test_case_recall,
        "avg_scenario_recall": avg_scenario_recall,
        "total_missed_affected_test_cases": total_missed_test_cases,
        "total_missed_affected_scenarios": total_missed_scenarios,
        "worst_change": worst_change["change"] if worst_change else None,
        "worst_change_test_case_recall": (
            worst_change["test_case_recall"] if worst_change else 1.0
        ),
    }


def run_experiments(
    ecus: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    keywords: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_pipeline = run_selection_pipeline(
        ecus,
        signals,
        keywords,
        scenarios,
        changes,
        test_cases,
    )
    baseline_results = baseline_pipeline["selection_results"]

    experiment_outputs: dict[str, Any] = {}

    for experiment in experiments:
        experiment_pipeline = run_selection_pipeline(
            ecus,
            signals,
            keywords,
            scenarios,
            changes,
            test_cases,
            experiment=experiment,
        )
        experiment_results = experiment_pipeline["selection_results"]
        comparison_by_change = compare_selection_results(
            baseline_results,
            experiment_results,
        )

        experiment_outputs[experiment["id"]] = {
            "experiment": experiment,
            "selection_results": experiment_results,
            "comparison_by_change": comparison_by_change,
            "summary": summarize_experiment_comparison(
                experiment,
                comparison_by_change,
            ),
        }

    return {
        "baseline_selection_results": baseline_results,
        "experiments": experiment_outputs,
    }
