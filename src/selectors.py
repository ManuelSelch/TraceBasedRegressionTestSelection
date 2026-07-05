from __future__ import annotations
from typing import Any

def generate_test_case_trace(
    test_case: dict[str, Any], keyword_traces: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    reached_ecus: set[str] = set()

    for keyword_id in test_case.get("keywords", []):
        if keyword_id not in keyword_traces:
            raise ValueError(f"Unknown keyword trace: '{keyword_id}'")
        reached_ecus.update(keyword_traces[keyword_id]["reached_ecus"])

    return {
        "test_case": test_case["id"],
        "logical_scenario": test_case["logical_scenario"],
        "keywords": list(test_case.get("keywords", [])),
        "reached_ecus": sorted(reached_ecus),
    }

def generate_all_test_case_traces(
    test_cases: list[dict[str, Any]],
    keyword_traces_by_scenario: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        test_case["id"]: generate_test_case_trace(
            test_case,
            keyword_traces_by_scenario[test_case["logical_scenario"]],
        )
        for test_case in test_cases
    }

def build_ecu_to_test_cases_map(
    test_case_traces: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    ecu_to_test_cases: dict[str, set[str]] = {}

    for test_case_id, trace in test_case_traces.items():
        for ecu_id in trace["reached_ecus"]:
            ecu_to_test_cases.setdefault(ecu_id, set()).add(test_case_id)

    return {
        ecu_id: sorted(test_case_ids)
        for ecu_id, test_case_ids in sorted(ecu_to_test_cases.items())
    }

def select_test_cases_for_change(
    change: dict[str, Any], test_case_traces: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    changed_ecus = set(change.get("changed_ecus", []))
    selected_test_cases: list[str] = []
    selected_logical_scenarios: set[str] = set()

    for test_case_id, trace in test_case_traces.items():
        if changed_ecus.intersection(trace["reached_ecus"]):
            selected_test_cases.append(test_case_id)
            selected_logical_scenarios.add(trace["logical_scenario"])

    all_test_cases = len(test_case_traces)
    selected_count = len(selected_test_cases)
    reduction_rate = 1.0 - (selected_count / all_test_cases) if all_test_cases else 0.0

    return {
        "change": change["id"],
        "changed_ecus": sorted(changed_ecus),
        "selected_test_cases": sorted(selected_test_cases),
        "selected_logical_scenarios": sorted(selected_logical_scenarios),
        "selected_count": selected_count,
        "all_test_cases": all_test_cases,
        "reduction_rate": reduction_rate,
    }

def select_test_cases_for_all_changes(
    changes: list[dict[str, Any]], test_case_traces: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        change["id"]: select_test_cases_for_change(change, test_case_traces)
        for change in changes
    }
