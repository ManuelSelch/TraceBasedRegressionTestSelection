from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from graphviz import Digraph

from src.graph_builder import ECU_KIND, build_function_web

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "model"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
OUT_DIR = PROJECT_ROOT / "out"

ECU_FILL = "lightblue"
SIGNAL_FILL = "orange"
INACTIVE_FILL = "#f2f2f2"
ACTIVE_ECU_FILL = "#9be9a8"
ACTIVE_SIGNAL_FILL = "#ffd58a"
CHANGED_FILL = "#ff9aa2"
INACTIVE_EDGE_COLOR = "#d0d0d0"
ACTIVE_EDGE_COLOR = "#444444"
CHANGED_BORDER_COLOR = "#cc0000"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_dot(
    graph,
    *,
    active_ecus: set[str] | None = None,
    active_signals: set[str] | None = None,
    active_edges: set[tuple[str, str]] | None = None,
    changed_ecus: set[str] | None = None,
    title: str | None = None,
) -> Digraph:
    active_ecus = active_ecus or set()
    active_signals = active_signals or set()
    active_edges = active_edges or set()
    changed_ecus = changed_ecus or set()

    dot = Digraph("function_web")
    dot.attr(rankdir="LR", nodesep="0.8", ranksep="1.2")
    dot.attr("graph", splines="polyline")
    dot.attr("node", style="filled", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    if title:
        dot.attr(label=title, labelloc="t", fontsize="20")

    for node, attrs in sorted(graph.nodes(data=True)):
        is_ecu = attrs.get("kind") == ECU_KIND
        is_active = node in active_ecus if is_ecu else node in active_signals
        is_changed = is_ecu and node in changed_ecus

        if is_changed:
            fillcolor = CHANGED_FILL
            color = CHANGED_BORDER_COLOR
            penwidth = "2.5"
        elif is_active:
            fillcolor = ACTIVE_ECU_FILL if is_ecu else ACTIVE_SIGNAL_FILL
            color = "black"
            penwidth = "1.5"
        else:
            fillcolor = ECU_FILL if is_ecu and not active_ecus and not active_signals else INACTIVE_FILL
            if not is_ecu and not active_ecus and not active_signals:
                fillcolor = SIGNAL_FILL
            color = "#999999" if active_ecus or active_signals else "black"
            penwidth = "1.0"

        dot.node(
            node,
            shape="box" if is_ecu else "ellipse",
            fillcolor=fillcolor,
            color=color,
            penwidth=penwidth,
        )

    for source, target, attrs in sorted(graph.edges(data=True)):
        is_active_edge = (source, target) in active_edges
        dot.edge(
            source,
            target,
            label=attrs.get("relation", ""),
            color=ACTIVE_EDGE_COLOR if is_active_edge or not active_edges else INACTIVE_EDGE_COLOR,
            penwidth="2.0" if is_active_edge else "1.0",
        )

    return dot


def export(
    graph,
    *,
    output_name: str = "function_web",
    active_ecus: set[str] | None = None,
    active_signals: set[str] | None = None,
    active_edges: set[tuple[str, str]] | None = None,
    changed_ecus: set[str] | None = None,
    title: str | None = None,
) -> None:
    dot = build_dot(
        graph,
        active_ecus=active_ecus,
        active_signals=active_signals,
        active_edges=active_edges,
        changed_ecus=changed_ecus,
        title=title,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUT_DIR / output_name
    dot.render(str(output_base), format="png", cleanup=False)


def collect_change_view(
    change_id: str,
    selection_results: dict[str, Any],
    test_cases: list[dict[str, Any]],
    keyword_traces_by_scenario: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if change_id not in selection_results:
        raise ValueError(f"Unknown change id: '{change_id}'")

    result = selection_results[change_id]
    test_cases_by_id = {test_case["id"]: test_case for test_case in test_cases}

    active_ecus: set[str] = set()
    active_signals: set[str] = set()
    active_edges: set[tuple[str, str]] = set()

    for test_case_id in result["selected_test_cases"]:
        test_case = test_cases_by_id[test_case_id]
        scenario_id = test_case["logical_scenario"]
        scenario_keyword_traces = keyword_traces_by_scenario[scenario_id]

        for keyword_id in test_case.get("keywords", []):
            trace = scenario_keyword_traces[keyword_id]
            active_ecus.update(trace.get("reached_ecus", []))
            active_signals.update(trace.get("reached_signals", []))
            active_edges.update(tuple(edge) for edge in trace.get("reached_edges", []))

    changed_ecus = set(result.get("changed_ecus", []))

    return {
        "change_id": change_id,
        "changed_ecus": changed_ecus,
        "active_ecus": active_ecus,
        "active_signals": active_signals,
        "active_edges": active_edges,
        "selected_test_cases": result.get("selected_test_cases", []),
        "selected_logical_scenarios": result.get("selected_logical_scenarios", []),
    }


def export_change_view(change_id: str) -> None:
    ecus = load_yaml(MODEL_DIR / "ecus.yaml")["ecus"]
    signals = load_yaml(MODEL_DIR / "signals.yaml")["signals"]
    test_cases = load_yaml(BENCHMARK_DIR / "test_cases.yaml")["test_cases"]
    selection_results = load_json(OUT_DIR / "selection_results.json")
    keyword_traces_by_scenario = load_json(OUT_DIR / "keyword_traces.json")

    function_web = build_function_web(ecus, signals)
    change_view = collect_change_view(
        change_id,
        selection_results,
        test_cases,
        keyword_traces_by_scenario,
    )

    export(
        function_web,
        output_name=f"change_{change_id}",
        active_ecus=change_view["active_ecus"],
        active_signals=change_view["active_signals"],
        active_edges=change_view["active_edges"],
        changed_ecus=change_view["changed_ecus"],
        title=(
            f"{change_id} | "
            f"tests: {len(change_view['selected_test_cases'])} | "
            f"scenarios: {len(change_view['selected_logical_scenarios'])}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--change", help="Render highlighted view for a specific change id")
    args = parser.parse_args()

    if args.change:
        export_change_view(args.change)
        return

    ecus = load_yaml(MODEL_DIR / "ecus.yaml")["ecus"]
    signals = load_yaml(MODEL_DIR / "signals.yaml")["signals"]
    function_web = build_function_web(ecus, signals)
    export(function_web)


if __name__ == "__main__":
    main()
