from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml
from graphviz import Digraph

from src.graph_builder import ECU_KIND, SIGNAL_KIND, build_function_web

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "model"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
OUT_DIR = PROJECT_ROOT / "out"

ECU_FILL = "lightblue"
INACTIVE_FILL = "#f2f2f2"
ACTIVE_ECU_FILL = "#9be9a8"
CHANGED_FILL = "#ff9aa2"
INACTIVE_EDGE_COLOR = "#d0d0d0"
ACTIVE_EDGE_COLOR = "#444444"
CHANGED_BORDER_COLOR = "#cc0000"
COVERAGE_MIN_FILL = "#deebf7"
COVERAGE_MAX_FILL = "#08519c"
SINK_NODE_FILL = "#fff7bc"
SINK_NODE_COLOR = "#b8860b"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _rgb_to_hex(rgb: Sequence[int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _interpolate_color(start: str, end: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    mixed = tuple(
        round(start_channel + (end_channel - start_channel) * ratio)
        for start_channel, end_channel in zip(start_rgb, end_rgb)
    )
    return _rgb_to_hex(mixed)


def _build_visual_edges(graph) -> tuple[dict[tuple[str, str], list[str]], dict[str, str]]:
    ecu_edges: dict[tuple[str, str], list[str]] = {}
    sink_signals: dict[str, str] = {}

    for signal_id, attrs in sorted(graph.nodes(data=True)):
        if attrs.get("kind") != SIGNAL_KIND:
            continue

        producers = [node for node in graph.predecessors(signal_id) if graph.nodes[node].get("kind") == ECU_KIND]
        consumers = [node for node in graph.successors(signal_id) if graph.nodes[node].get("kind") == ECU_KIND]

        if not producers:
            continue

        producer = producers[0]

        if consumers:
            for consumer in consumers:
                ecu_edges.setdefault((producer, consumer), []).append(signal_id)
        else:
            sink_signals[signal_id] = producer

    for edge_key in ecu_edges:
        ecu_edges[edge_key] = sorted(ecu_edges[edge_key])

    return ecu_edges, sink_signals


def _build_active_visual_edges(
    graph,
    active_signals: set[str],
) -> tuple[set[tuple[str, str]], set[str]]:
    active_ecu_edges: set[tuple[str, str]] = set()
    active_sink_signals: set[str] = set()

    for signal_id in active_signals:
        if signal_id not in graph or graph.nodes[signal_id].get("kind") != SIGNAL_KIND:
            continue

        producers = [node for node in graph.predecessors(signal_id) if graph.nodes[node].get("kind") == ECU_KIND]
        consumers = [node for node in graph.successors(signal_id) if graph.nodes[node].get("kind") == ECU_KIND]

        if not producers:
            continue

        producer = producers[0]

        if consumers:
            for consumer in consumers:
                active_ecu_edges.add((producer, consumer))
        else:
            active_sink_signals.add(signal_id)

    return active_ecu_edges, active_sink_signals


def build_dot(
    graph,
    *,
    active_ecus: set[str] | None = None,
    active_signals: set[str] | None = None,
    changed_ecus: set[str] | None = None,
    ecu_coverage_counts: dict[str, int] | None = None,
    total_test_cases: int = 0,
    title: str | None = None,
) -> Digraph:
    active_ecus = active_ecus or set()
    active_signals = active_signals or set()
    changed_ecus = changed_ecus or set()
    ecu_coverage_counts = ecu_coverage_counts or {}
    is_coverage_mode = bool(ecu_coverage_counts)

    visual_edges, sink_signals = _build_visual_edges(graph)
    active_visual_edges, active_sink_signals = _build_active_visual_edges(graph, active_signals)
    has_active_view = bool(active_ecus or active_signals)

    dot = Digraph("function_web")
    dot.attr(rankdir="LR", nodesep="0.8", ranksep="1.2")
    dot.attr("graph", splines="polyline")
    dot.attr("node", style="filled", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    if title:
        dot.attr(label=title, labelloc="t", fontsize="20")

    for node, attrs in sorted(graph.nodes(data=True)):
        if attrs.get("kind") != ECU_KIND:
            continue

        is_active = node in active_ecus
        is_changed = node in changed_ecus
        label = node
        fontcolor = "black"

        if is_changed:
            fillcolor = CHANGED_FILL
            color = CHANGED_BORDER_COLOR
            penwidth = "2.5"
        elif is_coverage_mode:
            count = ecu_coverage_counts.get(node, 0)
            ratio = count / total_test_cases if total_test_cases else 0.0
            fillcolor = _interpolate_color(COVERAGE_MIN_FILL, COVERAGE_MAX_FILL, ratio)
            color = "black"
            penwidth = "1.2"
            label = f"{node}\n{count}/{total_test_cases}"
            if ratio >= 0.55:
                fontcolor = "white"
        elif is_active:
            fillcolor = ACTIVE_ECU_FILL
            color = "black"
            penwidth = "1.5"
        else:
            fillcolor = ECU_FILL if not has_active_view else INACTIVE_FILL
            color = "black" if not has_active_view else "#999999"
            penwidth = "1.0"

        dot.node(
            node,
            label=label,
            shape="box",
            fillcolor=fillcolor,
            color=color,
            penwidth=penwidth,
            fontcolor=fontcolor,
        )

    for signal_id, producer in sorted(sink_signals.items()):
        sink_node_id = f"__sink__{signal_id}"
        is_active = signal_id in active_sink_signals
        fillcolor = SINK_NODE_FILL if not has_active_view or is_active else INACTIVE_FILL
        color = SINK_NODE_COLOR if not has_active_view or is_active else "#999999"
        fontcolor = "black" if not has_active_view or is_active else "#999999"

        dot.node(
            sink_node_id,
            label=signal_id,
            shape="ellipse",
            fillcolor=fillcolor,
            color=color,
            penwidth="1.2",
            fontcolor=fontcolor,
        )
        dot.edge(
            producer,
            sink_node_id,
            color=ACTIVE_EDGE_COLOR if (not has_active_view or is_active) else INACTIVE_EDGE_COLOR,
            penwidth="2.0" if is_active else "1.0",
        )

    for (producer, consumer), signal_ids in sorted(visual_edges.items()):
        is_active_edge = (producer, consumer) in active_visual_edges
        label = "\n".join(signal_ids)
        dot.edge(
            producer,
            consumer,
            label=label,
            color=ACTIVE_EDGE_COLOR if (not has_active_view or is_active_edge) else INACTIVE_EDGE_COLOR,
            penwidth="2.0" if is_active_edge else "1.0",
        )

    return dot


def export(
    graph,
    *,
    output_name: str = "function_web",
    active_ecus: set[str] | None = None,
    active_signals: set[str] | None = None,
    changed_ecus: set[str] | None = None,
    ecu_coverage_counts: dict[str, int] | None = None,
    total_test_cases: int = 0,
    title: str | None = None,
) -> None:
    dot = build_dot(
        graph,
        active_ecus=active_ecus,
        active_signals=active_signals,
        changed_ecus=changed_ecus,
        ecu_coverage_counts=ecu_coverage_counts,
        total_test_cases=total_test_cases,
        title=title,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUT_DIR / output_name
    dot.render(str(output_base), format="png", cleanup=False)


def collect_change_view(
    change_id: str,
    selection_results: dict[str, Any],
    scenarios_by_id: dict[str, dict[str, Any]],
    test_cases: list[dict[str, Any]],
    keyword_traces_by_scenario: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if change_id not in selection_results:
        raise ValueError(f"Unknown change id: '{change_id}'")

    result = selection_results[change_id]
    test_cases_by_id = {test_case["id"]: test_case for test_case in test_cases}

    active_ecus: set[str] = set()
    active_signals: set[str] = set()

    for test_case_id in result["selected_test_cases"]:
        test_case = test_cases_by_id[test_case_id]
        scenario_id = test_case["logical_scenario"]
        scenario = scenarios_by_id[scenario_id]
        scenario_keyword_traces = keyword_traces_by_scenario[scenario_id]

        for keyword_id in scenario.get("keywords", []):
            trace = scenario_keyword_traces[keyword_id]
            active_ecus.update(trace.get("reached_ecus", []))
            active_signals.update(trace.get("reached_signals", []))

    changed_ecus = set(result.get("changed_ecus", []))

    return {
        "change_id": change_id,
        "changed_ecus": changed_ecus,
        "active_ecus": active_ecus,
        "active_signals": active_signals,
        "selected_test_cases": result.get("selected_test_cases", []),
        "selected_logical_scenarios": result.get("selected_logical_scenarios", []),
    }


def collect_ecu_coverage(test_case_traces: dict[str, dict[str, Any]]) -> tuple[dict[str, int], int]:
    ecu_coverage_counts: dict[str, int] = {}

    for trace in test_case_traces.values():
        for ecu_id in trace.get("reached_ecus", []):
            ecu_coverage_counts[ecu_id] = ecu_coverage_counts.get(ecu_id, 0) + 1

    return ecu_coverage_counts, len(test_case_traces)


def export_change_view(change_id: str) -> None:
    ecus = load_yaml(MODEL_DIR / "ecus.yaml")["ecus"]
    signals = load_yaml(MODEL_DIR / "signals.yaml")["signals"]
    scenarios = load_yaml(MODEL_DIR / "scenarios.yaml")["scenarios"]
    test_cases = load_yaml(BENCHMARK_DIR / "test_cases.yaml")["test_cases"]
    selection_results = load_json(OUT_DIR / "selection_results.json")
    keyword_traces_by_scenario = load_json(OUT_DIR / "keyword_traces.json")

    function_web = build_function_web(ecus, signals)
    change_view = collect_change_view(
        change_id,
        selection_results,
        {scenario["id"]: scenario for scenario in scenarios},
        test_cases,
        keyword_traces_by_scenario,
    )

    export(
        function_web,
        output_name=f"change_{change_id}",
        active_ecus=change_view["active_ecus"],
        active_signals=change_view["active_signals"],
        changed_ecus=change_view["changed_ecus"],
        title=(
            f"{change_id} | "
            f"tests: {len(change_view['selected_test_cases'])} | "
            f"scenarios: {len(change_view['selected_logical_scenarios'])}"
        ),
    )


def export_coverage_view() -> None:
    ecus = load_yaml(MODEL_DIR / "ecus.yaml")["ecus"]
    signals = load_yaml(MODEL_DIR / "signals.yaml")["signals"]
    test_case_traces = load_json(OUT_DIR / "test_case_traces.json")

    function_web = build_function_web(ecus, signals)
    ecu_coverage_counts, total_test_cases = collect_ecu_coverage(test_case_traces)

    export(
        function_web,
        output_name="function_web_coverage",
        ecu_coverage_counts=ecu_coverage_counts,
        total_test_cases=total_test_cases,
        title=f"ECU coverage by test cases ({total_test_cases} total tests)",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--change", help="Render highlighted view for a specific change id")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Render ECU coverage counts across all test cases",
    )
    args = parser.parse_args()

    if args.change:
        export_change_view(args.change)
        return

    if args.coverage:
        export_coverage_view()
        return

    ecus = load_yaml(MODEL_DIR / "ecus.yaml")["ecus"]
    signals = load_yaml(MODEL_DIR / "signals.yaml")["signals"]
    function_web = build_function_web(ecus, signals)
    export(function_web)


if __name__ == "__main__":
    main()
