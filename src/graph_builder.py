from __future__ import annotations
from typing import Any
import networkx as nx


ECU_KIND = "ecu"
SIGNAL_KIND = "signal"

PRODUCES_RELATION = "produces"
CONSUMED_BY_RELATION = "consumed_by"


def build_function_web(ecus: list[dict[str, Any]], signals: list[dict[str, Any]]) -> nx.DiGraph:
    """
    Build the Function Web as a directed bipartite graph.

    Nodes:
    - ECU nodes
    - Signal nodes

    Edges:
    - ECU -> Signal   (produces)
    - Signal -> ECU   (consumed_by)

    Expected ECU shape:
    {
        "id": "planner",
        "type": "processing",
        "forwards_only": False,
        "inputs": ["detected_objects", "ego_pose"],
        "outputs": ["target_path"]
    }

    Expected Signal shape:
    {
        "id": "target_path",
        "level": "planning",
        "producer": "planner",
        "consumers": ["controller"]
    }
    """
    graph = nx.DiGraph(name="function_web")

    ecu_ids = {ecu["id"] for ecu in ecus}
    signal_ids = {signal["id"] for signal in signals}

    _add_ecu_nodes(graph, ecus)
    _add_signal_nodes(graph, signals)
    _add_signal_edges(graph, signals, ecu_ids, signal_ids)

    return graph


def _add_ecu_nodes(graph: nx.DiGraph, ecus: list[dict[str, Any]]) -> None:
    for ecu in ecus:
        ecu_id = ecu["id"]
        graph.add_node(
            ecu_id,
            kind=ECU_KIND,
            ecu_type=ecu.get("type"),
            forwards_only=ecu.get("forwards_only", False),
            description=ecu.get("description"),
            inputs=list(ecu.get("inputs", [])),
            outputs=list(ecu.get("outputs", [])),
        )


def _add_signal_nodes(graph: nx.DiGraph, signals: list[dict[str, Any]]) -> None:
    for signal in signals:
        signal_id = signal["id"]
        graph.add_node(
            signal_id,
            kind=SIGNAL_KIND,
            level=signal.get("level"),
            producer=signal.get("producer"),
            consumers=list(signal.get("consumers", [])),
            description=signal.get("description"),
        )


def _add_signal_edges(
    graph: nx.DiGraph,
    signals: list[dict[str, Any]],
    ecu_ids: set[str],
    signal_ids: set[str],
) -> None:
    for signal in signals:
        signal_id = signal["id"]
        producer = signal.get("producer")
        consumers = signal.get("consumers", [])

        if signal_id not in signal_ids:
            raise ValueError(f"Unknown signal id: {signal_id}")

        if producer is not None:
            if producer not in ecu_ids:
                raise ValueError(
                    f"Signal '{signal_id}' references unknown producer ECU '{producer}'"
                )
            graph.add_edge(producer, signal_id, relation=PRODUCES_RELATION)

        for consumer in consumers:
            if consumer not in ecu_ids:
                raise ValueError(
                    f"Signal '{signal_id}' references unknown consumer ECU '{consumer}'"
                )
            graph.add_edge(signal_id, consumer, relation=CONSUMED_BY_RELATION)


if __name__ == "__main__":
    # Minimal smoke-test example
    example_ecus = [
        {"id": "camera", "type": "sensor", "outputs": ["camera_image"]},
        {
            "id": "perception",
            "type": "processing",
            "inputs": ["camera_image"],
            "outputs": ["detected_objects"],
        },
    ]
    example_signals = [
        {
            "id": "camera_image",
            "level": "sensor",
            "producer": "camera",
            "consumers": ["perception"],
        },
        {
            "id": "detected_objects",
            "level": "perception",
            "producer": "perception",
            "consumers": [],
        },
    ]

    function_web = build_function_web(example_ecus, example_signals)
    print("Nodes:", list(function_web.nodes(data=True)))
    print("Edges:", list(function_web.edges(data=True)))
