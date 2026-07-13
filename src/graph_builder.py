from __future__ import annotations

from collections import deque
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
            required_features=list(ecu.get("required_features", [])),
            mode_inputs=dict(ecu.get("mode_inputs", {})),
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


def _is_ecu_active(
    graph: nx.DiGraph,
    ecu_id: str,
    active_features: set[str] | None,
    enabled_features: set[str] | None,
) -> bool:
    required_features = set(graph.nodes[ecu_id].get("required_features", []))

    if active_features is not None and not required_features.issubset(active_features):
        return False

    if enabled_features is not None and not required_features.issubset(enabled_features):
        return False

    return True


def _find_arbiters(graph: nx.DiGraph) -> list[str]:
    return sorted(
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("ecu_type") == "arbiter"
    )


def _compute_output_relevant_nodes(
    graph: nx.DiGraph,
    active_arbiter_modes: dict[str, str],
) -> set[str]:
    """Compute all nodes that can contribute to any observable sink output.

    Observable sinks are signal nodes without consumers. Reverse traversal keeps
    all branches, except when traversing backwards through an arbiter ECU: in
    that case only the input signal allowed by the active arbiter mode is kept.
    """
    sink_signals = [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("kind") == SIGNAL_KIND and graph.out_degree(node) == 0
    ]

    relevant: set[str] = set(sink_signals)
    queue: deque[str] = deque(sink_signals)

    while queue:
        current = queue.popleft()
        if current not in graph:
            continue

        predecessors = list(graph.predecessors(current))
        if (
            graph.nodes[current].get("kind") == ECU_KIND
            and graph.nodes[current].get("ecu_type") == "arbiter"
        ):
            arbiter_mode = active_arbiter_modes.get(current)
            if arbiter_mode is not None:
                allowed_signal = graph.nodes[current].get("mode_inputs", {}).get(arbiter_mode)
                predecessors = [pred for pred in predecessors if pred == allowed_signal]

        for predecessor in predecessors:
            if predecessor in relevant:
                continue
            relevant.add(predecessor)
            queue.append(predecessor)

    return relevant


def generate_keyword_trace(
    graph: nx.DiGraph,
    keyword: dict[str, Any],
    active_features: list[str] | set[str] | None = None,
    enabled_features: list[str] | set[str] | None = None,
    allow_missing_initial_nodes: bool = False,
    active_mode: str | None = None,
    active_arbiter_modes: dict[str, str] | None = None,
) -> dict[str, Any]:
    keyword_id = keyword["id"]
    initial_ecus = list(keyword.get("initial_ecus", []))
    initial_signals = list(keyword.get("initial_signals", []))

    if not initial_ecus and not initial_signals:
        raise ValueError(f"Keyword '{keyword_id}' has no initial ECUs or signals")

    normalized_active_features = set(active_features) if active_features is not None else None
    normalized_enabled_features = (
        set(enabled_features) if enabled_features is not None else None
    )

    valid_initial_ecus: list[str] = []
    valid_initial_signals: list[str] = []
    missing_initial_ecus: list[str] = []
    missing_initial_signals: list[str] = []

    for ecu_id in initial_ecus:
        if ecu_id not in graph:
            if allow_missing_initial_nodes:
                missing_initial_ecus.append(ecu_id)
                continue
            raise ValueError(
                f"Keyword '{keyword_id}' references unknown initial ECU '{ecu_id}'"
            )
        if graph.nodes[ecu_id].get("kind") != ECU_KIND:
            raise ValueError(
                f"Keyword '{keyword_id}' initial node '{ecu_id}' is not an ECU"
            )
        valid_initial_ecus.append(ecu_id)

    for signal_id in initial_signals:
        if signal_id not in graph:
            if allow_missing_initial_nodes:
                missing_initial_signals.append(signal_id)
                continue
            raise ValueError(
                f"Keyword '{keyword_id}' references unknown initial signal '{signal_id}'"
            )
        if graph.nodes[signal_id].get("kind") != SIGNAL_KIND:
            raise ValueError(
                f"Keyword '{keyword_id}' initial node '{signal_id}' is not a signal"
            )
        valid_initial_signals.append(signal_id)

    effective_arbiter_modes = dict(active_arbiter_modes or {})
    if active_mode is not None:
        for arbiter_id in _find_arbiters(graph):
            effective_arbiter_modes.setdefault(arbiter_id, active_mode)

    # Compute output-relevant subgraph when arbiter modes are active.
    # A node is relevant if it can contribute to any observable sink output and
    # is not cut off by an arbiter in the current scenario.
    mode_relevant_nodes: set[str] | None = None
    if effective_arbiter_modes:
        mode_relevant_nodes = _compute_output_relevant_nodes(
            graph,
            effective_arbiter_modes,
        )

    distances: dict[str, int] = {}
    queue: deque[str] = deque()

    for ecu_id in valid_initial_ecus:
        if not _is_ecu_active(
            graph,
            ecu_id,
            normalized_active_features,
            normalized_enabled_features,
        ):
            continue
        if mode_relevant_nodes is not None and ecu_id not in mode_relevant_nodes:
            continue

        distances[ecu_id] = 0
        queue.append(ecu_id)

    for signal_id in valid_initial_signals:
        if mode_relevant_nodes is not None and signal_id not in mode_relevant_nodes:
            continue
        distances[signal_id] = 0
        queue.append(signal_id)
        # Include producer ECU so sensor/component changes are caught
        producer = graph.nodes[signal_id].get("producer")
        if (
            producer
            and producer in graph
            and graph.nodes[producer].get("kind") == ECU_KIND
            and (mode_relevant_nodes is None or producer in mode_relevant_nodes)
            and producer not in distances
        ):
            distances[producer] = 0
            queue.append(producer)

    while queue:
        node_id = queue.popleft()
        current_distance = distances[node_id]

        for successor_id in graph.successors(node_id):
            if mode_relevant_nodes is not None and successor_id not in mode_relevant_nodes:
                continue
            if graph.nodes[successor_id].get("kind") == ECU_KIND and not _is_ecu_active(
                graph,
                successor_id,
                normalized_active_features,
                normalized_enabled_features,
            ):
                continue

            if successor_id in distances:
                continue

            distances[successor_id] = current_distance + 1
            queue.append(successor_id)

    reached_ecus = sorted(
        node
        for node, distance in distances.items()
        if graph.nodes[node].get("kind") == ECU_KIND and distance >= 0
    )
    reached_signals = sorted(
        node
        for node, distance in distances.items()
        if graph.nodes[node].get("kind") == SIGNAL_KIND and distance >= 1
    )
    direct_ecus = sorted(
        node
        for node, distance in distances.items()
        if graph.nodes[node].get("kind") == ECU_KIND and distance == 2
    )
    indirect_ecus = sorted(
        node
        for node, distance in distances.items()
        if graph.nodes[node].get("kind") == ECU_KIND and distance > 2
    )

    reached_node_set = set(distances.keys())
    reached_edges = sorted(
        [source, target]
        for source, target in graph.edges()
        if source in reached_node_set and target in reached_node_set
    )

    return {
        "keyword": keyword_id,
        "initial_ecus": sorted(valid_initial_ecus),
        "initial_signals": sorted(valid_initial_signals),
        "missing_initial_ecus": sorted(missing_initial_ecus),
        "missing_initial_signals": sorted(missing_initial_signals),
        "active_features": sorted(normalized_active_features or []),
        "active_mode": active_mode,
        "active_arbiter_modes": dict(sorted(effective_arbiter_modes.items())),
        "enabled_features": sorted(normalized_enabled_features or []),
        "direct_ecus": direct_ecus,
        "indirect_ecus": indirect_ecus,
        "reached_ecus": reached_ecus,
        "reached_signals": reached_signals,
        "reached_edges": reached_edges,
    }


def generate_all_keyword_traces(
    graph: nx.DiGraph,
    keywords: list[dict[str, Any]],
    active_features: list[str] | set[str] | None = None,
    enabled_features: list[str] | set[str] | None = None,
    allow_missing_initial_nodes: bool = False,
    active_mode: str | None = None,
    active_arbiter_modes: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        keyword["id"]: generate_keyword_trace(
            graph,
            keyword,
            active_features=active_features,
            enabled_features=enabled_features,
            allow_missing_initial_nodes=allow_missing_initial_nodes,
            active_mode=active_mode,
            active_arbiter_modes=active_arbiter_modes,
        )
        for keyword in keywords
    }
