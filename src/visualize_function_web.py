from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from graphviz import Digraph

from graph_builder import ECU_KIND, build_function_web


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "model"
OUT_DIR = PROJECT_ROOT / "out"

ECU_FILL = "lightblue"
SIGNAL_FILL = "orange"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_dot(graph) -> Digraph:
    dot = Digraph("function_web")
    dot.attr(rankdir="LR", nodesep="0.8", ranksep="1.2")
    dot.attr("graph", splines="polyline")
    dot.attr("node", style="filled", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    for node, attrs in sorted(graph.nodes(data=True)):
        if attrs.get("kind") == ECU_KIND:
            dot.node(node, shape="box", fillcolor=ECU_FILL)
        else:
            dot.node(node, shape="ellipse", fillcolor=SIGNAL_FILL)

    for source, target, attrs in sorted(graph.edges(data=True)):
        dot.edge(source, target, label=attrs.get("relation", ""))

    return dot


def main() -> None:
    ecus = load_yaml(MODEL_DIR / "ecus.yaml")["ecus"]
    signals = load_yaml(MODEL_DIR / "signals.yaml")["signals"]

    graph = build_function_web(ecus, signals)
    dot = build_dot(graph)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUT_DIR / "function_web"

    dot.save(str(output_base.with_suffix(".dot")))
    rendered_path = dot.render(str(output_base), format="png", cleanup=False)

    print(f"Saved DOT to: {output_base.with_suffix('.dot')}")
    print(f"Saved image to: {rendered_path}")


if __name__ == "__main__":
    main()
