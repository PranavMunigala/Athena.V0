"""Render the Athena v4 graph to docs/graph.png. Invoked by `make graph`.

Uses graph.get_graph().draw_mermaid_png(), which renders via the remote
Mermaid.ink API by default -- no local Graphviz install required (see
v4/DESIGN.md, "Graph visualization"). Requires outbound network access to
https://mermaid.ink at build time.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v4.graph import build_graph

OUTPUT_PATH = ROOT_DIR / "docs" / "graph.png"


def main() -> None:
    graph = build_graph().compile()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = graph.get_graph().draw_mermaid_png()
    OUTPUT_PATH.write_bytes(png_bytes)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
