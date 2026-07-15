.PHONY: graph

# Renders v4's LangGraph state machine to docs/graph.png via the remote
# Mermaid.ink API (draw_mermaid_png()) -- no local Graphviz install required.
# See v4/DESIGN.md ("Graph visualization") for the no-Graphviz-needed note.
graph:
	uv run python v4/render_graph.py
