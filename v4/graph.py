"""LangGraph wiring for Athena v4.

Graph sketch (see v4/DESIGN.md for the full mapping to v3 functions and the
per-field reducer rationale):

                 +------------+
      START ---> | plan_node  |<---------------------+
                 +-----+------+                       |
                       | (fixed edge)                  |
                       v                                | replan
                 +------------+                         |
                 |  act_node  |                         |
                 +-----+------+                         |
                       | (fixed edge -- always taken,    |
                       |  even if actor chose "done")    |
                       v                                 |
                 +-------------+                         |
                 | observe_node|                          |
                 +-----+-------+                          |
                       | (fixed edge)                       |
                       v                                     |
                 +-------------+   continue    +------------+|
                 | reflect_node|-------------->|  act_node   ||
                 +-----+-------+               +------------+|
                       |                                      |
                       | conditional edge: route_after_reflect |
                       |   iteration >= 8         -> finalize_node
                       |   decision == "done"     -> finalize_node
                       |   decision == "replan"   -> plan_node  (above)
                       |   decision == "continue" -> act_node   (above)
                       v
                 +--------------+
                 | finalize_node|  <-- interrupt_before=["finalize_node"]
                 +------+-------+       (human approval gate)
                        v
                       END

Only reflect_node fans out via a conditional edge; every other edge is fixed.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from v4.nodes import act_node, finalize_node, observe_node, plan_node, reflect_node
from v4.state import MAX_ITERATIONS, AthenaState

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "checkpoints.db"

# Checkpoints store v3/v4 Pydantic models (Plan, ActionDecision, ReflectDecision,
# Observation) via msgpack. Newer langgraph versions require an explicit allowlist
# for non-builtin types deserialized from a checkpoint; without this, resuming a
# run will start warning now and raise in a future langgraph release.
_ALLOWED_MSGPACK_MODULES = [
    ("v3.trajectory", "Plan"),
    ("v3.trajectory", "ActionDecision"),
    ("v3.trajectory", "ReflectDecision"),
    ("v4.state", "Observation"),
]


def route_after_reflect(state: AthenaState) -> Literal["act_node", "plan_node", "finalize_node"]:
    """The one conditional edge in the graph. Mirrors v3/research_agent.py's
    loop body (`if reflection.decision == "done": break / elif "replan": ... /
    else: continue`) plus its `range(max_steps)` iteration cap -- both now
    live here instead of inside a node function."""
    if state["iteration"] >= MAX_ITERATIONS:
        return "finalize_node"

    reflection = state.get("reflection")
    decision = reflection.decision if reflection is not None else "done"
    if decision == "continue":
        return "act_node"
    if decision == "replan":
        return "plan_node"
    return "finalize_node"


def build_graph() -> StateGraph:
    """Construct (but do not compile) the Athena v4 state graph."""
    builder = StateGraph(AthenaState)

    builder.add_node("plan_node", plan_node)
    builder.add_node("act_node", act_node)
    builder.add_node("observe_node", observe_node)
    builder.add_node("reflect_node", reflect_node)
    builder.add_node("finalize_node", finalize_node)

    builder.set_entry_point("plan_node")
    builder.add_edge("plan_node", "act_node")
    builder.add_edge("act_node", "observe_node")
    builder.add_edge("observe_node", "reflect_node")
    builder.add_conditional_edges(
        "reflect_node",
        route_after_reflect,
        {
            "act_node": "act_node",
            "plan_node": "plan_node",
            "finalize_node": "finalize_node",
        },
    )
    builder.add_edge("finalize_node", END)

    return builder


@contextmanager
def compiled_app(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    interrupt_before_finalize: bool = True,
) -> Iterator["langgraph.graph.state.CompiledStateGraph"]:  # noqa: F821
    """Compile the graph with a SqliteSaver checkpointer open for the life of
    this context. SqliteSaver.from_conn_string() is itself a context manager
    over the sqlite connection, so a second process opening the same db_path
    later sees every checkpoint this process committed -- this is what makes
    the kill/restart resume demo (v4/demo_resume.md) work.
    """
    builder = build_graph()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        checkpointer = SqliteSaver(conn, serde=serde)
        app = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["finalize_node"] if interrupt_before_finalize else None,
        )
        yield app
