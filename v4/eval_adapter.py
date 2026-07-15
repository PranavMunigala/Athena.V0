"""Eval-harness adapter for Athena v4 (evals/run.py --version wk8).

Exposes the same query_fn(str) -> str shape v1/v2/v3 use, plus a
get_last_v4_run() accessor mirroring v3's get_last_trajectory() so
evals/run.py can report iteration counts for wk8 the same way it does
for wk7.

Runs the real compiled graph (same build_graph() as v4/app.py, same
interrupt_before=["finalize_node"] gate) against an in-memory SqliteSaver
per call, auto-approving the interrupt immediately since eval runs are
non-interactive. This exercises the identical node/edge/reducer code path
the Streamlit app uses -- only the checkpoint backend (in-memory vs. file)
and the auto-approval differ.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from typing import Optional

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from v4.graph import _ALLOWED_MSGPACK_MODULES, build_graph
from v4.state import initial_state

_LAST_ITERATION: Optional[int] = None


class LastRunSummary:
    """Minimal stand-in for v3.trajectory.Trajectory's `.iteration` attribute
    access pattern used by evals/run.py's _score_row()."""

    def __init__(self, iteration: int) -> None:
        self.iteration = iteration


def query_athena_v4(query: str) -> str:
    """Run one question through the full v4 graph, auto-approving the
    human gate, and return the final cited answer."""
    global _LAST_ITERATION

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        checkpointer = SqliteSaver(conn, serde=serde)
        app = build_graph().compile(
            checkpointer=checkpointer,
            interrupt_before=["finalize_node"],
        )

        for _ in app.stream(initial_state(query), config, stream_mode="updates"):
            pass
        # Auto-approve: resume past the interrupt straight to finalize_node.
        for _ in app.stream(None, config, stream_mode="updates"):
            pass

        snapshot = app.get_state(config)

    _LAST_ITERATION = int(snapshot.values.get("iteration", 0))
    return snapshot.values.get("final_answer") or ""


def get_last_v4_run() -> Optional[LastRunSummary]:
    """Return the most recent run's iteration count for metrics/reporting."""
    if _LAST_ITERATION is None:
        return None
    return LastRunSummary(iteration=_LAST_ITERATION)
