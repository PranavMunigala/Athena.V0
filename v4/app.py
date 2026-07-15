"""Athena v4 — Streamlit UI for the LangGraph plan/act/observe/reflect graph.

Two things this app demonstrates that v1-v3's Streamlit apps do not:

1. Human-in-the-loop approval: the compiled graph is built with
   interrupt_before=["finalize_node"] (v4/graph.py), so every run pauses
   before the final answer is synthesized and shows the accumulated
   research notes as a "draft brief" for approval/rejection.
2. Live intermediate state: graph.stream(..., stream_mode="updates") is used
   to show which node just ran and what it produced, rather than blocking
   until the run completes. interrupt_before is only used for the approval
   gate above, never for this progress display.

Because SqliteSaver persists every checkpoint to v4/checkpoints.db, this app
relies on Streamlit's "rerun the whole script on every interaction" model
naturally: run progress lives in the database, not in in-memory session
state, so re-opening the same thread_id after a rerun (or after the process
restarts) picks up exactly where the graph left off. See v4/demo_resume.md
for a scripted demonstration of that outside Streamlit.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from langchain_core.messages import HumanMessage

from v3.trajectory import ReflectDecision
from v4.graph import DEFAULT_DB_PATH, compiled_app
from v4.state import initial_state

st.set_page_config(
    page_title="Athena v4 — LangGraph Research Agent",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NODE_LABELS = {
    "plan_node": "📝 Plan",
    "act_node": "🎯 Act",
    "observe_node": "🔎 Observe",
    "reflect_node": "🤔 Reflect",
    "finalize_node": "✅ Finalize",
}


def _config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _describe_update(node_name: str, update: Dict[str, Any]) -> str:
    """One-line-plus-detail description of a single node's update, used both
    for the live 'current node' badge and the accumulated timeline log."""
    if node_name == "plan_node" and update.get("plan") is not None:
        plan = update["plan"]
        return f"Steps: {plan.steps}\n\n{plan.rationale}"
    if node_name == "act_node" and update.get("action") is not None:
        action = update["action"]
        return f"Tool: `{action.tool}` — {action.reasoning}"
    if node_name == "observe_node":
        return "\n\n".join(f"`{obs.tool}` -> {obs.summary}" for obs in update.get("observations", []))
    if node_name == "reflect_node" and update.get("reflection") is not None:
        reflection = update["reflection"]
        return f"Decision: **{reflection.decision}** — {reflection.reasoning}"
    return ""


def _run_until_interrupt_or_end(app, config: Dict[str, Any], input_state, progress) -> None:
    """Stream one segment of the graph, showing a live 'current node' badge
    plus a growing timeline of every node's output as events arrive --
    intermediate state via graph.stream(), never interrupt_before (that is
    reserved for the finalize_node approval gate)."""
    st.session_state.setdefault("timeline", [])
    current = progress.empty()
    log = progress.container()

    for step in app.stream(input_state, config, stream_mode="updates"):
        (node_name, update), = step.items()
        label = NODE_LABELS.get(node_name, node_name)
        detail = _describe_update(node_name, update)
        current.info(f"Running: **{label}**")
        st.session_state.timeline.append((label, detail))
        with log.expander(f"{label} (iteration {update.get('iteration', '')})".strip(), expanded=True):
            st.write(detail or "_(no summary)_")
    current.empty()


def _render_timeline(progress) -> None:
    """Redraw the accumulated timeline after a rerun (session_state persists
    across Streamlit reruns even though the graph.stream() generator does not)."""
    for label, detail in st.session_state.get("timeline", []):
        with progress.expander(label, expanded=False):
            st.write(detail or "_(no summary)_")


def _draft_brief(state_values: Dict[str, Any]) -> None:
    st.subheader("Draft brief (pending your approval)")
    st.markdown("**Compressed research notes:**")
    st.write(state_values.get("notes") or "_No notes yet._")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Athena notes sources**")
        for src in state_values.get("notes_sources") or ["_none_"]:
            st.write(f"- {src}")
    with col2:
        st.markdown("**Web sources**")
        for src in state_values.get("web_sources") or ["_none_"]:
            st.write(f"- {src}")
    st.caption(
        f"Iteration {state_values.get('iteration', 0)} / 8 — "
        f"{len(state_values.get('completed_steps') or [])} completed steps."
    )


def main() -> None:
    st.title("🏛️ Athena v4 — LangGraph Research Agent")
    st.caption(
        "Plan → Act → Observe → Reflect, rebuilt as a LangGraph state machine "
        "(see v4/DESIGN.md). Resumable via SqliteSaver; pauses before the "
        "final answer for human approval."
    )

    with st.sidebar:
        st.header("⚙️ Session")
        db_path = Path(
            st.text_input("Checkpoint DB path", value=str(DEFAULT_DB_PATH))
        )
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.thread_id = st.text_input(
            "Thread ID",
            value=st.session_state.thread_id,
            help="Reuse a previous run's thread ID to resume it (see v4/demo_resume.md).",
        )
        if st.button("New thread"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.timeline = []
            st.rerun()

    config = _config(st.session_state.thread_id)

    # If this thread already has state (fresh start, mid-run, or awaiting
    # approval), reflect that instead of forcing a brand-new question.
    with compiled_app(db_path=db_path) as app:
        snapshot = app.get_state(config)

    existing_question = snapshot.values.get("question") if snapshot.values else None

    question = st.text_input(
        "Research question",
        value=existing_question or "",
        placeholder="e.g., What medical problem is Synchron trying to solve?",
    )

    st.subheader("Live progress")
    progress = st.container()
    _render_timeline(progress)

    start_disabled = not question or (snapshot.values and snapshot.next)
    if st.button("Start research", disabled=bool(start_disabled)):
        st.session_state.timeline = []
        with compiled_app(db_path=db_path) as app:
            _run_until_interrupt_or_end(app, config, initial_state(question), progress)
            snapshot = app.get_state(config)
        st.rerun()

    # Awaiting approval before finalize_node.
    if snapshot.values and snapshot.next == ("finalize_node",):
        _draft_brief(snapshot.values)

        approve_col, reject_col = st.columns(2)
        with approve_col:
            if st.button("✅ Approve — write final answer"):
                with compiled_app(db_path=db_path) as app:
                    _run_until_interrupt_or_end(app, config, None, progress)
                    snapshot = app.get_state(config)
                st.rerun()

        with reject_col:
            feedback = st.text_area("Feedback (required to reject)")
            if st.button("❌ Reject — send back to planning", disabled=not feedback):
                with compiled_app(db_path=db_path) as app:
                    app.update_state(
                        config,
                        {
                            "notes": (snapshot.values.get("notes") or "")
                            + f"\n\nHuman feedback: {feedback}",
                            "messages": [HumanMessage(content=feedback)],
                            "reflection": ReflectDecision(
                                decision="replan",
                                reasoning=f"Human rejected the draft brief: {feedback}",
                                new_plan=None,
                            ),
                        },
                        as_node="reflect_node",
                    )
                    _run_until_interrupt_or_end(app, config, None, progress)
                    snapshot = app.get_state(config)
                st.rerun()

    # Finished (no more pending nodes) and a final answer exists.
    if snapshot.values and not snapshot.next and snapshot.values.get("final_answer"):
        st.subheader("Final answer")
        st.markdown(snapshot.values["final_answer"])


if __name__ == "__main__":
    main()
