"""Node functions for the Athena v4 graph.

Each node is a pure function AthenaState -> partial AthenaState update. All
LLM/tool logic is delegated to the unmodified v3 functions; nodes only bridge
AthenaState to/from an ephemeral v3.trajectory.Trajectory. See
v4/DESIGN.md ("Bridging AthenaState <-> v3.Trajectory") for why.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from v3.actor import act as v3_act
from v3.observer import observe as v3_observe
from v3.planner import plan as v3_plan
from v3.reflector import reflect as v3_reflect
from v3.research_agent import final_answer as v3_final_answer
from v3.trajectory import Trajectory
from v4.state import AthenaState, Observation


def _trajectory_from_state(state: AthenaState) -> Trajectory:
    """Rebuild the mutable v3.Trajectory that v3's functions expect."""
    return Trajectory(
        iteration=state["iteration"],
        question=state["question"],
        current_plan=state["plan"],
        notes=state["notes"],
        observations=[obs.raw for obs in state["observations"]],
        completed_steps=list(state["completed_steps"]),
        notes_sources=list(state["notes_sources"]),
        web_sources=list(state["web_sources"]),
    )


def plan_node(state: AthenaState) -> Dict[str, Any]:
    """Generate (or adopt, on replan) the current research plan.

    On the replan path, v3.reflector.reflect() has already computed
    reflection.new_plan (v3/reflector.py:24-25) — plan_node adopts it rather
    than calling the planner LLM a second time, mirroring
    research_agent.py:61.
    """
    reflection = state.get("reflection")
    if reflection is not None and reflection.decision == "replan" and reflection.new_plan is not None:
        new_plan = reflection.new_plan
    else:
        trajectory = _trajectory_from_state(state)
        new_plan = v3_plan(state["question"], trajectory)

    messages: list = []
    if not state.get("messages"):
        messages.append(HumanMessage(content=state["question"]))
    messages.append(
        AIMessage(content=f"Plan: {'; '.join(new_plan.steps)}\nRationale: {new_plan.rationale}")
    )

    return {"plan": new_plan, "reflection": None, "messages": messages}


def act_node(state: AthenaState) -> Dict[str, Any]:
    """Choose the next bounded action from the current plan and trajectory."""
    trajectory = _trajectory_from_state(state)
    action = v3_act(state["plan"], trajectory)

    return {
        "action": action,
        "messages": [AIMessage(content=f"Action: {action.tool}\nReasoning: {action.reasoning}")],
    }


def observe_node(state: AthenaState) -> Dict[str, Any]:
    """Execute the pending action and compress the result into notes.

    Always runs, even when act_node chose tool="done" — see v4/DESIGN.md
    ("done handling") for why v4 relies on the tool="done" branch already
    present in v3/observer.py::observe() rather than short-circuiting via a
    new edge out of act_node.
    """
    trajectory = _trajectory_from_state(state)
    action = state["action"]
    result = v3_observe(action, trajectory)

    observation = Observation(
        tool=action.tool,
        raw=result["raw"],
        summary=result["summary"],
        source_type=result["source_type"],
    )

    return {
        "notes": trajectory.notes,
        "observations": [observation],
        "completed_steps": trajectory.completed_steps[len(state["completed_steps"]) :],
        "notes_sources": trajectory.notes_sources,
        "web_sources": trajectory.web_sources,
        "iteration": trajectory.iteration,
        "messages": [AIMessage(content=f"Observation ({action.tool}): {result['summary']}")],
    }


def reflect_node(state: AthenaState) -> Dict[str, Any]:
    """Decide continue/replan/done. Does NOT branch on the decision itself —
    branching is route_after_reflect, a conditional edge in v4/graph.py."""
    trajectory = _trajectory_from_state(state)
    reflection = v3_reflect(trajectory)

    return {
        "reflection": reflection,
        "messages": [AIMessage(content=f"Reflection: {reflection.decision} — {reflection.reasoning}")],
    }


def finalize_node(state: AthenaState) -> Dict[str, Any]:
    """Synthesize the final cited answer. Gated by interrupt_before in
    v4/graph.py so a human can review the accumulated notes first."""
    trajectory = _trajectory_from_state(state)
    answer = v3_final_answer(trajectory)

    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
