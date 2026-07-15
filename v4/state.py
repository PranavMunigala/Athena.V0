"""Typed LangGraph state for Athena v4.

See v4/DESIGN.md for the full field-by-field reducer rationale. Reuses
v3's Pydantic models (Plan, ReflectDecision) instead of forking them.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel

from v3.trajectory import ActionDecision, Plan, ReflectDecision


class Observation(BaseModel):
    """One executed tool step. Formalizes the ad hoc dict v3's
    observer.observe() already returns (v3 has no Pydantic model for this)."""

    tool: Literal["search_notes", "web_search", "done"]
    raw: str
    summary: str
    source_type: Literal["notes", "web"]


def keep_recent_observations(
    existing: List[Observation], new: List[Observation]
) -> List[Observation]:
    """Keep only the two most recent observations, matching v3's
    Trajectory.add_observation() compression (v3/trajectory.py:46-49)."""
    return (existing + new)[-2:]


def add_unique(existing: List[str], new: List[str]) -> List[str]:
    """Dedup-append, matching v3's Trajectory.add_source() (v3/trajectory.py:51-55)."""
    result = list(existing)
    for item in new:
        if item not in result:
            result.append(item)
    return result


MAX_ITERATIONS = 8


class AthenaState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    plan: Optional[Plan]
    action: Optional[ActionDecision]
    notes: str
    observations: Annotated[List[Observation], keep_recent_observations]
    completed_steps: Annotated[List[str], operator.add]
    notes_sources: Annotated[List[str], add_unique]
    web_sources: Annotated[List[str], add_unique]
    iteration: int
    reflection: Optional[ReflectDecision]
    final_answer: Optional[str]


def initial_state(question: str) -> AthenaState:
    """Build the starting state for a new run (before plan_node executes)."""
    return AthenaState(
        messages=[],
        question=question,
        plan=None,
        action=None,
        notes="",
        observations=[],
        completed_steps=[],
        notes_sources=[],
        web_sources=[],
        iteration=0,
        reflection=None,
        final_answer=None,
    )
