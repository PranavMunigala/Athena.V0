"""Typed trajectory state for Athena v3's manual research loop."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Ordered research strategy produced before acting."""

    steps: List[str] = Field(default_factory=list)
    rationale: str


class ActionDecision(BaseModel):
    """One action selected by the actor."""

    tool: Literal["search_notes", "web_search", "done"]
    args: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str


class ReflectDecision(BaseModel):
    """Reflection result after observing a tool output."""

    decision: Literal["continue", "replan", "done"]
    reasoning: str
    new_plan: Optional[Plan] = None


class Trajectory(BaseModel):
    """Compressed working state for the planning research agent."""

    iteration: int = 0
    question: str
    current_plan: Optional[Plan] = None
    notes: str = ""
    observations: List[str] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)
    notes_sources: List[str] = Field(default_factory=list)
    web_sources: List[str] = Field(default_factory=list)

    def add_observation(self, observation: str) -> None:
        """Keep only the two most recent raw observations."""
        self.observations.append(observation)
        self.observations = self.observations[-2:]

    def add_source(self, source_type: Literal["notes", "web"], source: str) -> None:
        """Track unique citations by source type."""
        target = self.notes_sources if source_type == "notes" else self.web_sources
        if source and source not in target:
            target.append(source)

    def as_prompt_context(self) -> str:
        """Render the bounded state that may be placed into model prompts."""
        plan = self.current_plan.model_dump() if self.current_plan else None
        recent = "\n\n".join(self.observations[-2:]) or "No recent observations."
        completed = "; ".join(self.completed_steps) or "No completed steps yet."
        notes = self.notes or "No compressed notes yet."
        return (
            f"Question: {self.question}\n"
            f"Iteration: {self.iteration}\n"
            f"Current plan: {plan}\n"
            f"Compressed notes: {notes}\n"
            f"Recent observations: {recent}\n"
            f"Completed steps: {completed}"
        )


def utc_timestamp() -> str:
    """Return an ISO timestamp suitable for logs."""
    return datetime.now(timezone.utc).isoformat()
