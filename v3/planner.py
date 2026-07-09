"""Planning stage for Athena v3."""

from __future__ import annotations

from v3.llm import structured_completion
from v3.trajectory import Plan, Trajectory


def plan(question: str, trajectory: Trajectory | None = None) -> Plan:
    """Generate an ordered research plan for a question."""
    context = trajectory.as_prompt_context() if trajectory else f"Question: {question}"
    try:
        return structured_completion(
            response_model=Plan,
            system=(
                "You are Athena v3's planner. Produce a concise ordered research "
                "strategy. Prefer Athena notes first for course/corpus facts, web "
                "search for current or external facts, and synthesis last."
            ),
            user=context,
            temperature=0.1,
        )
    except Exception:
        steps = ["Search Athena notes for grounded context"]
        if _looks_current_or_external(question):
            steps.append("Search the web for current external context")
        steps.append("Synthesize a cited answer separating notes and web evidence")
        return Plan(steps=steps, rationale="Fallback plan based on question type.")


def _looks_current_or_external(question: str) -> bool:
    lowered = question.lower()
    triggers = [
        "latest",
        "current",
        "today",
        "as of",
        "compare",
        "web",
        "news",
        "price",
        "approval status",
    ]
    return any(trigger in lowered for trigger in triggers)
