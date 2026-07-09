"""Action selection for Athena v3."""

from __future__ import annotations

from v3.llm import structured_completion
from v3.trajectory import ActionDecision, Plan, Trajectory


def act(plan: Plan, trajectory: Trajectory) -> ActionDecision:
    """Choose the next bounded action from the current plan and trajectory."""
    try:
        return structured_completion(
            response_model=ActionDecision,
            system=(
                "You are Athena v3's actor. Choose exactly one tool: search_notes, "
                "web_search, or done. Use only the current plan, compressed notes, "
                "recent observations, and completed steps. Do not assume access to "
                "older raw observations."
            ),
            user=trajectory.as_prompt_context(),
            temperature=0.1,
        )
    except Exception:
        return _fallback_action(trajectory)


def _fallback_action(trajectory: Trajectory) -> ActionDecision:
    question = trajectory.question
    completed = " ".join(trajectory.completed_steps).lower()
    lowered = question.lower()
    needs_web = any(
        token in lowered
        for token in ["latest", "current", "today", "as of", "news", "price", "compare"]
    )

    if "search_notes" not in completed:
        return ActionDecision(
            tool="search_notes",
            args={"query": question},
            reasoning="Fallback actor starts with Athena notes for grounded context.",
        )
    if needs_web and "web_search" not in completed:
        return ActionDecision(
            tool="web_search",
            args={"query": question},
            reasoning="Fallback actor adds web evidence for current or external context.",
        )
    return ActionDecision(
        tool="done",
        args={},
        reasoning="Fallback actor has gathered the available evidence.",
    )
