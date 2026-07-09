"""Reflection stage for Athena v3."""

from __future__ import annotations

from v3.llm import structured_completion
from v3.planner import plan
from v3.trajectory import ReflectDecision, Trajectory


def reflect(trajectory: Trajectory) -> ReflectDecision:
    """Decide whether to continue, replan, or terminate early."""
    try:
        decision = structured_completion(
            response_model=ReflectDecision,
            system=(
                "You are Athena v3's reflector. Inspect the plan, compressed notes, "
                "recent observations, and completed steps. Decide continue, replan, "
                "or done. Do not rubber-stamp continue; identify missing evidence, "
                "redundant searches, or sufficient evidence."
            ),
            user=trajectory.as_prompt_context(),
            temperature=0.1,
        )
        if decision.decision == "replan" and decision.new_plan is None:
            decision.new_plan = plan(trajectory.question, trajectory)
        return decision
    except Exception:
        return _fallback_reflection(trajectory)


def _fallback_reflection(trajectory: Trajectory) -> ReflectDecision:
    completed = " ".join(trajectory.completed_steps).lower()
    lowered = trajectory.question.lower()
    needs_web = any(
        token in lowered
        for token in ["latest", "current", "today", "as of", "news", "price", "compare"]
    )
    if "search_notes" not in completed:
        return ReflectDecision(
            decision="continue",
            reasoning="Need at least one notes search before answering.",
        )
    if needs_web and "web_search" not in completed:
        return ReflectDecision(
            decision="continue",
            reasoning="Question appears to need external or current evidence.",
        )
    return ReflectDecision(
        decision="done",
        reasoning="Available evidence is sufficient for a best-effort answer.",
    )
