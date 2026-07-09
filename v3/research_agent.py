"""Manual Plan-Act-Observe-Reflect research agent for Athena v3."""

from __future__ import annotations

from typing import Optional

from v3.actor import act
from v3.llm import text_completion
from v3.logging import TrajectoryLogger
from v3.observer import observe
from v3.planner import plan
from v3.reflector import reflect
from v3.trajectory import ActionDecision, Trajectory

LAST_TRAJECTORY: Optional[Trajectory] = None


def research_agent(
    question: str,
    max_steps: int = 8,
    *,
    enable_reflection: bool = True,
) -> str:
    """Answer a research question with a bounded planning loop."""
    global LAST_TRAJECTORY
    max_steps = min(max_steps, 8)
    trajectory = Trajectory(question=question)
    logger = TrajectoryLogger()

    trajectory.current_plan = plan(question, trajectory)

    for _ in range(max_steps):
        action = act(trajectory.current_plan, trajectory)
        if action.tool == "done":
            logger.append(
                trajectory=trajectory,
                action=action,
                tool_result_summary="Actor selected done.",
                reflection=None,
            )
            break

        result = observe(action, trajectory)
        reflection = (
            reflect(trajectory)
            if enable_reflection
            else None
        )
        logger.append(
            trajectory=trajectory,
            action=action,
            tool_result_summary=result["summary"],
            reflection=reflection,
        )

        if reflection is None:
            continue
        if reflection.decision == "done":
            break
        if reflection.decision == "replan":
            trajectory.current_plan = reflection.new_plan or plan(question, trajectory)

    LAST_TRAJECTORY = trajectory
    return final_answer(trajectory)


def final_answer(trajectory: Trajectory) -> str:
    """Generate the required separated final response."""
    try:
        return text_completion(
            system=(
                "You are Athena v3. Write a final answer with exactly these section "
                "headings: Summary, Key Findings, Information from Athena Notes, "
                "Information from Web Search, Citations. Clearly separate note "
                "evidence from web evidence. Preserve inline citations."
            ),
            user=trajectory.as_prompt_context(),
            temperature=0.2,
            max_tokens=1200,
        )
    except Exception:
        notes = trajectory.notes or "No supporting evidence was gathered."
        note_sources = "\n".join(f"- {source}" for source in trajectory.notes_sources) or "- None"
        web_sources = "\n".join(f"- {source}" for source in trajectory.web_sources) or "- None"
        return (
            "Summary\n"
            f"{_first_paragraph(notes)}\n\n"
            "Key Findings\n"
            f"{notes}\n\n"
            "Information from Athena Notes\n"
            f"{_source_summary(trajectory, 'search_notes')}\n\n"
            "Information from Web Search\n"
            f"{_source_summary(trajectory, 'web_search')}\n\n"
            "Citations\n"
            f"Athena Notes:\n{note_sources}\n\nWeb Search:\n{web_sources}"
        )


def _first_paragraph(text: str) -> str:
    return text.strip().split("\n\n", 1)[0][:800]


def _source_summary(trajectory: Trajectory, tool_name: str) -> str:
    lines = [
        step for step in trajectory.completed_steps if step.lower().startswith(tool_name)
    ]
    if not lines:
        return "No information gathered from this source."
    return trajectory.notes or "Information was gathered but could not be summarized."


def query_athena_v3(query: str) -> str:
    """Public v3 query function used by evals and app integrations."""
    return research_agent(query)


def query_athena_v3_no_reflection(query: str) -> str:
    """Public v3 query function with reflection disabled for ablation evals."""
    return research_agent(query, enable_reflection=False)


def get_last_trajectory() -> Optional[Trajectory]:
    """Return the most recent trajectory for metrics/reporting."""
    return LAST_TRAJECTORY


def done_action(reasoning: str = "Done.") -> ActionDecision:
    """Small helper for tests and future integrations."""
    return ActionDecision(tool="done", args={}, reasoning=reasoning)
