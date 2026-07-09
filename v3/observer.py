"""Observation and memory-compression stage for Athena v3."""

from __future__ import annotations

import json
from typing import Dict

from v3.llm import compact_lines, text_completion
from v3.tools import search_notes, web_search
from v3.trajectory import ActionDecision, Trajectory


def observe(action: ActionDecision, trajectory: Trajectory) -> Dict[str, str]:
    """Execute one action, update compressed memory, and bound observations."""
    if action.tool == "search_notes":
        query = str(action.args.get("query") or trajectory.question)
        raw = search_notes(query=query, k=int(action.args.get("k", 5)))
        observation = _format_notes_result(raw)
        source_type = "notes"
    elif action.tool == "web_search":
        query = str(action.args.get("query") or trajectory.question)
        raw = web_search(query=query)
        observation = _format_web_result(raw)
        source_type = "web"
    else:
        observation = "No tool executed; actor selected done."
        raw = {"summary": observation}
        source_type = "notes"

    summary = summarize_observation(trajectory, action, observation)
    trajectory.notes = _merge_notes(trajectory.notes, summary)
    trajectory.add_observation(observation)
    trajectory.completed_steps.append(f"{action.tool}: {action.reasoning}")
    trajectory.iteration += 1

    for source in _sources_from_raw(raw, source_type):
        trajectory.add_source(source_type, source)

    return {
        "raw": observation,
        "summary": summary,
        "source_type": source_type,
    }


def summarize_observation(
    trajectory: Trajectory,
    action: ActionDecision,
    observation: str,
) -> str:
    """Compress one tool output into trajectory notes."""
    try:
        return text_completion(
            system=(
                "Compress the new observation into durable research notes. Keep "
                "facts, citations, conflicts, and missing evidence. Be concise."
            ),
            user=(
                f"Question: {trajectory.question}\n"
                f"Existing notes:\n{trajectory.notes or 'None'}\n\n"
                f"Action: {action.model_dump()}\n\n"
                f"Observation:\n{observation}"
            ),
            temperature=0.1,
            max_tokens=500,
        )
    except Exception:
        head = compact_lines(observation.splitlines(), limit=8)
        return f"{action.tool} found:\n{head}"


def _merge_notes(existing: str, addition: str, limit: int = 5000) -> str:
    merged = f"{existing.strip()}\n\n{addition.strip()}".strip()
    return merged[-limit:]


def _format_notes_result(raw: Dict[str, object]) -> str:
    chunks = raw.get("chunks", [])
    lines = [f"Athena notes search query: {raw.get('query', '')}"]
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        lines.append(
            "\n".join(
                [
                    f"Result {index}: {chunk.get('citation')} similarity={float(chunk.get('similarity_score', 0.0)):.3f}",
                    str(chunk.get("text", ""))[:1200],
                ]
            )
        )
    return "\n\n".join(lines)


def _format_web_result(raw: Dict[str, object]) -> str:
    return (
        f"Web search query: {raw.get('query', '')}\n"
        f"Summary:\n{raw.get('summary', '')}\n"
        f"Sources: {json.dumps(raw.get('sources', []))}"
    )


def _sources_from_raw(raw: Dict[str, object], source_type: str) -> list[str]:
    if source_type == "notes":
        sources = []
        for chunk in raw.get("chunks", []):
            if isinstance(chunk, dict):
                citation = str(chunk.get("citation", ""))
                if citation:
                    sources.append(citation)
        return sources
    return [str(source) for source in raw.get("sources", []) if source]
