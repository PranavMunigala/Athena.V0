"""JSONL trajectory logging for Athena v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from v3.trajectory import ActionDecision, ReflectDecision, Trajectory, utc_timestamp

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAJECTORY_DIR = ROOT_DIR / "trajectories"


class TrajectoryLogger:
    """Append replayable trajectory events to one JSONL file."""

    def __init__(self, log_path: Optional[Path] = None) -> None:
        TRAJECTORY_DIR.mkdir(exist_ok=True)
        self.path = log_path or TRAJECTORY_DIR / f"{utc_timestamp().replace(':', '-')}.jsonl"

    def append(
        self,
        *,
        trajectory: Trajectory,
        action: ActionDecision,
        tool_result_summary: str,
        reflection: Optional[ReflectDecision],
    ) -> None:
        """Append one replayable iteration event."""
        entry: Dict[str, Any] = {
            "timestamp": utc_timestamp(),
            "iteration": trajectory.iteration,
            "current_plan": (
                trajectory.current_plan.model_dump() if trajectory.current_plan else None
            ),
            "chosen_action": action.model_dump(),
            "tool": action.tool,
            "tool_arguments": action.args,
            "tool_result_summary": tool_result_summary,
            "reflection": reflection.model_dump() if reflection else None,
            "notes": trajectory.notes,
            "completed_steps": trajectory.completed_steps,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
