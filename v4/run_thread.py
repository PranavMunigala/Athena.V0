"""CLI runner for the kill/restart resume demo. See v4/demo_resume.md.

Supports --kill-after NODE_NAME, which hard-exits (os._exit, no clean
shutdown -- simulating a real crash/kill) immediately after that node's
update is yielded by graph.stream(). Since LangGraph commits a checkpoint
before yielding a node's update, this deterministically reproduces "the
process died right after node X finished" without racing a wall-clock sleep
against an LLM call.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v4.graph import DEFAULT_DB_PATH, compiled_app
from v4.state import initial_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--question", help="Required only when starting a new thread.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument(
        "--kill-after",
        help="Node name to hard-exit after (e.g. act_node), simulating a crash.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="If the run pauses before finalize_node, resume past the approval gate too.",
    )
    args = parser.parse_args()

    config = {"configurable": {"thread_id": args.thread_id}}

    with compiled_app(db_path=Path(args.db_path)) as app:
        snapshot = app.get_state(config)

        if not snapshot.values:
            if not args.question:
                parser.error("--question is required when starting a new thread")
            print(f"Starting new thread '{args.thread_id}'.")
            input_state = initial_state(args.question)
        else:
            print(
                f"Resuming existing thread '{args.thread_id}' "
                f"(next: {snapshot.next or 'none -- already finished'})."
            )
            input_state = None

        for step in app.stream(input_state, config, stream_mode="updates"):
            (node_name, _update), = step.items()
            print(f"  [{node_name}] completed and checkpointed")
            if args.kill_after and node_name == args.kill_after:
                print(f"  Simulating a crash immediately after {node_name} (hard exit).")
                sys.stdout.flush()
                os._exit(1)

        snapshot = app.get_state(config)

        if snapshot.next == ("finalize_node",) and args.auto_approve:
            print("  Auto-approving draft brief, resuming to finalize_node...")
            for step in app.stream(None, config, stream_mode="updates"):
                (node_name, _update), = step.items()
                print(f"  [{node_name}] completed and checkpointed")
            snapshot = app.get_state(config)

        if snapshot.next:
            print(f"Paused before: {snapshot.next}")
        else:
            print("Run complete.")
            print(snapshot.values.get("final_answer"))


if __name__ == "__main__":
    main()
