# Athena v4 — kill/restart resume demo

Demonstrates that `v4`'s `SqliteSaver` checkpointing actually survives a
process crash: kill the process partway through a run, restart it pointing at
the same `thread_id` and the same `v4/checkpoints.db` file, and confirm it
resumes from the last checkpoint instead of starting over from `plan_node`
(which re-runs an LLM planning call, is the most expensive step to redo, and
is the step whose repetition would be the most obviously wrong "resume").

This is scripted and reproducible via `v4/run_thread.py`, whose `--kill-after
NODE_NAME` flag hard-exits (`os._exit`, no clean shutdown) immediately after
a given node's update is observed on `graph.stream(..., stream_mode=
"updates")` — the same mechanism a real `SIGKILL`/crash would trigger, just
timed deterministically instead of racing a wall-clock `sleep` against an LLM
call. Per the assignment, this doc documents the reproducible procedure;
capturing it as a screen recording (per the "Ship" criteria) is left to you,
but every command below is copy-pasteable as-is.

## Run it

```bash
rm -f v4/checkpoints.db   # start from a clean db for a fresh demo

# 1. Start a new thread, simulate a crash right after act_node completes.
uv run python v4/run_thread.py \
    --thread-id demo-1 \
    --question "Who founded Synchron?" \
    --kill-after act_node
```

Expected output (process then hard-exits with code 1):

```
Starting new thread 'demo-1'.
  [plan_node] completed and checkpointed
  [act_node] completed and checkpointed
  Simulating a crash immediately after act_node (hard exit).
```

```bash
# 2. Restart, pointing at the SAME thread_id and the SAME db file.
uv run python v4/run_thread.py --thread-id demo-1 --auto-approve
```

Expected output:

```
Resuming existing thread 'demo-1' (next: ('act_node',)).
  [act_node] completed and checkpointed
  [observe_node] completed and checkpointed
  [reflect_node] completed and checkpointed
  [__interrupt__] completed and checkpointed
  Auto-approving draft brief, resuming to finalize_node...
  [finalize_node] completed and checkpointed
Run complete.
### Summary
...
```

## What this proves, precisely

`plan_node` is **never re-run** on resume — the LLM planning call that
produced the research plan is not repeated, and `snapshot.next` on restart
is never `('plan_node',)` after a kill that happened at or after `act_node`.
That is the literal claim this demo needs to establish.

`act_node` itself *is* re-run in the transcript above. This is not a bug —
it's checkpoint-commit timing, and it's worth understanding precisely rather
than glossing over: `graph.stream(..., stream_mode="updates")` yields a
node's update as soon as that node's logic finishes, but LangGraph durably
commits the checkpoint recording "this node ran, here's what's next" as part
of *starting the following super-step*, not synchronously before the yield.
Killing the process the instant a node's line is printed therefore lands in
the small window before that node's own checkpoint is committed — so on
resume, the **last durably committed checkpoint** is the one written after
the *previous* node, and its `next` still points at the node you thought
you'd just finished. You can see this directly:

```bash
uv run python -c "
from v4.graph import compiled_app
from pathlib import Path
config = {'configurable': {'thread_id': 'demo-1'}}
with compiled_app(db_path=Path('v4/checkpoints.db')) as app:
    for snap in app.get_state_history(config):
        print('next=', snap.next, 'has_plan=', snap.values.get('plan') is not None, 'has_action=', snap.values.get('action') is not None)
"
```

Right after the simulated crash, this prints three checkpoints: `__start__`,
one after `plan_node` (`next=('act_node',)`, `has_plan=True`), and nothing
for `act_node` itself — confirming `act_node`'s own checkpoint never made it
to disk. Whichever node you `--kill-after`, that same node is the one
re-executed on resume; every node *before* it is not. Pass `--kill-after
observe_node` instead to see a cleaner resume (`next=('observe_node',)`,
skipping both `plan_node` and `act_node`).

## Resuming through the approval gate

If you kill *after* `reflect_node` on the iteration that decides `"done"`,
the restart resumes straight into the `interrupt_before=["finalize_node"]`
gate — `snapshot.next == ("finalize_node",)` — and `v4/app.py` (or
`--auto-approve` here) picks up the approval flow exactly as if the process
had never died. This is the same checkpoint mechanism the Streamlit app
relies on; nothing about the approval gate is special-cased for crash
recovery.
