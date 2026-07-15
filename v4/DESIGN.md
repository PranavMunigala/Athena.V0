# Athena v4 — LangGraph Design

v4 rebuilds v3's hand-rolled plan/act/observe/reflect loop
(`v3/research_agent.py`'s `for` loop) as a `langgraph.graph.StateGraph`. v3's
Pydantic models (`Plan`, `ActionDecision`, `ReflectDecision`, `Trajectory`) and
tool wrappers (`v3/tools.py`) are reused, not forked. v1/v2/v3 are untouched.

## Graph sketch

```
                 ┌────────────┐
      START ───▶ │ plan_node  │◀────────────────────┐
                 └─────┬──────┘                      │
                        │ (fixed edge)                │ replan
                        ▼                              │
                 ┌────────────┐                        │
                 │  act_node  │                        │
                 └─────┬──────┘                        │
                        │ (fixed edge, always taken —   │
                        │  even if actor chose "done")  │
                        ▼                              │
                 ┌─────────────┐                       │
                 │ observe_node│                        │
                 └─────┬───────┘                        │
                        │ (fixed edge)                    │
                        ▼                                │
                 ┌─────────────┐   continue    ┌─────────┴───┐
                 │ reflect_node│──────────────▶ │  act_node   │ (loop back)
                 └─────┬───────┘                └─────────────┘
                        │
                        │ conditional edge: route_after_reflect(state)
                        │   - iteration >= 8            -> finalize_node
                        │   - reflection.decision=="done"     -> finalize_node
                        │   - reflection.decision=="replan"   -> plan_node
                        │   - reflection.decision=="continue" -> act_node
                        ▼
                 ┌──────────────┐
                 │ finalize_node│  <-- interrupt_before=["finalize_node"]
                 └──────┬───────┘       (human approval gate, see Task 5)
                        ▼
                       END
```

Only **one** conditional edge exists in the whole graph: the fan-out leaving
`reflect_node`. Every other edge (`plan_node -> act_node`,
`act_node -> observe_node`, `observe_node -> reflect_node`) is fixed. This is
the literal requirement from the assignment: v3's `if decision == "done": break
/ elif decision == "replan": ... / else: continue` inside `research_agent.py`'s
loop body becomes `route_after_reflect`, a top-level function passed to
`graph.add_conditional_edges("reflect_node", route_after_reflect, {...})` — it
is never called from inside a node function.

## v3 function -> v4 node mapping

| v3 function | v4 node | Notes |
|---|---|---|
| `v3/planner.py::plan()` | `plan_node` | Called unchanged against an ephemeral `Trajectory` rebuilt from `AthenaState` (see "Bridging" below). On the replan path, `plan_node` adopts `state["reflection"].new_plan` (already computed by `v3.reflector.reflect`) instead of calling the LLM a second time — this mirrors `research_agent.py:61` (`trajectory.current_plan = reflection.new_plan or plan(...)`). |
| `v3/actor.py::act()` | `act_node` | Called unchanged. Produces an `ActionDecision`, does **not** execute the tool. |
| `v3/observer.py::observe()` | `observe_node` | Called unchanged. Executes `search_notes`/`web_search` (or the `tool="done"` no-op branch — see "done handling" below), compresses the result into `trajectory.notes`, appends to `trajectory.observations` / `completed_steps`, increments `trajectory.iteration`. |
| `v3/reflector.py::reflect()` | `reflect_node` | Called unchanged. Produces a `ReflectDecision`. **Does not branch on the decision** — branching moves to `route_after_reflect`, a conditional edge function in `graph.py`. |
| `v3/research_agent.py::final_answer()` | `finalize_node` | Called unchanged against the reconstructed `Trajectory`. |
| `research_agent.py`'s `for _ in range(max_steps)` loop + `if action.tool=="done": break` | `route_after_reflect` iteration cap + fixed `act_node -> observe_node` edge | See "done handling" and "iteration cap" below. |

### Bridging AthenaState <-> v3.Trajectory

v3's functions take/mutate a `v3.trajectory.Trajectory` (a Pydantic
`BaseModel`), not a LangGraph `TypedDict`. Rather than reimplement
`Trajectory.as_prompt_context()` and the notes-merging logic, each v4 node:

1. Rebuilds an ephemeral `Trajectory` from the incoming `AthenaState` fields
   (`_trajectory_from_state`).
2. Calls the unmodified v3 function, which reads/mutates that `Trajectory`.
3. Reads the (possibly mutated) `Trajectory` back into a **partial**
   `AthenaState` update dict, which LangGraph merges via the reducers below.

This keeps v3's prompt-construction, note-compression, and source-dedup logic
as the single source of truth, while keeping v4 nodes pure functions of
`AthenaState -> partial AthenaState`, as LangGraph expects.

### "done" handling — a deliberate behavior change from v3

`v3/research_agent.py`'s loop special-cases `action.tool == "done"`: it breaks
*before* calling `observe()` or `reflect()` at all. v4's graph has **fixed**
edges `act_node -> observe_node -> reflect_node`, so that shortcut does not
exist as a graph edge (the assignment explicitly forbids adding new edges
beyond the ones specified). Instead, v4 relies on a branch that already exists
in `v3/observer.py::observe()` but was dead code in v3 (`research_agent.py`
never reached it): when `action.tool == "done"`, `observe()` records the
observation `"No tool executed; actor selected done."`, still increments
`trajectory.iteration`, and returns. `reflect_node` then runs once more and
(per `v3/reflector.py`'s prompt, which is told to inspect `completed_steps`)
almost always returns `decision="done"`, which `route_after_reflect` sends to
`finalize_node`. Net effect: v4 costs one extra LLM call (`reflect`) compared
to v3 in the case where the actor volunteers "done" — traded for not needing a
second conditional edge out of `act_node`. This is called out here rather than
silently changing behavior.

### Iteration cap

v3 caps at `max_steps = min(max_steps, 8)` via `range(max_steps)` around the
whole act/observe/reflect body. v4 has no external loop — the cap is enforced
inside `route_after_reflect`: if `state["iteration"] >= 8`, route to
`finalize_node` regardless of what `reflect_node` decided. `state["iteration"]`
is written by `observe_node` (mirroring `trajectory.iteration += 1` in
`v3/observer.py`), so it only advances once per act/observe cycle, matching
v3's semantics exactly.

## AthenaState fields and reducers

`AthenaState` is a `TypedDict`. LangGraph's default merge behavior for a field
is **overwrite** (last node to touch it wins) unless the field's type is
wrapped in `Annotated[T, reducer]`, in which case LangGraph calls
`reducer(existing, update)` to combine values. See
https://docs.langchain.com/oss/python/langgraph/overview (State & reducers)
and https://docs.langchain.com/oss/python/langgraph/persistence (checkpointing
persists whatever `AthenaState` looks like after each super-step, so a missing
reducer is a silent history-loss bug, not just a style issue).

| Field | Type | Reducer | Accumulate? | Why |
|---|---|---|---|---|
| `messages` | `list[BaseMessage]` | `add_messages` (LangGraph built-in) | yes | Standard LangGraph message-history reducer (append-by-id, supports edits/removals). Used purely for the UI timeline (Task 7) — human question, plan summary, per-iteration observation summaries, reflection reasoning, final answer. v3 has no equivalent field; this is new surface area needed because LangGraph's `graph.stream()` / Streamlit UI wants a message-shaped timeline rather than a single mutable notes string. |
| `question` | `str` | none (replace) | no | Set once by the caller before `plan_node` runs; never changes during a run. Matches `Trajectory.question` (immutable for the life of one `research_agent()` call). |
| `plan` | `Plan \| None` | none (replace) | no | Mirrors `Trajectory.current_plan` — only the *current* plan is ever read by `act`/`reflect`/prompt context; old plans are not consulted, so overwrite is correct and matches v3 exactly. |
| `action` | `ActionDecision \| None` | none (replace) | no | Handoff slot for `act_node -> observe_node`: `act_node` produces one `ActionDecision`, `observe_node` executes exactly that one. v3 doesn't need this field because `act()` and `observe()` are called back-to-back in the same Python stack frame in `research_agent.py`; v4 splits them into separate graph nodes, so the decision has to travel through state instead of a local variable. |
| `notes` | `str` | none (replace) | no | Mirrors `Trajectory.notes`. v3 already does the accumulation *inside* `_merge_notes()` (string concatenation + a 5000-char tail-truncate) before assigning `trajectory.notes = ...`; the value handed back from `observe_node` is already the fully-merged string, so a reducer would double-merge it. Replace is correct given v3's merge-then-assign pattern. |
| `observations` | `Annotated[list[Observation], keep_recent_observations]` | custom: `(existing + new)[-2:]` | yes, bounded to 2 | v3's `Trajectory.add_observation()` explicitly keeps only the two most recent raw observations "preventing prompt context from growing unbounded" (`v3/trajectory.py:47`, `v3/research_agent.py` docstring in README). v4 **preserves this compression behavior** rather than accumulating full history, for the same reason v3 adopted it: raw tool output (especially notes-search chunks) is large and only the compressed `notes` string needs unbounded history. The reducer, not the node, enforces the cap so it holds even if a node is replayed/retried. |
| `completed_steps` | `Annotated[list[str], operator.add]` | append-all | yes, unbounded | Mirrors `Trajectory.completed_steps.append(...)` — v3 never truncates this list (it's short strings like `"search_notes: <reasoning>"`, bounded anyway by the 8-iteration cap), and both `act` and `reflect` prompts consult the full list. Full accumulation matches v3. |
| `notes_sources` | `Annotated[list[str], add_unique]` | dedup-append | yes | Mirrors `Trajectory.add_source("notes", ...)`, which only appends a citation if not already present. |
| `web_sources` | `Annotated[list[str], add_unique]` | dedup-append | yes | Same as above, for `Trajectory.add_source("web", ...)`. |
| `iteration` | `int` | none (replace) | no | Mirrors `Trajectory.iteration` — a counter that `observe_node` sets to `trajectory.iteration` (post-increment), not something that should be summed across updates. |
| `reflection` | `ReflectDecision \| None` | none (replace) | no | Transient per-cycle signal produced by `reflect_node` and consumed immediately by `route_after_reflect` (and, on the replan path, by `plan_node`). Only the latest reflection matters, so overwrite is correct; there is no v3 equivalent field (v3 keeps it as a local variable inside the loop) since v3 doesn't need to persist it across a node boundary. |
| `final_answer` | `str \| None` | none (replace) | no | Set once by `finalize_node`, mirrors the return value of `v3.research_agent.final_answer()`. |

`Observation` (`v4/state.py`) is a small Pydantic model
`{tool, raw, summary, source_type}`. v3 has no formal `Observation` type — the
raw string list `Trajectory.observations: List[str]` and the ad hoc dict
`{"raw": ..., "summary": ..., "source_type": ...}` returned by
`v3/observer.py::observe()` are the two places this shape already exists in
v3. `Observation` formalizes that existing ad hoc dict rather than forking any
Pydantic schema v3 already had (v3 never had a Pydantic model for it).

## Checkpointer

`SqliteSaver` (from `langgraph-checkpoint-sqlite`) is wired in from the first
commit that adds a checkpointer (Task 4) — not `MemorySaver` — so the resume
demo (Task 9) works without retrofitting. `SqliteSaver.from_conn_string(path)`
is a context manager (`Iterator[SqliteSaver]`), so both `v4/app.py` and the
resume-demo CLI open `v4/checkpoints.db` inside a `with` block for the
lifetime of one process; a second process opening the same path later sees
every checkpoint the first process committed, which is what makes
kill/restart resume work. Every node boundary is a checkpoint, so a process
killed after `act_node` resumes into `observe_node` on restart rather than
re-running `plan_node`.

## Human approval gate (Task 5)

The compiled graph uses `interrupt_before=["finalize_node"]`. This means the
interrupt fires *before* v3's `final_answer()` LLM call has run — the "draft
brief" shown to the user for approval is the accumulated `notes`,
`notes_sources`/`web_sources`, and `completed_steps` in `AthenaState`, not a
pre-written final answer (there isn't one yet). On approval, `graph.stream(None,
config)` resumes and lets `finalize_node` run normally. On reject-with-feedback,
the app calls `graph.update_state(config, {...replan ReflectDecision...},
as_node="reflect_node")` — this rewrites history as if `reflect_node` had just
produced a `replan` decision, so resuming re-enters `route_after_reflect` and
is sent to `plan_node` with the user's feedback appended to `messages` and
folded into the next planning prompt's context.

`interrupt_before` is used **only** for this approval gate. Progress
logging/live intermediate state for the UI (Task 7) uses `graph.stream(...)`
events instead, per the assignment's explicit instruction not to conflate the
two.

## Graph visualization (Task 6)

`graph.get_graph().draw_mermaid_png()` renders via the Mermaid.ink web API by
default (no local Graphviz needed) — LangGraph only falls back to a local
`pygraphviz`/Graphviz renderer if `draw_method=CURVED` or a local method is
explicitly requested. `make graph` uses the default (remote Mermaid) path, so
**no Graphviz install is required** locally or in CI; the only requirement is
outbound network access to `https://mermaid.ink` at build time. If that's ever
unavailable (offline CI), `draw_mermaid()` (text-only, no PNG) still works
without network access as a fallback source of truth.

## Out of scope this round

Retry edges, broader error-handling/reliability polish, and multi-agent
fan-out are explicitly out of scope per the assignment and are not added here
even though `act_node`/`observe_node` failures would be natural places for
them.
