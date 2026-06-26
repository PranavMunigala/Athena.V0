"""
Run Athena v1/v2 regression evaluations and upload experiments to LangSmith.

Usage:
    python evals/run.py --version wk4
    python evals/run.py --version wk5

Required environment:
    OPENAI_API_KEY for Athena and the judge model.
    LANGSMITH_API_KEY for dataset and experiment upload.
    LANGCHAIN_TRACING_V2=true to trace model calls during the eval run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from v1.main import query_athena_v1
from v2.chain import query_athena_v2


load_dotenv()


DATASET_PATH = ROOT_DIR / "evals" / "athena_v1.jsonl"
DATASET_NAME = os.getenv("ATHENA_EVAL_DATASET", "Athena RAG QA Fixtures")
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "athena-rag-evals")
JUDGE_MODEL = os.getenv("ATHENA_EVAL_JUDGE_MODEL", "gpt-4o-mini")
CITATION_RE = re.compile(r"\[[^\]]+\.pdf p\.\d+\]")
VERSION_TARGETS = {
    "wk4": ("v1", query_athena_v1),
    "wk5": ("v2", query_athena_v2),
}


def _configure_langsmith_tracing() -> None:
    """Enable modern LangSmith tracing when the requested v2 flag is present."""
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", PROJECT_NAME)
    os.environ.setdefault("LANGSMITH_PROJECT", PROJECT_NAME)


def load_fixtures(path: Path = DATASET_PATH) -> List[Dict[str, str]]:
    """Load JSONL fixtures with one QA expectation per line."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _get_or_create_dataset(client: Client, fixtures: List[Dict[str, str]]):
    """Create the LangSmith dataset once, then reuse it on later runs."""
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Athena v1/v2 RAG diagnostic fixtures.",
        )

    existing = list(client.list_examples(dataset_id=dataset.id))
    if not existing:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"query": row["query"]},
                    "outputs": {
                        "category": row["category"],
                        "expected_answer_substring": row["expected_answer_substring"],
                        "expected_source_file": row["expected_source_file"],
                    },
                }
                for row in fixtures
            ],
        )
    return dataset


def _answer_text(outputs: Dict[str, Any]) -> str:
    """Normalize target outputs for all evaluators."""
    return str(outputs.get("answer", ""))


def rigid_substring_and_citation(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic checks for refusal, required substrings, and citations."""
    answer = _answer_text(outputs)
    category = reference_outputs["category"]
    expected = reference_outputs["expected_answer_substring"]
    expected_source = reference_outputs.get("expected_source_file", "")

    if category in {"not in my notes", "needs web search"}:
        passed = answer.strip() == expected
        return {
            "key": "rigid_refusal",
            "score": int(passed),
            "comment": "Expected exact low-context refusal.",
        }

    has_substring = expected.lower() in answer.lower()
    citations = CITATION_RE.findall(answer)
    cites_expected_source = any(expected_source in citation for citation in citations)
    passed = has_substring and cites_expected_source
    return {
        "key": "rigid_grounded_answer",
        "score": int(passed),
        "comment": (
            f"substring={has_substring}; citations={citations}; "
            f"expected_source={expected_source}"
        ),
    }


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    """Accept strict JSON or a fenced JSON object from the judge model."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _get_judge() -> ChatOpenAI:
    """Construct the judge lazily so importing this module stays side-effect light."""
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def llm_judge(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    reference_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """LLM-as-judge check for semantic accuracy and structural adherence."""
    prompt = f"""
You are grading a RAG study assistant answer.

Question: {inputs["query"]}
Category: {reference_outputs["category"]}
Expected substring or exact refusal: {reference_outputs["expected_answer_substring"]}
Expected source file: {reference_outputs.get("expected_source_file", "")}
Actual answer: {_answer_text(outputs)}

Grade PASS only when:
- in-my-notes answers are semantically correct, grounded, and cite sources as [filename.pdf p.X]
- not-in-notes and needs-web-search answers are exactly "I don't see this in your notes."

Return compact JSON with keys "score" as 0 or 1 and "reason" as a short string.
"""
    try:
        raw = _get_judge().invoke(prompt).content
        parsed = _parse_judge_json(raw)
        score = int(parsed.get("score", 0))
        reason = str(parsed.get("reason", raw))
    except Exception as exc:
        score = 0
        reason = f"Judge failed or returned invalid JSON: {exc}"

    return {"key": "llm_judge", "score": score, "comment": reason}


def make_target(name: str, query_fn: Callable[[str], str]) -> Callable[[Dict[str, str]], Dict[str, str]]:
    """Wrap a plain Athena query function in the shape LangSmith evaluate expects."""

    @traceable(name=f"athena_{name}_target")
    def target(inputs: Dict[str, str]) -> Dict[str, str]:
        return {"answer": query_fn(inputs["query"])}

    return target


def run_experiment(
    client: Client,
    dataset_name: str,
    target_name: str,
    query_fn: Callable[[str], str],
):
    """Upload one named LangSmith experiment for a single Athena implementation."""
    return client.evaluate(
        make_target(target_name, query_fn),
        data=dataset_name,
        evaluators=[rigid_substring_and_citation, llm_judge],
        experiment_prefix=f"athena-{target_name}",
        description=f"Athena {target_name} RAG regression benchmark.",
        max_concurrency=1,
        metadata={
            "models": [JUDGE_MODEL],
            "target": target_name,
            "dataset_file": str(DATASET_PATH),
        },
    )


def _score_row(row: Dict[str, str], query_fn: Callable[[str], str]) -> Dict[str, Any]:
    """Run one fixture through the target and both evaluators."""
    inputs = {"query": row["query"]}
    reference_outputs = {
        "category": row["category"],
        "expected_answer_substring": row["expected_answer_substring"],
        "expected_source_file": row["expected_source_file"],
    }
    outputs = {"answer": query_fn(row["query"])}
    rigid = rigid_substring_and_citation(inputs, outputs, reference_outputs)
    judge = llm_judge(inputs, outputs, reference_outputs)
    total = (float(rigid["score"]) + float(judge["score"])) / 2.0
    return {
        "query": row["query"],
        "category": row["category"],
        "rigid": float(rigid["score"]),
        "judge": float(judge["score"]),
        "total": total,
        "answer": outputs["answer"],
        "rigid_comment": rigid["comment"],
        "judge_comment": judge["comment"],
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate evaluator scores overall and by category."""
    overall = {
        "rigid": sum(row["rigid"] for row in results) / len(results),
        "judge": sum(row["judge"] for row in results) / len(results),
        "total": sum(row["total"] for row in results) / len(results),
    }
    categories = {}
    for category in sorted({row["category"] for row in results}):
        rows = [row for row in results if row["category"] == category]
        categories[category] = {
            "count": len(rows),
            "rigid": sum(row["rigid"] for row in rows) / len(rows),
            "judge": sum(row["judge"] for row in rows) / len(rows),
            "total": sum(row["total"] for row in rows) / len(rows),
        }
    return {"overall": overall, "categories": categories}


def _pct(value: float) -> str:
    """Format a 0-1 score as a percentage."""
    return f"{value * 100:6.1f}%"


def _print_score_tables(version: str, results: List[Dict[str, Any]]) -> None:
    """Print per-example and aggregate score tables directly to stdout."""
    summary = _summarize(results)
    print(f"\nAthena {version.upper()} Evaluation Results")
    print("=" * 88)
    print(f"{'#':>2}  {'category':<18}  {'rigid':>8}  {'judge':>8}  {'total':>8}  query")
    print("-" * 88)
    for index, row in enumerate(results, start=1):
        print(
            f"{index:>2}  {row['category']:<18}  "
            f"{_pct(row['rigid']):>8}  {_pct(row['judge']):>8}  "
            f"{_pct(row['total']):>8}  {row['query'][:72]}"
        )

    print("\nCategory Summary")
    print("-" * 64)
    print(f"{'category':<18}  {'n':>3}  {'rigid':>8}  {'judge':>8}  {'total':>8}")
    for category, scores in summary["categories"].items():
        print(
            f"{category:<18}  {scores['count']:>3}  "
            f"{_pct(scores['rigid']):>8}  {_pct(scores['judge']):>8}  "
            f"{_pct(scores['total']):>8}"
        )

    overall = summary["overall"]
    print("-" * 64)
    print(
        f"{'OVERALL':<18}  {len(results):>3}  "
        f"{_pct(overall['rigid']):>8}  {_pct(overall['judge']):>8}  "
        f"{_pct(overall['total']):>8}"
    )


def _save_results(version: str, results: List[Dict[str, Any]]) -> Path:
    """Persist local results so separate wk4/wk5 invocations can be compared."""
    path = ROOT_DIR / "evals" / f"results_{version}.json"
    payload = {"version": version, "summary": _summarize(results), "results": results}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_saved(version: str) -> Optional[Dict[str, Any]]:
    """Load saved results from a previous run if present."""
    path = ROOT_DIR / "evals" / f"results_{version}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _explain_delta(current_version: str, current: Dict[str, Any], other: Dict[str, Any]) -> None:
    """Print a concise 5% delta comparison when both week outputs are available."""
    current_total = current["summary"]["overall"]["total"]
    other_total = other["summary"]["overall"]["total"]
    delta = current_total - other_total
    print("\nWeek 4 vs Week 5 Delta")
    print("-" * 64)
    if current_version == "wk5":
        print(f"wk4 total: {_pct(other_total)}")
        print(f"wk5 total: {_pct(current_total)}")
    else:
        print(f"wk4 total: {_pct(current_total)}")
        print(f"wk5 total: {_pct(other_total)}")
    print(f"absolute delta: {abs(delta) * 100:.1f} percentage points")

    if abs(delta) <= 0.05:
        print("Result: within the required 5% performance delta.")
        return

    wk4 = current if current_version == "wk4" else other
    wk5 = current if current_version == "wk5" else other
    category_deltas = {
        category: wk5["summary"]["categories"][category]["total"]
        - wk4["summary"]["categories"][category]["total"]
        for category in wk4["summary"]["categories"]
    }
    largest_category = min(category_deltas, key=lambda key: category_deltas[key])
    rigid_delta = wk5["summary"]["overall"]["rigid"] - wk4["summary"]["overall"]["rigid"]
    judge_delta = wk5["summary"]["overall"]["judge"] - wk4["summary"]["overall"]["judge"]
    likely_cause = "retrieval/refusal gating" if rigid_delta < judge_delta else "prompt wrapping or answer formatting"
    print(
        "Technical explanation: Week 5 falls outside the 5% tolerance because the "
        f"largest category delta is {largest_category} ({category_deltas[largest_category] * 100:.1f} "
        f"points), with rigid delta {rigid_delta * 100:.1f} points and judge delta "
        f"{judge_delta * 100:.1f} points; this points most directly to {likely_cause} "
        "rather than corpus chunking, since both versions query the same root Chroma "
        "collection and evaluation fixtures."
    )


def _maybe_upload_to_langsmith(
    fixtures: List[Dict[str, str]],
    version: str,
    target_name: str,
    query_fn: Callable[[str], str],
) -> None:
    """Upload a named LangSmith experiment when credentials are available."""
    if not os.getenv("LANGSMITH_API_KEY"):
        print("\nLangSmith upload: skipped because LANGSMITH_API_KEY is not set.")
        return

    client = Client()
    dataset = _get_or_create_dataset(client, fixtures)
    experiment = run_experiment(client, dataset.name, target_name, query_fn)
    print(f"\nLangSmith upload: complete for {version} ({experiment})")


def parse_args() -> argparse.Namespace:
    """Parse the Week 4/Week 5 target selector."""
    parser = argparse.ArgumentParser(description="Run Athena evals for one version.")
    parser.add_argument(
        "--version",
        choices=sorted(VERSION_TARGETS),
        required=True,
        help="wk4 evaluates v1; wk5 evaluates v2.",
    )
    parser.add_argument(
        "--skip-langsmith",
        action="store_true",
        help="Run local scoring only and do not upload a LangSmith experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """Load fixtures, run one target, print scores, and upload when configured."""
    args = parse_args()
    _configure_langsmith_tracing()
    fixtures = load_fixtures()
    target_name, query_fn = VERSION_TARGETS[args.version]

    results = [_score_row(row, query_fn) for row in fixtures]
    _print_score_tables(args.version, results)
    results_path = _save_results(args.version, results)
    print(f"\nSaved local results: {results_path}")

    other_version = "wk5" if args.version == "wk4" else "wk4"
    other = _load_saved(other_version)
    if other:
        _explain_delta(args.version, _load_saved(args.version), other)

    if not args.skip_langsmith:
        _maybe_upload_to_langsmith(fixtures, args.version, target_name, query_fn)


if __name__ == "__main__":
    main()
