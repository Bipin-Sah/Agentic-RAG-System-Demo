"""Run the fixed question set against the agent and the naive baseline; print a comparison.

python -m evals.run_eval --ingest            # ingest data/sample_corpus into schema "eval", run
python -m evals.run_eval --per-question --output eval_results/run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from tabulate import tabulate

from evals.scoring import (
    Question,
    QuestionScore,
    Summary,
    comparison_rows,
    error_score,
    load_questions,
    per_question_rows,
    score_answer,
    summarize,
)
from rag.answer import Answer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "evals" / "questions.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "sample_corpus"


class AnswerSystem(Protocol):
    def answer(self, question: str, history: list[dict] | None = None) -> Answer: ...


def run_system(
    system: AnswerSystem,
    questions: list[Question],
    on_result: Callable[[QuestionScore], None] | None = None,
) -> list[QuestionScore]:
    scores = []
    for q in questions:
        try:
            score = score_answer(q, system.answer(q.question))
        except Exception as exc:  # one failing question must not abort the whole run
            score = error_score(q, exc)
        scores.append(score)
        if on_result:
            on_result(score)
    return scores


def evaluate(
    systems: dict[str, AnswerSystem],
    questions: list[Question],
    on_result: Callable[[str, QuestionScore], None] | None = None,
) -> tuple[dict[str, list[QuestionScore]], list[Summary]]:
    results = {}
    for name, system in systems.items():
        callback = (lambda s, n=name: on_result(n, s)) if on_result else None
        results[name] = run_system(system, questions, callback)
    return results, [summarize(name, scores) for name, scores in results.items()]


def render_report(
    results: dict[str, list[QuestionScore]], summaries: list[Summary], per_question: bool
) -> str:
    parts = []
    if per_question:
        headers, rows = per_question_rows(results)
        parts.append(tabulate(rows, headers=headers, tablefmt="github"))
    headers, rows = comparison_rows(summaries)
    parts.append(tabulate(rows, headers=headers, tablefmt="github"))
    return "\n\n".join(parts)


def results_to_json(results: dict[str, list[QuestionScore]], summaries: list[Summary]) -> dict:
    def item(s: QuestionScore) -> dict:
        a = s.answer
        return {
            "id": s.question.id,
            "category": s.question.category,
            "question": s.question.question,
            "answer": a.answer if a else None,
            "correct": s.correct,
            "source_recall": s.source_recall,
            "retrieval_decision_ok": s.retrieval_decision_ok,
            "cited_sources": a.cited_sources if a else [],
            "searches": [q.query for q in a.searches] if a else [],
            "total_ms": s.latency_ms,
            "error": s.error,
        }

    return {
        "summaries": [asdict(s) for s in summaries],
        "results": {name: [item(s) for s in scores] for name, scores in results.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--schema",
        default=None,
        help="pgvector schema with the corpus (default: eval; eval_offline if RAG_OFFLINE=1)",
    )
    parser.add_argument("--ingest", action="store_true", help="ingest --corpus first (deduped)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--systems", default="agent,baseline", help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--per-question", action="store_true", help="also print per-question table")
    parser.add_argument("--output", type=Path, default=None, help="write full results as JSON")
    args = parser.parse_args(argv)

    from rag.config import Settings
    from rag.factory import (
        MissingAPIKey,
        build_agent,
        build_baseline,
        build_embedder,
        build_store,
        require_api_key,
    )
    from rag.ingest import Ingestor

    questions = load_questions(args.questions)[: args.limit]
    settings = Settings.from_env()
    builders = {"agent": build_agent, "baseline": build_baseline}
    names = [n.strip() for n in args.systems.split(",") if n.strip()]
    if not names:
        parser.error("--systems must name at least one of: agent, baseline")
    unknown = set(names) - set(builders)
    if unknown:
        parser.error(f"unknown systems: {', '.join(sorted(unknown))}")

    if not settings.offline:
        try:
            require_api_key(settings)
        except MissingAPIKey as exc:
            print(exc, file=sys.stderr)
            return 2
    args.schema = args.schema or ("eval_offline" if settings.offline else "eval")

    store = build_store(settings, schema=args.schema)
    if args.ingest:
        ingestor = Ingestor(
            store, build_embedder(settings), settings.chunk_size, settings.chunk_overlap
        )
        for r in ingestor.ingest_paths([args.corpus]):
            print(f"  {'skipped' if r.skipped else 'ingested'} {r.source} ({r.num_chunks} chunks)")
    if store.count_chunks() == 0:
        print(
            f"Schema '{args.schema}' is empty. Re-run with --ingest to load {args.corpus}.",
            file=sys.stderr,
        )
        return 1

    systems = {name: builders[name](settings, store) for name in names}

    def progress(name: str, s: QuestionScore) -> None:
        status = "ERR" if s.error else {True: "ok", False: "miss", None: "n/a"}[s.correct]
        print(f"  [{name}] {s.question.id:<28} {status:<4} {s.latency_ms:8,.0f} ms", flush=True)

    print(f"Running {len(questions)} questions against: {', '.join(names)}")
    start = time.perf_counter()
    results, summaries = evaluate(systems, questions, progress)
    print(f"\nFinished in {time.perf_counter() - start:.1f}s\n")
    print(render_report(results, summaries, args.per_question))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results_to_json(results, summaries), indent=2))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
