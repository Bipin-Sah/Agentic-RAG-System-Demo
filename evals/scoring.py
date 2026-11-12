"""Question set loading, per-answer scoring and per-system summaries."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from rag.answer import Answer

NO_RETRIEVAL = "no_retrieval"
UNANSWERABLE = "unanswerable"

_REFUSAL = re.compile(
    r"\b(?:"
    r"(?:do(?:es)?|did) ?n[o']t (?:contain|mention|include|say|specify|provide|have|cover|know)"
    r"|(?:no|not any) (?:information|mention|details?|data|record)"
    r"|(?:could|can|was) ?n[o']t (?:find|locate|determine|be found)"
    r"|(?:cannot|can not|unable to) (?:find|answer|determine|locate)"
    r"|not (?:mentioned|available|specified|provided|found|covered|stated|included)"
    r"|(?:i )?don[o']?t know"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    reference_answer: str = ""

    @property
    def should_retrieve(self) -> bool:
        return self.category != NO_RETRIEVAL


def load_questions(path: str | Path) -> list[Question]:
    questions = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append(Question(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid question record: {exc}") from exc
    if len({q.id for q in questions}) != len(questions):
        raise ValueError(f"{path}: duplicate question ids")
    return questions


def keyword_pass(answer: str, keyword_groups: list[str]) -> bool:
    """Every group must appear (case-insensitive); '|' separates alternatives within a group."""
    text = answer.lower()
    return all(
        any(alt.strip().lower() in text for alt in group.split("|") if alt.strip())
        for group in keyword_groups
    )


def source_matches(expected: str, cited: list[str]) -> bool:
    """Expected sources are file names; stored sources may carry a directory prefix."""
    return any(c == expected or c.endswith("/" + expected) for c in cited)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))


@dataclass
class QuestionScore:
    question: Question
    answer: Answer | None
    correct: bool | None  # None = not automatically gradable
    source_recall: float | None  # None = no expected sources
    retrieval_decision_ok: bool
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return self.answer.total_ms if self.answer else 0.0


def score_answer(question: Question, answer: Answer) -> QuestionScore:
    if question.category == UNANSWERABLE:
        correct: bool | None = is_refusal(answer.answer)
    elif question.expected_keywords:
        correct = keyword_pass(answer.answer, question.expected_keywords)
    else:
        correct = None

    recall = None
    if question.expected_sources:
        cited = answer.cited_sources
        recall = sum(source_matches(e, cited) for e in question.expected_sources) / len(
            question.expected_sources
        )

    return QuestionScore(
        question=question,
        answer=answer,
        correct=correct,
        source_recall=recall,
        retrieval_decision_ok=answer.used_retrieval == question.should_retrieve,
    )


def error_score(question: Question, error: Exception) -> QuestionScore:
    return QuestionScore(
        question=question,
        answer=None,
        correct=False,
        source_recall=0.0 if question.expected_sources else None,
        retrieval_decision_ok=False,
        error=f"{type(error).__name__}: {error}",
    )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


@dataclass
class Summary:
    system: str
    n: int
    accuracy: float | None
    graded: int
    source_recall: float | None
    retrieval_decision: float | None
    unanswerable_refusal: float | None
    avg_searches: float | None
    avg_llm_calls: float | None
    avg_tokens: float | None
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    errors: int


def summarize(system: str, scores: list[QuestionScore]) -> Summary:
    graded = [s for s in scores if s.correct is not None]
    recalls = [s.source_recall for s in scores if s.source_recall is not None]
    refusals = [s for s in scores if s.question.category == UNANSWERABLE]
    answered = [s.answer for s in scores if s.answer is not None]
    latencies = [a.total_ms for a in answered]
    return Summary(
        system=system,
        n=len(scores),
        accuracy=_mean([float(bool(s.correct)) for s in graded]),
        graded=len(graded),
        source_recall=_mean(recalls),
        retrieval_decision=_mean([float(s.retrieval_decision_ok) for s in scores]),
        unanswerable_refusal=_mean([float(bool(s.correct)) for s in refusals]),
        avg_searches=_mean([float(len(a.searches)) for a in answered]),
        avg_llm_calls=_mean([float(a.llm_calls) for a in answered]),
        avg_tokens=_mean([float(a.prompt_tokens + a.completion_tokens) for a in answered]),
        mean_latency_ms=_mean(latencies),
        p95_latency_ms=_p95(latencies),
        errors=sum(s.error is not None for s in scores),
    )


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def _num(v: float | None, fmt: str = "{:.1f}") -> str:
    return "-" if v is None else fmt.format(v)


METRIC_ROWS: list[tuple[str, str, object]] = [
    ("Answer accuracy (keyword/refusal)", "accuracy", _pct),
    ("Cited expected sources (recall)", "source_recall", _pct),
    ("Correct retrieve/skip decision", "retrieval_decision", _pct),
    ("Refused unanswerable questions", "unanswerable_refusal", _pct),
    ("Avg searches / question", "avg_searches", _num),
    ("Avg LLM calls / question", "avg_llm_calls", _num),
    ("Avg tokens / question", "avg_tokens", lambda v: _num(v, "{:,.0f}")),
    ("Mean latency (ms)", "mean_latency_ms", lambda v: _num(v, "{:,.0f}")),
    ("p95 latency (ms)", "p95_latency_ms", lambda v: _num(v, "{:,.0f}")),
    ("Errors", "errors", str),
]


def comparison_rows(summaries: list[Summary]) -> tuple[list[str], list[list[str]]]:
    headers = ["Metric", *(s.system for s in summaries)]
    rows = [
        [label, *(fmt(getattr(s, attr)) for s in summaries)] for label, attr, fmt in METRIC_ROWS
    ]
    return headers, rows


def per_question_rows(
    results: dict[str, list[QuestionScore]],
) -> tuple[list[str], list[list[str]]]:
    systems = list(results)
    headers = ["Question", "Category"]
    for name in systems:
        headers += [f"{name} correct", f"{name} sources", f"{name} searches", f"{name} ms"]
    rows = []
    for i, first in enumerate(results[systems[0]]):
        row = [first.question.id, first.question.category]
        for name in systems:
            s = results[name][i]
            mark = {True: "yes", False: "no", None: "-"}[s.correct]
            row += [
                "ERR" if s.error else mark,
                _pct(s.source_recall),
                str(len(s.answer.searches)) if s.answer else "-",
                f"{s.latency_ms:,.0f}",
            ]
        rows.append(row)
    return headers, rows
