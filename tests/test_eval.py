import json

import pytest

from evals.run_eval import (
    DEFAULT_CORPUS,
    DEFAULT_QUESTIONS,
    evaluate,
    main,
    render_report,
    results_to_json,
    run_system,
)
from evals.scoring import (
    Question,
    comparison_rows,
    is_refusal,
    keyword_pass,
    load_questions,
    score_answer,
    source_matches,
    summarize,
)
from rag.agent import SEARCH_TOOL_NAME, RetrievalAgent
from rag.answer import Answer, SearchStep
from rag.baseline import NaiveRAG
from rag.chunking import chunk_pages
from rag.citations import Citation
from rag.ingest import expand_paths
from rag.llm import ChatResult, ToolCall
from rag.loaders import load_document
from rag.retriever import Retriever
from rag.types import RetrievedChunk
from tests.fakes import FakeStore, HashingEmbedder


def rc(source="handbook.pdf", chunk_id=1):
    return RetrievedChunk(chunk_id, 1, source, 0, "text", 0.9, None)


def ans(text, sources=(), searched=True, ms=100.0):
    return Answer(
        question="q",
        answer=text,
        citations=[Citation(f"S{i + 1}", rc(s, i)) for i, s in enumerate(sources)],
        searches=[SearchStep("q", 1, 1.0)] if searched else [],
        total_ms=ms,
        llm_calls=1,
        prompt_tokens=10,
        completion_tokens=5,
    )


def q(category="single_hop", keywords=("27",), sources=("employee_handbook.pdf",)):
    return Question("id", category, "question?", list(keywords), list(sources))


def test_fixed_question_set_is_valid():
    questions = load_questions(DEFAULT_QUESTIONS)
    assert len(questions) >= 10
    cats = {x.category for x in questions}
    assert cats == {"single_hop", "multi_hop", "no_retrieval", "unanswerable"}
    corpus = {p.name for p, _ in expand_paths([DEFAULT_CORPUS])}
    for x in questions:
        assert set(x.expected_sources) <= corpus, x.id
        if x.category in {"single_hop", "multi_hop"}:
            assert x.expected_keywords and x.expected_sources, x.id


def test_load_questions_rejects_bad_records(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"id": "a", "category": "x", "question": "?"}\n{"id": "a", "oops": 1}\n')
    with pytest.raises(ValueError, match=":2:"):
        load_questions(p)
    p.write_text('{"id": "a", "category": "x", "question": "?"}\n' * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(p)


def test_keyword_pass():
    assert keyword_pass("Mara LINDQVIST and Tomas Okafor", ["Lindqvist", "Okafor"])
    assert not keyword_pass("Mara Lindqvist", ["Lindqvist", "Okafor"])
    assert keyword_pass("within four hours", ["4 hours|four hours"])
    assert keyword_pass("anything", [])


@pytest.mark.parametrize(
    "text",
    [
        "I could not find that in the documents.",
        "The documents don't mention a Zephyr X3.",
        "There is no information about revenue in the sources.",
        "That is not specified in the provided documents.",
        "I couldn\u2019t find that in the documents.",
        "The documents don\u2019t mention it.",
    ],
)
def test_refusals_detected(text):
    assert is_refusal(text)


def test_answers_are_not_refusals():
    assert not is_refusal("The X2 costs $3,499 [S1].")
    assert not is_refusal("It costs $5,000. Ask if you need any information.")


def test_source_matches_ignores_directory_prefix():
    assert source_matches("a.pdf", ["data/sample_corpus/a.pdf"])
    assert source_matches("a.pdf", ["a.pdf"])
    assert not source_matches("a.pdf", ["data/ba.pdf"])


def test_score_answer_categories():
    s = score_answer(q(), ans("27 days [S1]", ["data/employee_handbook.pdf"]))
    assert s.correct and s.source_recall == 1.0 and s.retrieval_decision_ok

    s = score_answer(q(sources=("a.txt", "b.txt")), ans("26 days", ["a.txt"]))
    assert s.correct is False and s.source_recall == 0.5

    s = score_answer(q("no_retrieval", (), ()), ans("Hello!", searched=False))
    assert s.correct is None and s.source_recall is None and s.retrieval_decision_ok

    s = score_answer(q("no_retrieval", ("51",), ()), ans("51", searched=True))
    assert s.correct and not s.retrieval_decision_ok

    assert score_answer(q("unanswerable", (), ()), ans("I couldn't find it.")).correct
    assert not score_answer(q("unanswerable", (), ()), ans("It costs $5,000.")).correct


def test_summarize_and_table():
    scores = [
        score_answer(q(), ans("27", ["employee_handbook.pdf"], ms=100)),
        score_answer(q(), ans("wrong", [], ms=300)),
        score_answer(q("no_retrieval", (), ()), ans("hi", searched=False, ms=50)),
    ]
    s = summarize("agent", scores)
    assert s.n == 3 and s.graded == 2 and s.accuracy == 0.5
    assert s.source_recall == 0.5 and s.retrieval_decision == 1.0
    assert s.unanswerable_refusal is None
    assert s.mean_latency_ms == pytest.approx(150) and s.p95_latency_ms == 300
    assert s.avg_tokens == 15 and s.errors == 0
    headers, rows = comparison_rows([s, summarize("baseline", scores[:1])])
    assert headers == ["Metric", "agent", "baseline"]
    table = dict((r[0], r[1:]) for r in rows)
    assert table["Answer accuracy (keyword/refusal)"] == ["50%", "100%"]
    assert table["Refused unanswerable questions"] == ["-", "-"]


def test_run_system_records_errors_and_continues():
    class Flaky:
        calls = 0

        def answer(self, question, history=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("rate limited")
            return ans("27", ["employee_handbook.pdf"])

    scores = run_system(Flaky(), [q(), q()])
    assert scores[0].error == "RuntimeError: rate limited" and scores[0].correct is False
    assert scores[1].correct
    assert summarize("x", scores).errors == 1


class RuleChat:
    """Offline stand-in for the LLM: searches for document questions, quotes the top source."""

    def complete(self, messages, tools=None, tool_choice=None):
        last = messages[-1]
        if last["role"] == "tool" or "Sources:" in last["content"]:
            body = last["content"]
            if "[S1]" not in body:
                return ChatResult(content="I could not find that in the documents.")
            top = body.split("[S1]", 1)[1].split("\n[S2]", 1)[0]
            return ChatResult(content=f"{top.strip()[:400]} [S1]", prompt_tokens=100)
        question = last["content"]
        if "multiplied" in question:
            return ChatResult(content="51")
        if question.startswith("Hi"):
            return ChatResult(content="I answer questions about your documents.")
        call = ToolCall("c1", SEARCH_TOOL_NAME, {"query": question})
        return ChatResult(content=None, tool_calls=[call], prompt_tokens=50)


def sample_corpus_store(embedder):
    chunks, cid = [], 0
    for path, source in expand_paths([DEFAULT_CORPUS]):
        for c in chunk_pages(load_document(path), 800, 150):
            cid += 1
            chunk = RetrievedChunk(cid, 1, source, c.chunk_index, c.content, 0.0, c.page)
            chunks.append((chunk, embedder.embed([c.content])[0]))
    return FakeStore(chunks)


def test_offline_end_to_end_comparison():
    embedder = HashingEmbedder(dim=256)
    retriever = Retriever(sample_corpus_store(embedder), embedder, top_k=3)
    systems = {
        "agent": RetrievalAgent(RuleChat(), retriever, default_k=3),
        "baseline": NaiveRAG(RuleChat(), retriever, top_k=3),
    }
    questions = load_questions(DEFAULT_QUESTIONS)
    results, summaries = evaluate(systems, questions)
    agent, baseline = summaries
    assert agent.errors == baseline.errors == 0
    assert agent.retrieval_decision == 1.0  # skipped the 2 no-retrieval questions
    assert baseline.retrieval_decision == pytest.approx(12 / 14)
    assert agent.avg_llm_calls > baseline.avg_llm_calls == 1.0

    report = render_report(results, summaries, per_question=True)
    assert "| Metric" in report and "agent" in report and "baseline" in report
    assert "pto_days" in report
    data = json.loads(json.dumps(results_to_json(results, summaries)))
    assert len(data["results"]["agent"]) == len(questions)


def test_main_without_api_key_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert main(["--limit", "1"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_rejects_unknown_system(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(SystemExit):
        main(["--systems", "agent,magic"])
