"""Offline UI tests: drive the Streamlit script with injected fake components."""

from datetime import datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.services import Components
from rag.agent import SEARCH_TOOL_NAME, RetrievalAgent
from rag.baseline import NaiveRAG
from rag.llm import ChatResult, ToolCall
from rag.retriever import Retriever
from rag.store import DocumentInfo
from tests.fakes import ScriptedChatClient
from tests.test_agent import make_agent

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


class DocStore:
    def __init__(self, fake_store):
        self.fake = fake_store
        now = datetime.now(timezone.utc)
        self.docs = [DocumentInfo(1, "handbook.pdf", "sha", 3, now)]

    def search(self, embedding, k=5):
        return self.fake.search(embedding, k)

    def list_documents(self):
        return list(self.docs)

    def count_chunks(self):
        return sum(d.num_chunks for d in self.docs)

    def delete_document(self, doc_id):
        self.docs = [d for d in self.docs if d.id != doc_id]


def components(agent_responses, baseline_responses=()):
    agent, _, fake = make_agent([])
    store = DocStore(fake)
    retriever = Retriever(store, agent.retriever.embedder, top_k=2)
    return Components(
        store=store,
        ingestor=None,
        systems={
            "Agentic RAG": RetrievalAgent(ScriptedChatClient(agent_responses), retriever),
            "Naive RAG (baseline)": NaiveRAG(ScriptedChatClient(baseline_responses), retriever),
        },
    )


def run_app(comps):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["components"] = comps
    return at.run()


def test_renders_documents_and_modes():
    at = run_app(components([]))
    assert not at.exception
    assert at.title[0].value == "Agentic RAG"
    assert "1 documents · 3 chunks" in at.sidebar.caption[0].value
    assert at.sidebar.radio[0].options == ["Agentic RAG", "Naive RAG (baseline)"]


def test_agent_answer_shows_citations_and_latency():
    call = ToolCall("c1", SEARCH_TOOL_NAME, {"query": "vacation days"})
    comps = components(
        [
            ChatResult(content=None, tool_calls=[call]),
            ChatResult(content="You get 25 vacation days [S1].", prompt_tokens=40),
        ]
    )
    at = run_app(comps)
    at.chat_input[0].set_value("How many vacation days?").run()
    assert not at.exception
    markdown = " ".join(m.value for m in at.markdown)
    assert "You get 25 vacation days [S1]." in markdown
    assert "Agentic RAG · total" in markdown and "1 search" in markdown
    assert "handbook.pdf" in markdown
    assert [e.label for e in at.expander] == ["Sources (1)", "Retrieval trace (1 searches)"]
    assert len(at.session_state["latency"]) == 1
    assert at.dataframe  # latency table rendered
    assert [m.label for m in at.metric] == ["Last", "Mean", "p50", "p95"]


def test_baseline_mode_and_follow_up_history():
    comps = components(
        [ChatResult(content="Hello!")],
        [ChatResult(content="Battery lasts 47 minutes [S1].")],
    )
    at = run_app(comps)
    at.chat_input[0].set_value("hi").run()
    at.sidebar.radio[0].set_value("Naive RAG (baseline)").run()
    at.chat_input[0].set_value("battery?").run()
    assert not at.exception
    records = at.session_state["latency"]
    assert [r.mode for r in records] == ["Agentic RAG", "Naive RAG (baseline)"]
    assert records[0].searches == 0 and records[1].searches == 1
    baseline_chat = comps.systems["Naive RAG (baseline)"].chat
    sent = baseline_chat.requests[0]["messages"]
    assert [m["content"] for m in sent[1:3]] == ["hi", "Hello!"]


def test_delete_document_and_clear_chat():
    comps = components([ChatResult(content="Hello!")])
    at = run_app(comps)
    at.chat_input[0].set_value("hi").run()
    at.sidebar.button(key="del-1").click().run()
    assert comps.store.docs == []
    clear = next(b for b in at.sidebar.button if b.label == "Clear chat")
    clear.click().run()
    assert at.session_state["messages"] == [] and at.session_state["latency"] == []


def test_setup_error_is_shown_without_crashing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "OPENAI_API_KEY" in at.error[0].value
