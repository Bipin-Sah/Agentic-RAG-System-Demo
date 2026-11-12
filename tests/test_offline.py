import pytest

from rag.agent import RetrievalAgent
from rag.baseline import NaiveRAG
from rag.config import Settings
from rag.factory import build_chat, build_embedder
from rag.offline import HashingEmbedder, RuleChatClient
from rag.retriever import Retriever
from rag.types import RetrievedChunk
from tests.fakes import FakeStore

DOCS = [
    ("spec.docx", "The Zephyr X2 has a maximum payload of 2.3 kg. It weighs 4.1 kg."),
    ("spec.docx", "Flight time is 47 minutes with the standard battery."),
    ("handbook.pdf", "Employees receive 25 days of paid time off per year."),
]


@pytest.fixture
def retriever():
    emb = HashingEmbedder(dim=256)
    chunks = [
        (RetrievedChunk(i, i, src, 0, text, 0.0, None), emb.embed([text])[0])
        for i, (src, text) in enumerate(DOCS, 1)
    ]
    return Retriever(FakeStore(chunks), emb, top_k=2)


def test_agent_offline_searches_and_cites(retriever):
    answer = RetrievalAgent(RuleChatClient(), retriever).answer("What is the X2 maximum payload?")
    assert len(answer.searches) == 1
    assert "2.3 kg" in answer.answer and "[S1]" in answer.answer
    assert answer.citations[0].chunk.source == "spec.docx"


def test_agent_offline_skips_retrieval_for_greetings(retriever):
    answer = RetrievalAgent(RuleChatClient(), retriever).answer("Hi there!")
    assert answer.searches == [] and answer.answer.startswith("Hello")


def test_agent_offline_follow_up_folds_in_previous_question(retriever):
    history = [
        {"role": "user", "content": "Tell me about the Zephyr X2 payload"},
        {"role": "assistant", "content": "2.3 kg."},
    ]
    answer = RetrievalAgent(RuleChatClient(), retriever).answer(
        "What flight time does it have?", history=history
    )
    assert answer.searches[0].query.startswith("Tell me about the Zephyr X2")
    assert "47 minutes" in answer.answer


def test_offline_refuses_when_nothing_matches(retriever):
    answer = NaiveRAG(RuleChatClient(), retriever).answer("Quarterly revenue figures?")
    assert "couldn't find" in answer.answer and answer.citations == []


def test_baseline_offline_cites(retriever):
    answer = NaiveRAG(RuleChatClient(), retriever).answer("How many days of paid time off?")
    assert "25 days" in answer.answer and answer.citations


def test_settings_offline_flag(monkeypatch):
    monkeypatch.setenv("RAG_OFFLINE", "true")
    s = Settings.from_env(load_dotenv_file=False)
    assert s.offline
    assert isinstance(build_embedder(s), HashingEmbedder)
    assert isinstance(build_chat(s), RuleChatClient)
    assert build_embedder(s).dim == s.embed_dim
    monkeypatch.setenv("RAG_OFFLINE", "0")
    assert not Settings.from_env(load_dotenv_file=False).offline
