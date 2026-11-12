from rag.baseline import NaiveRAG
from rag.llm import ChatResult
from rag.retriever import Retriever
from tests.fakes import ScriptedChatClient
from tests.test_agent import make_agent


def make_baseline(responses):
    agent, _, store = make_agent([])
    chat = ScriptedChatClient(responses)
    return NaiveRAG(chat, Retriever(store, agent.retriever.embedder, top_k=2), top_k=2), chat


def test_always_retrieves_once_and_cites():
    rag, chat = make_baseline([ChatResult(content="25 days [S1] [S9].", prompt_tokens=30)])
    ans = rag.answer("How many vacation days?")
    assert len(ans.searches) == 1 and ans.searches[0].query == "How many vacation days?"
    assert ans.llm_calls == 1 and len(chat.requests) == 1
    assert chat.requests[0]["tools"] is None
    assert "[S1] (handbook.pdf, p.2, chunk 0)" in chat.requests[0]["messages"][-1]["content"]
    assert ans.answer == "25 days [S1]."
    assert [c.chunk.source for c in ans.citations] == ["handbook.pdf"]
    assert ans.prompt_tokens == 30


def test_retrieves_even_for_greetings():
    rag, _ = make_baseline([ChatResult(content="Hello!")])
    ans = rag.answer("hi")
    assert ans.used_retrieval
    assert ans.answer == "Hello!"


def test_history_labels_scrubbed():
    rag, chat = make_baseline([ChatResult(content="ok")])
    rag.answer("and?", history=[{"role": "assistant", "content": "25 days [S1]."}])
    assert chat.requests[0]["messages"][1] == {"role": "assistant", "content": "25 days."}
