from rag.agent import SEARCH_TOOL_NAME, RetrievalAgent
from rag.llm import ChatResult, ToolCall
from rag.retriever import Retriever
from rag.types import RetrievedChunk
from tests.fakes import FakeStore, HashingEmbedder, ScriptedChatClient

DOCS = [
    ("handbook.pdf", 2, "Employees receive 25 vacation days per calendar year."),
    ("specs.docx", None, "The Zephyr X2 drone battery lasts 47 minutes."),
    ("faq.txt", None, "The cafeteria opens at 8am on weekdays."),
]


def make_agent(responses, max_searches=4):
    emb = HashingEmbedder(dim=64)
    chunks = [
        (
            RetrievedChunk(
                chunk_id=i + 1,
                document_id=i + 1,
                source=src,
                chunk_index=0,
                content=text,
                score=0.0,
                page=page,
            ),
            emb.embed([text])[0],
        )
        for i, (src, page, text) in enumerate(DOCS)
    ]
    store = FakeStore(chunks)
    chat = ScriptedChatClient(responses)
    agent = RetrievalAgent(chat, Retriever(store, emb, top_k=2), max_searches=max_searches)
    return agent, chat, store


def search(query, call_id="c1", k=None):
    args = {"query": query} if k is None else {"query": query, "k": k}
    return ToolCall(id=call_id, name=SEARCH_TOOL_NAME, arguments=args)


def test_answers_directly_without_retrieval():
    agent, chat, store = make_agent([ChatResult(content="Hello! How can I help?")])
    ans = agent.answer("hi there")
    assert ans.answer == "Hello! How can I help?"
    assert not ans.used_retrieval
    assert ans.citations == []
    assert store.queries == []
    assert ans.llm_calls == 1
    assert chat.requests[0]["tools"][0]["function"]["name"] == SEARCH_TOOL_NAME


def test_searches_then_answers_with_citations():
    def final(messages):
        tool_msg = messages[-1]
        assert tool_msg["role"] == "tool"
        assert "[S1] (handbook.pdf, p.2, chunk 0)" in tool_msg["content"]
        return ChatResult(content="You get 25 vacation days [S1].", prompt_tokens=50)

    agent, chat, _ = make_agent(
        [ChatResult(content=None, tool_calls=[search("vacation days")], prompt_tokens=10), final]
    )
    ans = agent.answer("How much vacation do I get?")
    assert ans.used_retrieval
    assert [s.query for s in ans.searches] == ["vacation days"]
    assert [(c.label, c.chunk.source, c.chunk.page) for c in ans.citations] == [
        ("S1", "handbook.pdf", 2)
    ]
    assert ans.cited_sources == ["handbook.pdf"]
    assert ans.llm_calls == 2
    assert ans.prompt_tokens == 60
    assert ans.total_ms >= ans.llm_ms + ans.retrieval_ms - 1
    # transcript has the assistant tool-call message before the tool result
    msgs = chat.requests[1]["messages"]
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-2]["tool_calls"][0]["function"]["name"] == SEARCH_TOOL_NAME


def test_multi_query_decomposition_labels_are_cumulative():
    agent, _, _ = make_agent(
        [
            ChatResult(
                content=None,
                tool_calls=[search("vacation days", "a", k=1), search("drone battery", "b", k=1)],
            ),
            ChatResult(content="25 days [S1]; 47 minutes [S2]. Bogus [S7]."),
        ]
    )
    ans = agent.answer("Vacation days and drone battery life?")
    assert len(ans.searches) == 2
    assert [c.chunk.source for c in ans.citations] == ["handbook.pdf", "specs.docx"]
    assert ans.answer == "25 days [S1]; 47 minutes [S2]. Bogus."
    assert len(ans.retrieved) == 2


def test_search_budget_forces_final_answer():
    responses = [ChatResult(content=None, tool_calls=[search(f"q{i}", f"c{i}")]) for i in range(2)]
    responses.append(ChatResult(content="Could not find it in the documents."))
    agent, chat, _ = make_agent(responses, max_searches=2)
    ans = agent.answer("Something obscure?")
    assert len(ans.searches) == 2
    assert [r["tool_choice"] for r in chat.requests] == ["auto", "auto", "none"]
    assert ans.answer == "Could not find it in the documents."


def test_extra_tool_calls_beyond_budget_are_refused():
    agent, chat, _ = make_agent(
        [
            ChatResult(content=None, tool_calls=[search("a", "1"), search("b", "2")]),
            ChatResult(content="done"),
        ],
        max_searches=1,
    )
    ans = agent.answer("q")
    assert len(ans.searches) == 1
    tool_msgs = [m for m in chat.requests[1]["messages"] if m["role"] == "tool"]
    assert "budget exhausted" in tool_msgs[1]["content"]


def test_bad_tool_calls_return_errors_to_model():
    agent, chat, _ = make_agent(
        [
            ChatResult(
                content=None,
                tool_calls=[
                    ToolCall(id="x", name="delete_everything", arguments={}),
                    ToolCall(id="y", name=SEARCH_TOOL_NAME, arguments={"query": "  "}),
                    ToolCall(id="z", name=SEARCH_TOOL_NAME, arguments={"query": "drone", "k": "?"}),
                ],
            ),
            ChatResult(content="ok [S1]"),
        ]
    )
    ans = agent.answer("q")
    contents = [m["content"] for m in chat.requests[1]["messages"] if m["role"] == "tool"]
    assert contents[0].startswith("Error: unknown tool")
    assert contents[1].startswith("Error: 'query' is required")
    assert "[S1]" in contents[2]
    assert len(ans.searches) == 1


def test_history_is_included_before_question():
    agent, chat, _ = make_agent([ChatResult(content="sure")])
    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "25 days [S1]."},
    ]
    agent.answer("follow-up", history=history)
    msgs = chat.requests[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "25 days."  # stale labels scrubbed
    assert history[1]["content"] == "25 days [S1]."  # caller's history untouched
    assert msgs[-1]["content"] == "follow-up"


def test_direct_answer_keeps_literal_bracketed_text():
    agent, _, _ = make_agent([ChatResult(content="In Markdown, [S1] is a reference link.")])
    assert agent.answer("What does [S1] mean?").answer == "In Markdown, [S1] is a reference link."
