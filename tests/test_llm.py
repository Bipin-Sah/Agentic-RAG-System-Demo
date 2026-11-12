import json
from types import SimpleNamespace

from rag.llm import ChatResult, OpenAIChatClient, ToolCall, assistant_message


class FakeCompletions:
    def __init__(self, message):
        self.message = message
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self.message)],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


def client_for(message):
    completions = FakeCompletions(message)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIChatClient(model="m", client=fake), completions


def test_parses_tool_calls_and_usage():
    tc = SimpleNamespace(
        id="t1", function=SimpleNamespace(name="search", arguments='{"query": "x"}')
    )
    client, completions = client_for(SimpleNamespace(content=None, tool_calls=[tc]))
    res = client.complete([{"role": "user", "content": "q"}], tools=[{"t": 1}], tool_choice="auto")
    assert res.tool_calls == [ToolCall(id="t1", name="search", arguments={"query": "x"})]
    assert (res.prompt_tokens, res.completion_tokens) == (7, 3)
    assert completions.kwargs["tool_choice"] == "auto"
    assert completions.kwargs["temperature"] == 0.0


def test_invalid_tool_json_becomes_empty_args_and_no_tools_kwarg():
    tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="search", arguments="{oops"))
    client, completions = client_for(SimpleNamespace(content="hi", tool_calls=[tc]))
    res = client.complete([])
    assert res.tool_calls[0].arguments == {}
    assert "tools" not in completions.kwargs


def test_assistant_message_roundtrip():
    msg = assistant_message(
        ChatResult(content=None, tool_calls=[ToolCall("i", "search", {"query": "q"})])
    )
    assert msg["tool_calls"][0]["id"] == "i"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"query": "q"}
    assert "tool_calls" not in assistant_message(ChatResult(content="x"))


def test_non_object_tool_args_become_empty_dict():
    calls = [
        SimpleNamespace(id=str(i), function=SimpleNamespace(name="search", arguments=raw))
        for i, raw in enumerate(["[]", "null", "3"])
    ]
    client, _ = client_for(SimpleNamespace(content=None, tool_calls=calls))
    assert [tc.arguments for tc in client.complete([]).tool_calls] == [{}, {}, {}]
